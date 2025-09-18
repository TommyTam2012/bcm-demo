# app.py  — TAEASLA API (persisted DB + startup seed + enrollment + HeyGen)

from fastapi import FastAPI, HTTPException, Query, Security, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import sqlite3
import csv
import io
import time
import httpx
import json
import re  # <-- TTS normalization uses regex
import ssl, smtplib
from email.message import EmailMessage
from sqlite3 import Row
import shutil  # <-- for DB migration copy

# --- App & basic setup ---
APP_DIR = Path(__file__).parent.resolve()

# Prefer persistent disk at /data; else keep local file.
DATA_DIR = Path("/data")
LEGACY_DB = APP_DIR / "bcm_demo.db"
PERSIST_DB = DATA_DIR / "bcm_demo.db"
if DATA_DIR.is_dir():
    if LEGACY_DB.exists() and not PERSIST_DB.exists():
        try:
            PERSIST_DB.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(LEGACY_DB), str(PERSIST_DB))
        except Exception:
            pass
DB_PATH = str(PERSIST_DB if PERSIST_DB.exists() or DATA_DIR.is_dir() else LEGACY_DB)

app = FastAPI(
    title="TAEASLA API",
    version="1.7.2",
    description="TAEASLA backend: courses, enrollments, fees, schedules, HeyGen proxy, and email alerts.",
)

# Static files
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# --- CORS ---
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB helpers ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                fee REAL NOT NULL,
                start_date TEXT,
                end_date TEXT,
                time TEXT,
                venue TEXT,
                seats INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                program_code TEXT,
                cohort_code TEXT,
                timezone TEXT,
                notes TEXT,
                source TEXT,
                course_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                q TEXT,
                a TEXT
            )""")
        conn.commit()

# ---------- Seed courses ----------
BCM_SEED = [
    ("自然拼讀", 0.0, 30),
    ("拼寫", 0.0, 30),
    ("文法", 0.0, 30),
    ("青少年雅思", 0.0, 25),
    ("呈分試", 0.0, 40),
    ("香港 Band 1 入學考試", 0.0, 40),
    ("香港中學文憑試", 0.0, 35),
    ("雅思", 0.0, 35),
    ("托福", 0.0, 35),
]

def seed_if_empty():
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM courses").fetchone()
        if row and int(row["c"] or 0) == 0:
            for (name_zh, fee, seats) in BCM_SEED:
                conn.execute(
                    "INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats) "
                    "VALUES (?, ?, NULL, NULL, NULL, NULL, ?)",
                    (name_zh, float(fee), int(seats)),
                )
            conn.commit()
# --- Startup: ensure schema + auto-seed ----------
@app.on_event("startup")
def _startup():
    init_db()
    seed_if_empty()

# --- Basic routes ---
@app.get("/")
def root():
    return RedirectResponse(url="/static/enroll.html")

# --- Health check ---
@app.api_route("/health", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
@app.api_route("/health/", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={"Allow": "GET, HEAD, OPTIONS"})
    return {"ok": True}

# --- TAEASLA assistant rules ---
ENROLL_LINK = "/static/enroll.html"
TAEASLA_RULES = (
    "You are the TAEASLA assistant. Follow these rules strictly: "
    "1) Identity: Introduce yourself as school assistant. "
    "2) Scope: Answer only about TAEASLA courses, fees, schedule, enrollment. "
    "3) No Hallucination. "
    "4) Forbidden: Do not mention IELTS or non-TAEASLA courses. "
    "5) Tone: Warm, professional, helpful. "
    "6) Enrollment Step: After each answer, ask 'Would you like to enroll?'. "
    "7) Yes → 'Please click the enrollment form link.' "
    "8) No → 'Okay, let me know if you have more questions.' "
    "9) Off-topic → 'I can only answer TAEASLA-related questions.' "
    "10) Keep answers 1–3 sentences."
)

@app.get("/assistant/intro")
def assistant_intro():
    return {"intro": "Hello, I’m the TAEASLA assistant. I can answer about fees, schedule, and our latest courses."}

@app.get("/assistant/prompt")
def assistant_prompt():
    return {"prompt": TAEASLA_RULES, "enroll_link": ENROLL_LINK}

# --- FAQ ---
@app.get("/faq")
def get_faq() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, q, a FROM faq ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

class FAQIn(BaseModel):
    q: str
    a: str

@app.post("/faq", dependencies=[Security(APIKeyHeader(name='X-Admin-Key', auto_error=False))], tags=["admin"])
def add_faq(item: FAQIn):
    with get_db() as conn:
        cur = conn.execute("INSERT INTO faq (q, a) VALUES (?, ?)", (item.q, item.a))
        conn.commit()
        fid = cur.lastrowid
        row = conn.execute("SELECT id, q, a FROM faq WHERE id = ?", (fid,)).fetchone()
        return dict(row)

# --- Fees ---
@app.get("/fees/{program_code}")
def get_fees(program_code: str):
    code = (program_code or "").upper()
    mapping = {
        "GI":    {"program": "TAEASLA General English (GI)", "fee": 8800, "currency": "HKD"},
        "HKDSE": {"program": "TAEASLA HKDSE English",        "fee": 7600, "currency": "HKD"},
    }
    if code not in mapping:
        raise HTTPException(status_code=404, detail="Program not found")
    return mapping[code]

# --- Schedule ---
@app.get("/schedule")
def schedule(season: Optional[str] = None):
    if (season or "").lower() == "summer":
        return [{
            "course": "TAEASLA Summer Intensive",
            "weeks": 6,
            "days": ["Monday", "Wednesday", "Friday"],
            "time": "Mon/Wed/Fri 7–9pm",
        }]
    return []

# --- Admin check ---
@app.get("/admin/check", tags=["admin"])
def admin_check():
    return {"ok": True, "message": "Admin access confirmed."}

# =========================================================
# === HeyGen CONFIG + TOKEN/START + SAY + PROXY + STOP ====
# =========================================================
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY") or os.getenv("ADMIN_KEY")
HEYGEN_BASE = "https://api.heygen.com/v1"

def _ok_json_or_raise(r: httpx.Response):
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            raise HTTPException(502, f"Non-JSON body from HeyGen: {r.text[:500]}")
    raise HTTPException(r.status_code, r.text)

def _get_avatar_id() -> str:
    return (
        os.getenv("HEYGEN_AVATAR_ID_Alessandra")
        or os.getenv("HEYGEN_AVATAR_ID")
        or "Alessandra_ProfessionalLook_public"
    )

@app.post("/heygen/token")
async def heygen_token():
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")

    AVATAR_ID = _get_avatar_id()
    new_url = f"{HEYGEN_BASE}/streaming.new"
    start_url = f"{HEYGEN_BASE}/streaming.start"
    headers = {"X-Api-Key": HEYGEN_API_KEY, "Accept": "application/json", "Content-Type": "application/json"}
    new_body = {"avatar_id": AVATAR_ID, "quality": "high", "version": "v2"}

    async with httpx.AsyncClient(timeout=25.0) as client:
        r_new = await client.post(new_url, headers=headers, json=new_body)
        j_new = _ok_json_or_raise(r_new)
        d = j_new.get("data") or j_new
        session_id = d.get("session_id")
        if not session_id:
            raise HTTPException(502, "streaming.new did not return session_id")

        r_start = await client.post(start_url, headers=headers, json={"session_id": session_id})
        if r_start.status_code != 200 and "already" not in r_start.text.lower():
            raise HTTPException(r_start.status_code, r_start.text)

        return {"code": j_new.get("code", 100), "data": d, "started": True}
# =========================================================
# ================== COURSES & ENROLLMENT =================
# =========================================================
class CourseIn(BaseModel):
    name: str = Field(..., description="Course name (displayed in Chinese for BCM drop-down)")
    fee: float = Field(..., description="Fee amount (numeric)")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    time: Optional[str] = None
    venue: Optional[str] = None
    seats: Optional[int] = Field(0, ge=0)

# --- TTS helper ---
def tts_friendly_time(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    day_map = {
        "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
        "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
    }
    t = t.replace("/", ", ")
    for abbr, full in day_map.items():
        t = re.sub(rf"\b{abbr}\b", full, t)
    t = re.sub(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(am|pm|a\.?m\.?|p\.?m\.?)?", 
               lambda m: f"{m.group(1)} {m.group(3) or ''} to {m.group(2)} {m.group(3) or ''}", t, flags=re.I)
    t = re.sub(r"(\d)(am|pm)\b", r"\1 \2", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

# -------------------- ADMIN: add/list/delete/export --------------------
@app.post("/admin/courses", tags=["admin"])
def admin_add_course(course: CourseIn):
    return add_course(course)

@app.post("/courses", tags=["courses"])
def add_course(course: CourseIn):
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats) "
            "VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 0))",
            (course.name, course.fee, course.start_date, course.end_date, course.time, course.venue, course.seats),
        )
        course_id = cur.lastrowid
        row = conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
        conn.commit()
        return dict(row)

@app.get("/courses")
def list_courses() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM courses ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

@app.post("/courses/seed")
def seed_courses(reset: bool = True):
    with get_db() as conn:
        if reset:
            conn.execute("DELETE FROM courses")
        for (name_zh, fee, seats) in BCM_SEED:
            conn.execute("INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats) "
                         "VALUES (?, ?, NULL, NULL, NULL, NULL, ?)", (name_zh, float(fee), int(seats)))
        conn.commit()
    return {"seeded": len(BCM_SEED), "reset": bool(reset)}

@app.get("/courses/options")
def course_options() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, name, seats FROM courses ORDER BY id ASC").fetchall()
        return [{"id": r["id"], "name": r["name"], "seats": int(r["seats"] or 0)} for r in rows]

@app.get("/courses/{course_id}")
def get_course(course_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Course not found")
        return dict(row)

@app.delete("/courses/{course_id}", tags=["admin"])
def delete_course(course_id: int):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Course not found")
        return {"ok": True, "deleted": course_id}

# --- Export CSV ---
@app.get("/courses/export.csv")
def export_courses_csv():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM courses ORDER BY id DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([c[0] for c in rows[0].keys()] if rows else [])
    for r in rows:
        writer.writerow([r[c] for c in r.keys()])
    return Response(content=output.getvalue(), media_type="text/csv")

# =========================
# Email Notification Config
# =========================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "tommytam2012@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS")  # Gmail App Password
SMTP_TO   = os.getenv("SMTP_TO", "tommytam2012@gmail.com")

def send_enroll_email(name: str, email: str, phone: str, notes: str):
    if not SMTP_PASS:
        return
    msg = EmailMessage()
    msg["Subject"] = f"新入学申请通知 - {name}"
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_TO
    msg["Reply-To"] = email or SMTP_USER
    body = f"新入学申请：\n姓名：{name}\n电邮：{email or '-'}\n电话：{phone or '-'}\n备注：{notes or '-'}"
    msg.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

# --- Enrollment ---
class EnrollmentIn(BaseModel):
    full_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    program_code: Optional[str] = None
    cohort_code: Optional[str] = None
    timezone: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "web"
    course_id: Optional[int] = None

@app.post("/enroll", tags=["enroll"])
def enroll(data: EnrollmentIn, background_tasks: BackgroundTasks):
    full_name_val = (data.full_name or data.name or "").strip()
    if not full_name_val:
        raise HTTPException(status_code=422, detail="full_name or name is required")

    with get_db() as conn:
        if data.course_id:
            row = conn.execute("SELECT id, seats FROM courses WHERE id = ?", (int(data.course_id),)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Selected course not found")
        else:
            row = conn.execute("SELECT id, seats FROM courses ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No course available")

        if int(row["seats"] or 0) <= 0:
            return {"ok": False, "message": "Sorry, this course is full."}

        conn.execute("UPDATE courses SET seats = seats - 1 WHERE id = ? AND seats > 0", (row["id"],))
        conn.execute(
            "INSERT INTO enrollments (full_name, email, phone, program_code, cohort_code, timezone, notes, source, course_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (full_name_val, data.email, data.phone, data.program_code, data.cohort_code,
             data.timezone, data.notes, data.source, int(row["id"])),
        )
        conn.commit()

    background_tasks.add_task(send_enroll_email, full_name_val, data.email or "", data.phone or "", data.notes or "")
    return {"ok": True, "message": "Enrollment confirmed. Seat deducted.", "course_id": int(row["id"])}

@app.get("/enrollments/recent", tags=["admin"])
def recent_enrollments(limit: int = Query(10, ge=1, le=100)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, full_name, email, phone, program_code, cohort_code, timezone, notes, source, course_id, created_at "
            "FROM enrollments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
# --- Assistant: TAEASLA rule-based answers ---
class UserQuery(BaseModel):
    text: str

def _latest_course_summary() -> str:
    with get_db() as conn:
        row: Optional[Row] = conn.execute(
            "SELECT name, fee, start_date, end_date, time, venue, seats FROM courses ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return "We currently have no courses listed."
    parts = [f"Latest course: {row['name']}, fee {row['fee']}."]
    if row["start_date"] and row["end_date"]:
        parts.append(f"Runs {row['start_date']} to {row['end_date']}.")
    if row["time"]:
        parts.append(f"Time: {tts_friendly_time(str(row['time']))}.")
    if row["venue"]:
        parts.append(f"Venue: {row['venue']}.")
    if "seats" in row.keys() and row["seats"] is not None and int(row["seats"]) > 0:
        parts.append(f"Seats left: {int(row['seats'])}.")
    return " ".join(parts)

def _taeasla_answer_from_db(q: str) -> str:
    ql = (q or "").strip().lower()
    if "ielts" in ql:
        return "I can only answer TAEASLA questions."
    fee_words = ("fee","price","cost","tuition","學費","費用")
    sched_words = ("schedule","time","timetable","summer","暑期","夏天")
    course_words = ("course","courses","class","classes","課程","班")
    enroll_words = ("enroll","enrol","sign up","報名","登記","註冊")

    if any(w in ql for w in fee_words):
        try:
            gi = get_fees("GI")
            return f"{gi['program']} costs {gi['currency']} {gi['fee']}."
        except:
            return "I don't know the answer to that."
    if any(w in ql for w in sched_words):
        s = schedule(season="summer")
        if s:
            d = s[0]
            days = ", ".join(d.get("days", []))
            return f"{d['course']}: {d['weeks']} weeks, days: {days}, time: {tts_friendly_time(d['time'])}."
    if any(w in ql for w in course_words):
        return _latest_course_summary()
    if any(w in ql for w in enroll_words):
        return "You can enroll online."
    return "I can only answer TAEASLA-related questions."

def _is_yes(q: str) -> bool:
    return q.strip().lower() in {"yes","yeah","ok","okay","sure","please","好的","要","係","好","是","行","可以"}

def _is_no(q: str) -> bool:
    return q.strip().lower() in {"no","nope","nah","不用","唔要","不要","否"}

@app.post("/assistant/answer")
def assistant_answer(payload: UserQuery):
    user_text = (payload.text or "").strip()
    if _is_yes(user_text):
        return {"reply": "Please click the enrollment form link.", "enroll_link": ENROLL_LINK}
    if _is_no(user_text):
        return {"reply": "Okay, let me know if you have more questions."}
    base = _taeasla_answer_from_db(user_text)
    reply = f"{base} Would you like to enroll?"
    return {"reply": reply, "enroll_hint": f"If yes, please click: {ENROLL_LINK}"}

# ============================
# === A L E S S A N D R A  ===
# ============================
@app.get("/hotel/ping")
def hotel_ping(who: str = "guest"):
    return {"ok": True, "who": who, "ts": int(time.time())}

@app.get("/heygen/avatar_id")
def heygen_avatar_id():
    return {"avatar_id": _get_avatar_id()}

@app.get("/heygen/avatars/interactive")
async def heygen_list_interactive(q: Optional[str] = None):
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY missing")
    V2_BASE = "https://api.heygen.com/v2"
    url = f"{V2_BASE}/avatars"
    headers = {"Accept": "application/json", "X-Api-Key": HEYGEN_API_KEY}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        return Response(content=r.text, status_code=r.status_code, media_type="application/json")
    j = r.json()
    avatars = (j or {}).get("data", {}).get("avatars", []) or []
    slim = [{"avatar_id": a.get("avatar_id") or a.get("id"), "avatar_name": a.get("avatar_name") or a.get("name")} for a in avatars if a.get("avatar_id") or a.get("id")]
    if q:
        slim = [row for row in slim if q.lower() in (row["avatar_name"] or "").lower()]
    return {"ok": True, "count": len(slim), "avatars": slim}

@app.get("/stream")
def stream():
    async def gen():
        yield b"data: hello\n\n"
        time.sleep(0.5)
        yield b"data: world\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

# --- Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
