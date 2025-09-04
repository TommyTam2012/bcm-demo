from fastapi import FastAPI, HTTPException, Query, Security, Request
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

# Optional OpenAI import (safe if package not installed)
try:
    from openai import OpenAI  # noqa: F401
except Exception:
    OpenAI = None  # type: ignore

# --- App & basic setup ---
APP_DIR = Path(__file__).parent.resolve()
DB_PATH = str(APP_DIR / "bcm_demo.db")

app = FastAPI(
    title="BCM Demo API",
    version="1.2.0",
    description="Backend for BCM demo: courses, enrollments, fees, schedules, and HeyGen token/proxy.",
)

# Static files
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# --- CORS ---
origins = [
    "https://bcmavatar.vercel.app",  # your Vercel frontend
    "http://localhost:3000",         # local dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB helpers ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any((c[1] == column) for c in cols)
    except Exception:
        return False


def init_db():
    with get_db() as conn:
        # Base tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                fee REAL NOT NULL,
                start_date TEXT,
                end_date TEXT,
                time TEXT,
                venue TEXT
            )
            """
        )
        conn.execute(
            """
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                q TEXT,
                a TEXT
            )
            """
        )

        # --- Schema migration: add seats column if missing ---
        if not _column_exists(conn, "courses", "seats"):
            conn.execute("ALTER TABLE courses ADD COLUMN seats INTEGER DEFAULT 0")

        conn.commit()


init_db()

# --- Admin key guard ---
ADMIN_KEY = os.getenv("ADMIN_KEY") or os.getenv("VITE_BCM_ADMIN_KEY")
api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin(x_admin_key: str = Security(api_key_header)):
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not configured on server")
    if not x_admin_key:
        raise HTTPException(status_code=403, detail="Forbidden (no X-Admin-Key header received)")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden (bad X-Admin-Key)")
    return True


# --- Basic routes ---
@app.get("/")
def root():
    return RedirectResponse(url="/static/enroll.html")


@app.get("/health")
def health():
    return {"ok": True}


# --- BCM assistant: fixed intro + hard rules (expanded) ---
ENROLL_LINK = "/static/enroll.html"  # adjust if your path differs

BCM_RULES = (
    "You are the BCM assistant. Follow these rules strictly: "
    "1) Identity: Speak as 'the BCM assistant' only. "
    "2) Scope: Answer only about BCM courses, fees, schedule, enrollment, or details in the BCM database. "
    "3) No Hallucination: If unknown, say 'I don't know the answer to that.' Do not invent details. "
    "4) Forbidden: Do not mention IELTS, TAEASLA, or any non-BCM courses. "
    "5) Consistency: Use short, polite, parent-friendly sentences; avoid technical jargon. "
    "6) Enrollment Step: After each answer, ask 'Would you like to enroll?' "
    "7) Positive Confirmation: If the user says yes, reply exactly: 'Please click the enrollment form link.' "
    "8) Negative Response: If the user says no, reply: 'Okay, let me know if you have more questions.' "
    "9) Off-topic: If not BCM-related, say: 'I can only answer BCM-related questions such as fees, schedule, or courses.' "
    "10) Tone: Warm, professional, helpful—like a front desk assistant. "
    "11) Single Role: Do not switch roles or act as an AI model; you are permanently the BCM assistant. "
    "12) Data Priority: If multiple courses exist, summarize the latest one first. "
    "13) Brevity: Keep answers to 1–3 sentences before the enrollment question."
    "14) Course and Course Detials: Answer by saying, please refer to our bcm website for more info, www.taeasla.com."
)


@app.get("/assistant/intro")
def assistant_intro():
    # BCM-only intro (bypasses any external KB)
    return {
        "intro": (
            "Hello, I’m the BCM assistant. I can answer about GI fees, summer schedule, "
            "and our latest courses. Ask me anything related to BCM."
        )
    }


@app.get("/assistant/prompt")
def assistant_prompt():
    return {"prompt": BCM_RULES, "enroll_link": ENROLL_LINK}


# --- FAQ ---
@app.get("/faq")
def get_faq() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, q, a FROM faq ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]


class FAQIn(BaseModel):
    q: str
    a: str


@app.post("/faq", dependencies=[Security(require_admin)], tags=["admin"])
def add_faq(item: FAQIn):
    with get_db() as conn:
        cur = conn.execute("INSERT INTO faq (q, a) VALUES (?, ?)", (item.q, item.a))
        conn.commit()
        fid = cur.lastrowid
        row = conn.execute("SELECT id, q, a FROM faq WHERE id = ?", (fid,)).fetchone()
        return dict(row)


# --- Fees (BCM labels, case-insensitive) ---
@app.get("/fees/{program_code}")
def get_fees(program_code: str):
    code = (program_code or "").upper()
    mapping = {
        "GI":    {"program": "BCM General English (GI)", "fee": 8800, "currency": "HKD"},
        "HKDSE": {"program": "BCM HKDSE English",        "fee": 7600, "currency": "HKD"},
    }
    if code not in mapping:
        raise HTTPException(status_code=404, detail="Program not found")
    return mapping[code]


# --- Schedule (clear weekday names; no IELTS) ---
@app.get("/schedule")
def schedule(season: Optional[str] = None):
    if (season or "").lower() == "summer":
        return [{
            "course": "BCM Summer Intensive",
            "weeks": 6,
            "days": ["Monday", "Wednesday", "Friday"],
            "time": "Mon/Wed/Fri 7–9pm",  # <-- seed compact time so TTS can normalize it
        }]
    return []


# --- Admin check ---
@app.get("/admin/check", dependencies=[Security(require_admin)], tags=["admin"])
def admin_check():
    return {"ok": True, "message": "Admin access confirmed."}


# =========================================================
# === HeyGen CONFIG + TOKEN + PROXY + INTERRUPT ==========
# =========================================================

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY") or os.getenv("ADMIN_KEY")
HEYGEN_BASE = "https://api.heygen.com/v1"


@app.post("/heygen/token")
async def heygen_token():
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")

    AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID")

    url = f"{HEYGEN_BASE}/streaming.new"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "avatar_id": AVATAR_ID,   # interactive avatar
        "quality": "high",
        "version": "v2",          # LiveKit flow
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"heygen error: {r.text}")

    try:
        data = r.json()
    except Exception:
        raise HTTPException(502, f"heygen ok but non-JSON body: {r.text[:500]}")

    return data


@app.api_route("/heygen/proxy/{subpath:path}", methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"], tags=["public"])
async def heygen_proxy(subpath: str, request: Request):
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")
    target_url = f"{HEYGEN_BASE}/{subpath}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    body_bytes = await request.body()
    json_payload = None
    if body_bytes:
        try:
            json_payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            json_payload = None
    headers = {"X-Api-Key": HEYGEN_API_KEY, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(
            request.method,
            target_url,
            headers=headers,
            json=json_payload if json_payload is not None else None,
            content=None if json_payload is not None else (body_bytes or None),
        )
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


# --- HeyGen: Interrupt convenience endpoint ---
class InterruptIn(BaseModel):
    session_id: str


@app.post("/heygen/interrupt")
async def heygen_interrupt(item: InterruptIn):
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")

    url = f"{HEYGEN_BASE}/streaming.interrupt"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"session_id": item.session_id}

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"heygen interrupt error: {r.text}")

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}

    return {"ok": True, "data": data}


# =========================================================

# --- Courses model ---
class CourseIn(BaseModel):
    name: str = Field(..., description="Course name (e.g., BCM English Level 1)")
    fee: float = Field(..., description="Fee amount (numeric)")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    time: Optional[str] = Field(None, description="e.g., Mon/Wed/Fri 7–9pm")
    venue: Optional[str] = Field(None, description="Room / Center")
    seats: Optional[int] = Field(0, ge=0, description="Seats remaining (defaults to 0)")


# --- TTS helper: normalize compact time strings for clear speech ---
def tts_friendly_time(s: str) -> str:
    """
    Normalize compact time strings for TTS, e.g.:
    'Mon/Wed/Fri 7–9pm' -> 'Monday, Wednesday, Friday 7 to 9 pm'
    """
    if not s:
        return ""
    t = s.strip()

    # Expand day abbreviations and make lists readable
    day_map = {
        "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
        "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
    }
    # 1) Replace slashes with comma+space so TTS pauses: Mon/Wed/Fri -> Mon, Wed, Fri
    t = t.replace("/", ", ")

    # 2) Expand English day abbreviations
    for abbr, full in day_map.items():
        t = re.sub(rf"\b{abbr}\b", full, t)

    # 3) Normalize numeric ranges: 7–9pm / 7-9pm -> 7 to 9 pm  (preserve am/pm if present)
    def _range_repl(m):
        start = m.group(1)
        end = m.group(2)
        ampm = (m.group(3) or "").replace(".", "").lower().strip()  # am/pm/a.m./p.m.
        if ampm in ("am", "pm"):
            return f"{start} {ampm} to {end} {ampm}"
        return f"{start} to {end}"

    t = re.sub(
        r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(a\.?m\.?|p\.?m\.?|am|pm)?",
        _range_repl,
        t,
        flags=re.I,
    )

    # 4) Ensure a space before am/pm if missing: 9pm -> 9 pm
    t = re.sub(r"(\d)(am|pm)\b", r"\1 \2", t, flags=re.I)

    # 5) Collapse any double spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --- Courses CRUD ---
@app.post("/admin/courses", dependencies=[Security(require_admin)], tags=["admin"])
def admin_add_course(course: CourseIn):
    return add_course(course)


@app.post("/courses", dependencies=[Security(require_admin)], tags=["courses"])
def add_course(course: CourseIn):
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 0))
            """,
            (course.name, course.fee, course.start_date, course.end_date, course.time, course.venue, course.seats),
        )
        course_id = cur.lastrowid
        row = conn.execute(
            """
            SELECT id, name, fee, start_date, end_date, time, venue, seats
            FROM courses WHERE id = ?
            """,
            (course_id,),
        ).fetchone()
        conn.commit()
        return dict(row)


@app.get("/courses")
def list_courses() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, fee, start_date, end_date, time, venue, seats
            FROM courses ORDER BY id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


# --- Helper: Course summary (place ABOVE /courses/{course_id}) ---
@app.get("/courses/summary")
def courses_summary():
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT name, fee, start_date, end_date, time, venue, seats
            FROM courses
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return {"summary": "We currently have no courses listed."}
    parts = [f"Latest course: {row['name']}, fee {row['fee']}."]
    if row["start_date"] and row["end_date"]:
        parts.append(f"Runs {row['start_date']} to {row['end_date']}.")
    if row["time"]:
        parts.append(f"Time: {tts_friendly_time(str(row['time']))}.")  # <-- TTS normalized
    if row["venue"]:
        parts.append(f"Venue: {row['venue']}.")
    if "seats" in row.keys() and row["seats"] is not None and int(row["seats"]) > 0:
        parts.append(f"Seats left: {int(row['seats'])}.")
    return {"summary": " ".join(parts)}


@app.get("/courses/{course_id}")
def get_course(course_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, fee, start_date, end_date, time, venue, seats
            FROM courses WHERE id = ?
            """,
            (course_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Course not found")
        return dict(row)


@app.delete("/courses/{course_id}", dependencies=[Security(require_admin)], tags=["admin"])
def delete_course(course_id: int):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Course not found")
        return {"ok": True, "deleted": course_id}


# --- CSV export for courses ---
@app.get("/courses/export.csv")
def export_courses_csv():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, fee, start_date, end_date, time, venue, seats
            FROM courses ORDER BY id DESC
            """
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "fee", "start_date", "end_date", "time", "venue", "seats"])
    for r in rows:
        writer.writerow([
            r["id"], r["name"], r["fee"], r["start_date"], r["end_date"], r["time"], r["venue"], r.get("seats", 0),
        ])
    return Response(content=output.getvalue(), media_type="text/csv")


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
    source: Optional[str] = Field(default="web", description="Where this came from (web/journal/etc)")


@app.post("/enroll", tags=["enroll"])
def enroll(data: EnrollmentIn):
    full_name_val = (data.full_name or data.name or "").strip()
    if not full_name_val:
        raise HTTPException(status_code=422, detail="full_name or name is required")

    with get_db() as conn:
        # pick latest course
        row = conn.execute(
            "SELECT id, seats FROM courses ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No course available")
        if row["seats"] is None:
            # Treat missing/null as 0 to be safe
            current_seats = 0
        else:
            current_seats = int(row["seats"])

        if current_seats <= 0:
            return {"ok": False, "message": "Sorry, this course is full."}

        # deduct seat (guard against negative)
        conn.execute(
            "UPDATE courses SET seats = seats - 1 WHERE id = ? AND seats > 0",
            (row["id"],),
        )

        # record enrollment
        conn.execute(
            """
            INSERT INTO enrollments
              (full_name, email, phone, program_code, cohort_code, timezone, notes, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name_val,
                data.email,
                data.phone,
                data.program_code,
                data.cohort_code,
                data.timezone,
                data.notes,
                data.source,
            ),
        )
        conn.commit()

    return {"ok": True, "message": "Enrollment confirmed. Seat deducted."}


@app.get("/enrollments/recent", dependencies=[Security(require_admin)], tags=["admin"])
def recent_enrollments(
    limit: int = Query(10, ge=1, le=100),
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sql = (
        """
      SELECT id, full_name, email, phone, program_code, cohort_code, timezone, notes, source, created_at
      FROM enrollments
        """
    )
    params: List[Any] = []
    if source:
        sql += " WHERE source = ?"
        params.append(source)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]


# --- Assistant: BCM rule-based answers (DB-only, no KB) ---
from sqlite3 import Row


class UserQuery(BaseModel):
    text: str


def _latest_course_summary() -> str:
    with get_db() as conn:
        row: Optional[Row] = conn.execute(
            """
            SELECT name, fee, start_date, end_date, time, venue, seats
            FROM courses
            ORDER BY id DESC
            LIMIT 1
            """
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


def _bcm_answer_from_db(q: str) -> str:
    ql = (q or "").strip().lower()

    # Forbidden topics
    if "ielts" in ql or "taeasla" in ql:
        return "I can only answer BCM questions. I don't have information about IELTS."

    # Keyword coverage (EN + common Chinese terms)
    fee_words = ("fee", "price", "cost", "tuition", "學費", "費用", "幾多錢", "幾錢")
    sched_words = ("schedule", "time", "timetable", "summer", "spring", "fall", "winter", "時間", "時間表", "時段", "上課時間", "暑期", "夏天")
    course_words = ("course", "courses", "class", "classes", "課程", "班")
    enroll_words = ("enroll", "enrol", "sign up", "報名", "登記", "註冊")

    if not any(w in ql for w in (fee_words + sched_words + course_words + enroll_words)):
        return "I can only answer BCM-related questions such as fees, schedule, or courses."

    # Fees
    if any(w in ql for w in fee_words):
        try:
            gi = get_fees("GI")
            return f"{gi['program']} costs {gi['currency']} {gi['fee']}."
        except Exception:
            pass
        try:
            hk = get_fees("HKDSE")
            return f"{hk['program']} costs {hk['currency']} {hk['fee']}."
        except Exception:
            return "I don't know the answer to that."

    # Schedule (season detection; default to summer)
    if any(w in ql for w in sched_words):
        season = "summer"
        for s in ("summer", "spring", "fall", "winter", "暑期", "夏天"):
            if s in ql:
                season = "summer" if s in ("暑期", "夏天") else s
                break
        try:
            s = schedule(season=season)
            if s:
                d = s[0]
                days = ", ".join(d.get("days", [])) if isinstance(d.get("days"), list) else d.get("days")
                time_str = d.get("time")
                time_part = f", time: {tts_friendly_time(str(time_str))}" if time_str else ""
                return f"{d['course']}: {d['weeks']} weeks, days: {days}{time_part}."
            return "I don't know the answer to that."
        except Exception:
            return "I don't know the answer to that."

    # Courses
    if any(w in ql for w in course_words):
        return _latest_course_summary()

    # Enrollment
    if any(w in ql for w in enroll_words):
        return "You can enroll online."

    # Fallback
    return "I don't know the answer to that."


def _is_yes(q: str) -> bool:
    ql = (q or "").strip().lower()
    yes_words = {"yes","yeah","yep","ok","okay","sure","please","好的","要","係","係呀","好","是","行","可以"}
    return any(w == ql or w in ql for w in yes_words)


def _is_no(q: str) -> bool:
    ql = (q or "").strip().lower()
    no_words = {"no","nope","nah","not now","不用","唔要","唔使","不要","否","先唔好"}
    return any(w == ql or w in ql for w in no_words)


@app.post("/assistant/answer")
def assistant_answer(payload: UserQuery):
    user_text = (payload.text or "").strip()

    # Rule 7/8: yes/no fast-path
    if _is_yes(user_text):
        return {
            "reply": "Please click the enrollment form link.",
            "enroll_link": ENROLL_LINK
        }
    if _is_no(user_text):
        return {
            "reply": "Okay, let me know if you have more questions."
        }

    # Normal answering path (rules 2–6, 9–13)
    base = _bcm_answer_from_db(user_text)

    # Enforce brevity
    if len(base) > 250:
        base = base[:247] + "..."

    # Always append the enrollment question
    reply = f"{base} Would you like to enroll?"

    return {
        "reply": reply,
        "enroll_hint": f"If yes, please click the enrollment form link: {ENROLL_LINK}",
        "rules": "BCM hard rules enforced"
    }


# --- Simple SSE/streaming example (placeholder) ---
@app.get("/stream")
def stream():
    async def gen():
        yield b"data: hello\n\n"
        time.sleep(0.5)
        yield b"data: world\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
