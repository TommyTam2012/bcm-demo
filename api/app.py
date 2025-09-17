# app.py  — TAEASLA API (persisted DB + startup seed + HEAD-friendly health)

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

# Optional OpenAI import (safe if package not installed)
try:
    from openai import OpenAI  # noqa: F401
except Exception:
    OpenAI = None  # type: ignore

# --- App & basic setup ---
APP_DIR = Path(__file__).parent.resolve()

# Prefer persistent disk at /data; else keep local file.
# If we detect an older DB in app dir and none in /data, migrate it once.
DATA_DIR = Path("/data")
LEGACY_DB = APP_DIR / "bcm_demo.db"
PERSIST_DB = DATA_DIR / "bcm_demo.db"
if DATA_DIR.is_dir():
    if LEGACY_DB.exists() and not PERSIST_DB.exists():
        try:
            PERSIST_DB.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(LEGACY_DB), str(PERSIST_DB))
        except Exception:
            # If copy fails, we'll still fall back gracefully below
            pass
DB_PATH = str(PERSIST_DB if PERSIST_DB.exists() or DATA_DIR.is_dir() else LEGACY_DB)

app = FastAPI(
    title="TAEASLA API",
    version="1.7.0",
    description="TAEASLA backend: courses, enrollments, fees, schedules, HeyGen proxy, and email alerts.",
)

# Static files
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# --- CORS (configurable; no hardcoded BCM domain) ---
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
                venue TEXT,
                seats INTEGER DEFAULT 0
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
                course_id INTEGER,
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
        conn.commit()

# ---------- Chinese seed set (for startup and manual /courses/seed) ----------
BCM_SEED = [
    ("自然拼讀", 0.0, 30),              # Phonics
    ("拼寫", 0.0, 30),                  # Spelling
    ("文法", 0.0, 30),                  # Grammar
    ("青少年雅思", 0.0, 25),            # Junior IELTS
    ("呈分試", 0.0, 40),                # Primary/Junior assessment
    ("香港 Band 1 入學考試", 0.0, 40),   # Band1 Entrance Exam
    ("香港中學文憑試", 0.0, 35),          # HKDSE
    ("雅思", 0.0, 35),                  # IELTS
    ("托福", 0.0, 35),                  # TOEFL
]

def seed_if_empty():
    """Idempotent: only seeds when courses table is empty."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM courses").fetchone()
        if row and int(row["c"] or 0) == 0:
            for (name_zh, fee, seats) in BCM_SEED:
                conn.execute(
                    """
                    INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats)
                    VALUES (?, ?, NULL, NULL, NULL, NULL, ?)
                    """,
                    (name_zh, float(fee), int(seats)),
                )
            conn.commit()

# --- Admin key guard (accepts new + legacy names) ---
ADMIN_KEY = (
    os.getenv("ADMIN_KEY")
    or os.getenv("VITE_TAEASLA_ADMIN_KEY")
    or os.getenv("VITE_BCM_ADMIN_KEY")
)
api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

def require_admin(x_admin_key: str = Security(api_key_header)):
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not configured on server")
    if not x_admin_key:
        raise HTTPException(status_code=403, detail="Forbidden (no X-Admin-Key header received)")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden (bad X-Admin-Key)")
    return True

# ---------- Startup: ensure schema + auto-seed ----------
@app.on_event("startup")
def _startup():
    init_db()
    seed_if_empty()

# --- Basic routes ---
@app.get("/")
def root():
    return RedirectResponse(url="/static/enroll.html")

# --- Health: accept GET / HEAD / OPTIONS (fix 405 monitors) ---
@app.api_route("/health", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
@app.api_route("/health/", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={"Allow": "GET, HEAD, OPTIONS"})
    return {"ok": True}

# --- TAEASLA assistant: fixed intro + hard rules (expanded) ---
ENROLL_LINK = "/static/enroll.html"  # adjust if your path differs

TAEASLA_RULES = (
    "You are the TAEASLA assistant. Follow these rules strictly: "
    "1) Identity: Introduce yourself as school assistant; ask how you may help. "
    "2) Scope: Answer only about TAEASLA courses, fees, schedule, enrollment, or details in the TAEASLA database. "
    "3) No Hallucination: If unknown, say 'I don't know the answer to that.' Do not invent details. "
    "4) Forbidden: Do not mention IELTS or any non-TAEASLA courses. "
    "5) Consistency: Use short, polite, parent-friendly sentences; avoid technical jargon. "
    "6) Enrollment Step: After each answer, ask 'Would you like to enroll?' "
    "7) Positive Confirmation: If the user says yes, reply exactly: 'Please click the enrollment form link.' "
    "8) Negative Response: If the user says no, reply: 'Okay, let me know if you have more questions.' "
    "9) Off-topic: If not TAEASLA-related, say: 'I can only answer TAEASLA-related questions such as fees, schedule, or courses.' "
    "10) Tone: Warm, professional, helpful—like a front desk assistant. "
    "11) Single Role: Do not switch roles or act as an AI model; you are permanently the TAEASLA assistant. "
    "12) Data Priority: If multiple courses exist, summarize the latest one first. "
    "13) Brevity: Keep answers to 1–3 sentences before the enrollment question. "
    "14) Course and Course Details: Please refer to our TAEASLA website for more info, www.taeasla.com."
)

@app.get("/assistant/intro")
def assistant_intro():
    return {
        "intro": (
            "Hello, I’m the TAEASLA assistant. I can answer about GI fees, summer schedule, "
            "and our latest courses. Ask me anything related to TAEASLA."
        )
    }

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

@app.post("/faq", dependencies=[Security(require_admin)], tags=["admin"])
def add_faq(item: FAQIn):
    with get_db() as conn:
        cur = conn.execute("INSERT INTO faq (q, a) VALUES (?, ?)", (item.q, item.a))
        conn.commit()
        fid = cur.lastrowid
        row = conn.execute("SELECT id, q, a FROM faq WHERE id = ?", (fid,)).fetchone()
        return dict(row)

# --- Fees (TAEASLA labels, case-insensitive) ---
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

# --- Schedule (clear weekday names) ---
@app.get("/schedule")
def schedule(season: Optional[str] = None):
    if (season or "").lower() == "summer":
        return [{
            "course": "TAEASLA Summer Intensive",
            "weeks": 6,
            "days": ["Monday", "Wednesday", "Friday"],
            "time": "Mon/Wed/Fri 7–9pm",  # compact; TTS will normalize
        }]
    return []

# --- Admin check ---
@app.get("/admin/check", dependencies=[Security(require_admin)], tags=["admin"])
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

@app.post("/heygen/token")
async def heygen_token():
    """
    Mint AND START a Streaming v2 session in one call.
    Returns { code, data: { session_id, url, access_token, ... }, started: true }
    """
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")

    AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID")
    if not AVATAR_ID:
        raise HTTPException(500, "HEYGEN_AVATAR_ID missing")

    new_url = f"{HEYGEN_BASE}/streaming.new"
    start_url = f"{HEYGEN_BASE}/streaming.start"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    new_body = {
        "avatar_id": AVATAR_ID,   # ensure Alessandra (or whichever avatar_id you set)
        "quality": "high",
        "version": "v2",          # LiveKit flow
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        # 1) Mint
        r_new = await client.post(new_url, headers=headers, json=new_body)
        j_new = _ok_json_or_raise(r_new)
        d = j_new.get("data") or j_new  # tolerate both shapes
        session_id = d.get("session_id")
        if not session_id:
            raise HTTPException(502, "streaming.new did not return session_id")

        # 2) Start (idempotent handling)
        r_start = await client.post(start_url, headers=headers, json={"session_id": session_id})
        if r_start.status_code != 200:
            # If already started or similar, treat as OK if message implies that.
            msg = r_start.text.lower()
            if "already" not in msg and "started" not in msg:
                # hard fail
                raise HTTPException(r_start.status_code, r_start.text)

        # Combine: return the original "data" (includes url + access_token) plus flag
        out = {"code": j_new.get("code", 100), "data": d, "started": True}
        return out

@app.post("/heygen/start")
async def heygen_start(payload: dict):
    """
    Compatibility endpoint: Start a session explicitly.
    Safe to call multiple times; 'already started' is treated as OK.
    """
    if not HEYGEN_API_KEY:
        raise HTTPException(500, "HEYGEN_API_KEY missing")
    session_id = (payload or {}).get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")

    url = f"{HEYGEN_BASE}/streaming.start"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, headers=headers, json={"session_id": session_id})
    if r.status_code != 200:
        msg = r.text.lower()
        if "already" not in msg and "started" not in msg:
            return Response(content=r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type","application/json"))
    return {"ok": True, "session_id": session_id, "started": True}

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

# --- HeyGen: proxy "speak" so the browser never sees your API key ---
@app.post("/heygen/say")
async def heygen_say(payload: dict):
    """
    Body: { "session_id": "...", "text": "Hello", "task_type": "talk"|"repeat" }
    Default task_type = "talk" (LLM). Use "repeat" to echo exactly the text.
    """
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY missing")

    session_id = (payload or {}).get("session_id")
    text = (payload or {}).get("text", "").strip()
    task_type = (payload or {}).get("task_type", "talk").lower()

    if not session_id or not text:
        raise HTTPException(status_code=400, detail="session_id and text required")

    url = f"{HEYGEN_BASE}/streaming.task"
    headers = {
        "Authorization": f"Bearer {HEYGEN_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {"session_id": session_id, "text": text, "task_type": task_type}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"heygen speak error: {e!s}")

    return Response(content=r.text, status_code=r.status_code, media_type="application/json")

# =========================================================
# ================== COURSES & ENROLLMENT =================
# =========================================================
class CourseIn(BaseModel):
    name: str = Field(..., description="Course name (displayed in Chinese for BCM drop-down)")
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
    t = t.replace("/", ", ")
    for abbr, full in day_map.items():
        t = re.sub(rf"\b{abbr}\b", full, t)

    def _range_repl(m):
        start = m.group(1)
        end = m.group(2)
        ampm = (m.group(3) or "").replace(".", "").lower().strip()
        if ampm in ("am", "pm"):
            return f"{start} {ampm} to {end} {ampm}"
        return f"{start} to {end}"

    t = re.sub(
        r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(a\.?m\.?|p\.?m\.?|am|pm)?",
        _range_repl,
        t,
        flags=re.I,
    )
    t = re.sub(r"(\d)(am|pm)\b", r"\1 \2", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------- ADMIN: add/list/delete/export --------------------
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

class SeedResult(BaseModel):
    seeded: int
    reset: bool

@app.post("/courses/seed", response_model=SeedResult, dependencies=[Security(require_admin)], tags=["courses"])
def seed_courses(reset: bool = Query(True, description="If true, clears and reseeds")):
    """
    Seed the course catalog in Chinese for BCM drop-down.
    You can re-run from Swagger. If reset=True, clears the table first.
    """
    with get_db() as conn:
        if reset:
            conn.execute("DELETE FROM courses")
        count = 0
        for (name_zh, fee, seats) in BCM_SEED:
            conn.execute(
                """
                INSERT INTO courses (name, fee, start_date, end_date, time, venue, seats)
                VALUES (?, ?, NULL, NULL, NULL, NULL, ?)
                """,
                (name_zh, float(fee), int(seats)),
            )
            count += 1
        conn.commit()
    return SeedResult(seeded=count, reset=bool(reset))

@app.get("/courses/options", tags=["courses"])
def course_options() -> List[Dict[str, Any]]:
    """
    Minimal payload for front-end drop-down:
    [{ id, name, seats }]
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, seats FROM courses ORDER BY id ASC"
        ).fetchall()
        return [{"id": r["id"], "name": r["name"], "seats": int(r["seats"] or 0)} for r in rows]

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
        parts.append(f"Time: {tts_friendly_time(str(row['time']))}.")
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

# ============================
# === A L E S S A N D R A  ===
# ===  Smoke-test endpoints ===
# ============================

@app.get("/hotel/ping")
def hotel_ping(who: str = "guest"):
    """
    Returns a simple JSON proving the front-end can reach our backend.
    Example: GET /hotel/ping?who=alessandra
    """
    return {"ok": True, "who": who, "ts": int(time.time())}

@app.get("/heygen/avatar_id")
def heygen_avatar_id():
    """
    Shows which avatar_id the server will use when minting a streaming session.
    We keep the real API key server-side; the browser only sees this ID (safe).
    """
    return {"avatar_id": os.getenv("HEYGEN_AVATAR_ID") or None}

@app.get("/heygen/avatars/interactive")
async def heygen_list_interactive(q: Optional[str] = None):
    """
    Lists avatars visible to this account (server-side) and returns a slim list:
      [{ "avatar_id": "...", "avatar_name": "..." }, ...]
    Optional filter: /heygen/avatars/interactive?q=alessandra
    """
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY missing")

    V2_BASE = "https://api.heygen.com/v2"
    url = f"{V2_BASE}/avatars"
    headers = {
        "Accept": "application/json",
        "X-Api-Key": HEYGEN_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers)
    except Exception as e:
        # Network or timeout
        raise HTTPException(status_code=502, detail=f"heygen list error: {e!s}")

    if r.status_code != 200:
        # Bubble up HeyGen's response so we can see auth/permission issues
        return Response(content=r.text, status_code=r.status_code, media_type="application/json")

    try:
        j = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON from HeyGen /v2/avatars")

    avatars = (j or {}).get("data", {}).get("avatars", []) or []
    slim = []
    for a in avatars:
        name = a.get("avatar_name") or a.get("name")
        aid = a.get("avatar_id") or a.get("id")
        if not aid:
            continue
        slim.append({"avatar_id": aid, "avatar_name": name})

    if q:
        ql = q.lower()
        slim = [row for row in slim if (row.get("avatar_name") or "").lower().find(ql) >= 0]

    return {"ok": True, "count": len(slim), "avatars": slim}

# --- Simple SSE/streaming example (placeholder) ---
@app.get("/stream")
def stream():
    async def gen():
        yield b"data: hello\n\n"
        time.sleep(0.5)
        yield b"data: world\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

# --- Local dev entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
