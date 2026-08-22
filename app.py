from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import sqlite3
import os
import httpx
import time
import threading
import secrets
import json
from urllib.parse import quote

try:
    import bcrypt
    _HAS_BCRYPT = True
except Exception:
    _HAS_BCRYPT = False

# ── Configuration (all via environment variables — nothing secret in this file) ─
# RIPPL_PUBLIC_BASE  the public https origin this server is reachable at, e.g.
#                    https://api.rippl.example  — MUST be set when the frontend is
#                    served from a different origin (GitHub Pages), because stream
#                    URLs handed to <video> must be absolute.
# LASTFM_API_KEY     optional; similar-artist recommendations degrade gracefully
#                    to YouTube-search heuristics when it is absent.
# RIPPL_DB           path to the sqlite database.
# RIPPL_HTML         path to index.html (only used for same-origin serving at /).
PUBLIC_BASE = os.environ.get("RIPPL_PUBLIC_BASE", "").rstrip("/")

app = FastAPI()

# Auth travels in an Authorization header (not a cookie), so the browser never
# treats it as a third-party cookie — this is what makes cross-origin hosting
# (frontend on GitHub Pages, API here) work in Safari and Firefox.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Anon-Id"],
    max_age=86400,
)

# ── In-memory TTL cache ───────────────────────────────────────────────────────
# yt-dlp extraction is the slow part. Caching results makes repeat hits instant,
# which is by far the biggest speed win for searches, video loads and comments.
_CACHE = {}
_CACHE_LOCK = threading.Lock()

# Per-video extraction locks. yt-dlp extraction is slow AND re-extracting the same
# video twice within seconds makes YouTube throttle the second set of stream URLs
# (they then 403 through /proxy). These locks guarantee only ONE extraction per video
# runs at a time; any concurrent caller (warm/prefetch/hover-preview/play) waits and
# reuses the first result instead of firing a duplicate extraction.
_EXTRACT_LOCKS = {}
_EXTRACT_LOCKS_GUARD = threading.Lock()

def _get_extract_lock(video_id):
    with _EXTRACT_LOCKS_GUARD:
        lk = _EXTRACT_LOCKS.get(video_id)
        if lk is None:
            lk = threading.Lock()
            _EXTRACT_LOCKS[video_id] = lk
            # Keep the registry from growing unbounded on a long-lived server.
            if len(_EXTRACT_LOCKS) > 512:
                for k in [k for k in list(_EXTRACT_LOCKS) if k != video_id][:256]:
                    _EXTRACT_LOCKS.pop(k, None)
        return lk

def cache_get(key):
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if not item:
            return None
        value, expires = item
        if expires < time.time():
            _CACHE.pop(key, None)
            return None
        return value

def cache_set(key, value, ttl):
    with _CACHE_LOCK:
        _CACHE[key] = (value, time.time() + ttl)
        # Keep cache small for a phone-hosted server (Termux RAM is limited).
        if len(_CACHE) > 600:
            now = time.time()
            # First drop expired entries; if still too big, drop the oldest-expiring ones.
            expired = [k for k, (_, e) in _CACHE.items() if e < now]
            for k in expired[:200]:
                _CACHE.pop(k, None)
            if len(_CACHE) > 600:
                for k in sorted(_CACHE, key=lambda k: _CACHE[k][1])[:200]:
                    _CACHE.pop(k, None)

def cache_del_video(video_id):
    """Drop every cached entry for one video (raw info + all quality responses) so the
    next request re-extracts brand-new stream URLs. Used when a cached googlevideo URL
    has expired (the '/proxy' 403 → 'this vid is NOT going to load' after a while bug)."""
    with _CACHE_LOCK:
        for k in [k for k in _CACHE if k.startswith(f"info::{video_id}") or k.startswith(f"video::{video_id}::")]:
            _CACHE.pop(k, None)

# ── Last.fm similar-artist data (real "people who like X also like Y") ─────────
# Read-only API key (server-side only; the browser calls our /similar, never Last.fm).
# Set it in the environment: export LASTFM_API_KEY="..."  — never hardcode it here.
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "").strip()

def lastfm_similar_artists(artist, limit=8):
    """Return up to `limit` similar-artist names from Last.fm, cached for a week (artist
    similarity barely changes). Always returns a list; on ANY failure returns [] so the
    recommendation code falls back to its old YouTube-search heuristics."""
    artist = (artist or "").strip()
    if not artist or not LASTFM_API_KEY:
        return []
    ckey = f"lfm::sim::{artist.lower()}::{limit}"
    cached = cache_get(ckey)
    if cached is not None:
        return cached
    out = []
    try:
        with httpx.Client(timeout=8.0) as c:
            r = c.get("https://ws.audioscrobbler.com/2.0/", params={
                "method": "artist.getsimilar",
                "artist": artist,
                "api_key": LASTFM_API_KEY,
                "format": "json",
                "limit": limit,
                "autocorrect": 1,
            })
            data = r.json()
        out = [a.get("name") for a in data.get("similarartists", {}).get("artist", []) if a.get("name")][:limit]
    except Exception:
        out = []
    cache_set(ckey, out, 7 * 24 * 3600)
    return out

@app.get("/similar")
def get_similar(artist: str = Query(...), limit: int = Query(8, ge=1, le=20)):
    return {"artist": artist, "similar_artists": lastfm_similar_artists(artist, limit)}

DB_PATH = os.path.expanduser(os.environ.get("RIPPL_DB", "~/ytapp/db.sqlite"))
HTML_PATH = os.path.expanduser(os.environ.get("RIPPL_HTML", "~/ytapp/index.html"))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def _add_col(conn, table, colname, coldef):
    if colname not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")


def init_db():
    conn = get_db()
    # Base tables (history/playlists keep their data; we add ownership columns below).
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS playlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS playlist_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            playlist_id INTEGER NOT NULL,
            video_id TEXT NOT NULL,
            title TEXT,
            thumbnail TEXT,
            channel TEXT,
            FOREIGN KEY (playlist_id) REFERENCES playlists(id)
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            title TEXT,
            thumbnail TEXT,
            channel TEXT,
            watched_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    """)

    # Ownership columns. user_id = logged-in owner; anon_id = logged-out browser owner.
    # Likes & playlists are user-only. History & blocked can be owned by either.
    _add_col(conn, "playlists", "user_id", "user_id INTEGER")
    _add_col(conn, "history", "user_id", "user_id INTEGER")
    _add_col(conn, "history", "anon_id", "anon_id TEXT")
    _add_col(conn, "users", "avatar", "avatar TEXT")
    _add_col(conn, "users", "time_spent", "time_spent INTEGER DEFAULT 0")

    # likes & blocked_videos previously had a GLOBAL UNIQUE(video_id), which is wrong for
    # multi-user (two people couldn't like the same song). Recreate them once with per-owner
    # scoping. Safe because we're starting accounts fresh.
    ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if not ver or ver[0] != '2':
        conn.executescript("""
            DROP TABLE IF EXISTS likes;
            CREATE TABLE likes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT, thumbnail TEXT, channel TEXT,
                liked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, video_id)
            );
            DROP TABLE IF EXISTS blocked_videos;
            CREATE TABLE blocked_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                anon_id TEXT,
                video_id TEXT NOT NULL,
                title TEXT, reason TEXT,
                blocked_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','2');
        """)
    conn.commit()
    conn.close()


init_db()


# ── Auth helpers + identity middleware ────────────────────────────────────────
SESSION_DAYS = 30

def _hash_pw(pw):
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _check_pw(pw, h):
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), h.encode("utf-8"))
    except Exception:
        return False

def _user_for_token(token):
    if not token:
        return None
    conn = get_db()
    row = conn.execute("SELECT user_id FROM sessions WHERE token=?", (token,)).fetchone()
    conn.close()
    return row["user_id"] if row else None


def _token_from(request: Request):
    """Session token, preferring the Authorization header over the cookie.

    Cookies only work same-origin. When the frontend lives on GitHub Pages and this
    API lives on a tunnel hostname, the session cookie is a *third-party* cookie —
    Safari blocks it outright and Firefox partitions it, so logins would silently
    fail. The header path is the one that actually works cross-origin; the cookie
    path is kept so same-origin use (hitting this server directly) still works.
    """
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    return request.cookies.get("session")


@app.middleware("http")
async def identity_middleware(request: Request, call_next):
    # Resolve logged-in user from the bearer token or session cookie (None if logged out).
    request.state.user_id = _user_for_token(_token_from(request))
    # Ensure every browser has a stable anonymous id (drives logged-out personalization).
    # Cross-origin clients send it as a header and persist it in localStorage.
    anon = request.headers.get("x-anon-id") or request.cookies.get("anon_id")
    new_anon = None
    if not anon:
        anon = "a_" + secrets.token_hex(16)
        new_anon = anon
    request.state.anon_id = anon
    response = await call_next(request)
    if new_anon:
        # Header for cross-origin clients, cookie for same-origin ones.
        response.headers["X-Anon-Id"] = new_anon
        response.set_cookie("anon_id", new_anon, max_age=60*60*24*365*2,
                            httponly=False, samesite="none", secure=True, path="/")
    return response

def _uid(request):
    return getattr(request.state, "user_id", None)

def _anon(request):
    return getattr(request.state, "anon_id", None)

def _require_login(request):
    uid = _uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    return uid

def _username(uid):
    if not uid:
        return None
    conn = get_db()
    row = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return row["username"] if row else None


@app.post("/auth/signup")
async def auth_signup(request: Request, response: Response):
    if not _HAS_BCRYPT:
        raise HTTPException(status_code=503, detail="Server missing bcrypt. Run: pip install bcrypt")
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not (3 <= len(username) <= 20) or not username.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Username must be 3–20 letters/numbers/underscore")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    conn = get_db()
    if conn.execute("SELECT 1 FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="That username is taken")
    cur = conn.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                       (username, _hash_pw(password)))
    uid = cur.lastrowid
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions (token, user_id) VALUES (?,?)", (token, uid))
    # Fold this browser's anonymous history into the new account (so their adapted feed
    # carries over). Likes/playlists were never anonymous, so nothing to move there.
    anon = _anon(request)
    if anon:
        conn.execute("UPDATE history SET user_id=?, anon_id=NULL WHERE anon_id=?", (uid, anon))
    conn.commit()
    conn.close()
    response.set_cookie("session", token, max_age=60*60*24*SESSION_DAYS,
                        httponly=True, samesite="none", secure=True, path="/")
    # The token is also returned in the body so cross-origin clients can store it
    # and send it as `Authorization: Bearer <token>` on later requests.
    return {"username": username, "avatar": None, "token": token}

@app.post("/auth/login")
async def auth_login(request: Request, response: Response):
    if not _HAS_BCRYPT:
        raise HTTPException(status_code=503, detail="Server missing bcrypt. Run: pip install bcrypt")
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    conn = get_db()
    row = conn.execute("SELECT id, password_hash, username, avatar FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    if not row or not _check_pw(password, row["password_hash"]):
        conn.close()
        raise HTTPException(status_code=401, detail="Wrong username or password")
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions (token, user_id) VALUES (?,?)", (token, row["id"]))
    conn.commit()
    real_name = row["username"]; av = row["avatar"]
    conn.close()
    response.set_cookie("session", token, max_age=60*60*24*SESSION_DAYS,
                        httponly=True, samesite="none", secure=True, path="/")
    return {"username": real_name, "avatar": av, "token": token}

@app.post("/auth/logout")
def auth_logout(request: Request, response: Response):
    token = _token_from(request)
    if token:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit()
        conn.close()
    response.delete_cookie("session", path="/")
    return {"status": "ok"}

@app.get("/auth/me")
def auth_me(request: Request):
    uid = _uid(request)
    if not uid:
        return {"username": None, "avatar": None}
    conn = get_db()
    row = conn.execute("SELECT username, avatar FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not row:
        return {"username": None, "avatar": None}
    return {"username": row["username"], "avatar": row["avatar"]}

@app.get("/auth/stats")
def auth_stats(request: Request):
    uid = _require_login(request)
    conn = get_db()
    u = conn.execute("SELECT username, avatar, time_spent, created_at FROM users WHERE id=?", (uid,)).fetchone()
    likes = conn.execute("SELECT COUNT(*) AS c FROM likes WHERE user_id=?", (uid,)).fetchone()["c"]
    pls = conn.execute("SELECT COUNT(*) AS c FROM playlists WHERE user_id=?", (uid,)).fetchone()["c"]
    conn.close()
    if not u:
        raise HTTPException(status_code=404, detail="user gone")
    return {
        "username": u["username"],
        "avatar": u["avatar"],
        "time_spent": u["time_spent"] or 0,
        "created_at": u["created_at"],
        "videos_saved": likes,
        "playlists": pls,
    }

@app.post("/auth/avatar")
async def auth_avatar(request: Request):
    uid = _require_login(request)
    body = await request.json()
    avatar = body.get("avatar") or ""
    # Client downscales to a small square data URL; cap size as a safety net (~500KB).
    if len(avatar) > 700_000:
        raise HTTPException(status_code=413, detail="Image too large — try a smaller one")
    if avatar and not avatar.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Invalid image")
    conn = get_db()
    conn.execute("UPDATE users SET avatar=? WHERE id=?", (avatar or None, uid))
    conn.commit()
    conn.close()
    return {"avatar": avatar or None}

@app.post("/auth/heartbeat")
def auth_heartbeat(request: Request, seconds: int = 0):
    uid = _uid(request)
    if not uid:
        return {"status": "skip"}
    seconds = max(0, min(int(seconds or 0), 120))   # clamp so a bad client can't inflate it
    conn = get_db()
    conn.execute("UPDATE users SET time_spent = COALESCE(time_spent,0) + ? WHERE id=?", (seconds, uid))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ── Home ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return open(HTML_PATH).read()


# ── Search (with pagination) ──────────────────────────────────────────────────
# page=1 fetches results 1-20, page=2 fetches 21-40, etc.
# yt-dlp's ytsearchN supports large N values.

@app.get("/search")
def search(q: str = Query(...), page: int = Query(1, ge=1), n: int = Query(0, ge=0, le=40)):
    per_page = 25
    total_fetch = page * per_page

    qkey = q.lower().strip()
    base_key = f"searchset::{qkey}" if not n else f"searchset::{qkey}::n{n}"
    cached = cache_get(base_key)
    # Grow the search set as the user pages deeper, so they don't hit a wall after p2.
    # Hard ceiling at 200 results (8 pages of 25) — far more than the previous 40 cap.
    have_enough = cached and (len(cached) >= total_fetch + 25 or len(cached) >= 200)
    all_videos = cached if have_enough else None

    if all_videos is None:
        # n lets callers (e.g. the home feed) request a smaller batch for speed.
        fetch_n = n if n else min(max(total_fetch + 25, 60), 200)
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{fetch_n}:{q}", download=False)
        all_videos = []
        for entry in info.get("entries", []):
            if not entry:
                continue
            vid_id = entry.get("id")
            all_videos.append({
                "id": vid_id,
                "title": entry.get("title"),
                "thumbnail": entry.get("thumbnail") or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
                "channel": entry.get("channel") or entry.get("uploader"),
                "duration": entry.get("duration"),
                "view_count": entry.get("view_count"),
            })
        cache_set(base_key, all_videos, 1800)  # 30 min — search results barely change in that window

    # Slice to current page
    start = (page - 1) * per_page
    page_videos = all_videos[start:start + per_page]

    return {
        "videos": page_videos,
        "page": page,
        "per_page": per_page,
        "has_more": len(all_videos) > start + per_page,
    }


# ── Search for YouTube playlists ──────────────────────────────────────────────
# Uses YouTube's search with the "playlist" filter (sp=EgIQAw%3D%3D). This returns
# ALL public playlists matching the query — official artist playlists AND community/
# user-generated ones (what YouTube Music calls "community playlists").

@app.get("/search_playlists")
def search_playlists(q: str = Query(...), limit: int = Query(40, ge=1, le=80)):
    search_url = f"https://www.youtube.com/results?search_query={quote(q)}&sp=EgIQAw%3D%3D"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": limit,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
    except Exception:
        return {"playlists": []}

    playlists = []
    for entry in info.get("entries", []) or []:
        if not entry:
            continue
        etype = entry.get("_type") or ""
        pid = entry.get("id") or ""
        # Accept any playlist id. Skip auto-generated "Mix"/radio lists (RD...) which
        # aren't real community playlists and can't be opened normally.
        is_playlist = (etype == "playlist" or entry.get("ie_key") == "YoutubeTab"
                       or pid.startswith("PL") or pid.startswith("OL") or pid.startswith("FL"))
        if not is_playlist:
            continue
        if pid.startswith("RD"):   # radio/mix, skip
            continue
        thumb = entry.get("thumbnail")
        thumbs = entry.get("thumbnails") or []
        if not thumb and thumbs:
            thumb = thumbs[-1].get("url")
        playlists.append({
            "id": pid,
            "title": entry.get("title"),
            "channel": entry.get("channel") or entry.get("uploader") or "",
            "video_count": entry.get("playlist_count") or entry.get("video_count") or "",
            "thumbnail": thumb or "",
        })
        if len(playlists) >= limit:
            break
    return {"playlists": playlists}


# ── Fetch videos inside a YouTube playlist ────────────────────────────────────

@app.get("/playlist_videos")
def playlist_videos(playlist_id: str = Query(...)):
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": 100,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return {"videos": []}

    videos = []
    for entry in info.get("entries", []) or []:
        if not entry:
            continue
        vid_id = entry.get("id")
        if not vid_id:
            continue
        videos.append({
            "id": vid_id,
            "title": entry.get("title"),
            "thumbnail": entry.get("thumbnail") or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
            "channel": entry.get("channel") or entry.get("uploader") or "",
            "duration": entry.get("duration"),
            "view_count": entry.get("view_count"),
        })
    return {"videos": videos, "title": info.get("title")}


# ── Video (with quality selection) ───────────────────────────────────────────

def _raw_extract(video_id):
    """Extract full info once (all formats) and cache it. This is the expensive call;
    caching it makes quality switches and repeat opens instant.

    This is the PROVEN-WORKING extraction (the one that delivered 1080p/4K + slowed):
      • extract_info with NORMAL processing (NOT process=False). Normal processing is what
        descrambles YouTube's signature/`n` parameter, giving URLs that actually play.
        process=False returns raw URLs that load to a BLACK screen — that was the bug.
      • Client order [None, "web", "ios", "android", "tv"] — leading with None lets yt-dlp
        use its own (well-tuned) default client selection, which avoids the SABR/DRM trap
        that forcing a single old client name falls into. web/mweb expose the full DASH
        ladder (1080/1440/2160); android is a reliable progressive fallback.
      • Accept the first client that returns ANY non-empty formats list."""
    cache_key = f"info::{video_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    # Serialize per video: if another request is already extracting this id, wait for
    # it and reuse its result rather than launching a second, throttle-triggering one.
    lock = _get_extract_lock(video_id)
    with lock:
        cached = cache_get(cache_key)   # a concurrent caller may have just filled it
        if cached:
            return cached
        url = f"https://www.youtube.com/watch?v={video_id}"
        last_err = None
        for client in ["android", None, "web", "ios", "tv"]:
            opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
                    # Fail a stalled/blocked client fast (default is 20s) so the fallback
                    # chain below doesn't stack multiple 20s hangs before finding one that
                    # works — this was a real source of slow FIRST loads (cache misses).
                    "socket_timeout": 8}
            if client:
                opts["extractor_args"] = {"youtube": {"player_client": [client]}}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                if info and info.get("formats"):
                    cache_set(cache_key, info, 1800)  # 30 min — stream URLs stay valid a few hours
                    return info
            except Exception as e:
                last_err = e
                continue
        raise last_err if last_err else Exception("extraction failed")




import re as _re
_VALID_ID = _re.compile(r'^[A-Za-z0-9_-]{11}$')


@app.get("/video/{video_id}")
def get_video(video_id: str, quality: str = Query("best"), nohls: int = Query(0), nosplit: int = Query(0),
               fresh: int = Query(0), audio_only: int = Query(0)):
    # Reject malformed IDs immediately. A bad id (e.g. "#", truncated) otherwise sends
    # yt-dlp into a multi-client retry storm that floods logs and hammers the device.
    if not _VALID_ID.match(video_id or ""):
        raise HTTPException(status_code=400, detail="invalid video id")

    # fresh=1 → the previously-served stream URL expired (proxy 403). Drop all caches for
    # this video so we re-extract brand-new URLs.
    if fresh:
        cache_del_video(video_id)

    # The frontend's playback-failure fallback (refetchProgressive) asks for a SINGLE muxed
    # element by passing nosplit=1 and/or nohls=1. Honour either as "give me a progressive
    # (muxed) stream, never split" so that fallback path plays with audio instead of a
    # silent/black video-only stream.
    single_element = bool(nosplit or nohls)
    want_audio_only = bool(audio_only)

    # Short per-(video,quality) response cache so repeated opens are instant.
    resp_key = f"video::{video_id}::{quality}::{int(single_element)}::{int(want_audio_only)}"
    cached = cache_get(resp_key)
    if cached:
        return cached

    try:
        info = _raw_extract(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    formats = info.get("formats", [])

    # Build a list of all heights that have a video track.
    available_qualities = []
    seen_heights = set()
    for f in sorted(formats, key=lambda x: x.get("height") or 0, reverse=True):
        h = f.get("height")
        if h and h not in seen_heights and f.get("vcodec", "none") != "none":
            seen_heights.add(h)
            available_qualities.append({"label": f"{h}p", "value": str(h)})

    # Decide target height.
    heights = sorted(seen_heights, reverse=True)
    if quality in ("best", None, "") or quality == "auto":
        target_h = heights[0] if heights else None
    else:
        try:
            want = int(quality)
            target_h = min([h for h in heights if h <= want] or heights, key=lambda h: abs(h - want)) if heights else None
        except Exception:
            target_h = heights[0] if heights else None

    def best_progressive(maxh=None):
        cand = [f for f in formats
                if f.get("vcodec", "none") != "none" and f.get("acodec", "none") != "none"
                and f.get("url") and "m3u8" not in (f.get("protocol") or "")
                and (maxh is None or (f.get("height") or 0) <= maxh)]
        cand.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        return cand[0] if cand else None

    def best_video_only(h):
        cand = [f for f in formats
                if f.get("vcodec", "none") != "none" and f.get("acodec", "none") == "none"
                and f.get("url") and "m3u8" not in (f.get("protocol") or "") and (f.get("height") or 0) == h]
        # prefer mp4/h264 for browser compatibility
        cand.sort(key=lambda f: (f.get("ext") == "mp4", f.get("tbr") or 0), reverse=True)
        return cand[0] if cand else None

    def best_audio():
        cand = [f for f in formats
                if f.get("acodec", "none") != "none" and f.get("vcodec", "none") == "none"
                and f.get("url") and "m3u8" not in (f.get("protocol") or "")]
        cand.sort(key=lambda f: (f.get("ext") in ("m4a", "mp4"), f.get("abr") or 0), reverse=True)
        return cand[0] if cand else None

    video_url = None
    audio_url = None
    is_split = False
    is_audio_only = False

    # AUDIO-ONLY MODE: point the <video> element straight at an audio-only DASH track.
    # A <video> tag plays an audio-only file fine (sound only, no frame) — so this reuses
    # every existing control/sync/lock-screen code path untouched, while fetching NONE of
    # the video bytes. If no audio-only track exists for this video, we silently fall
    # through to normal video playback below rather than failing the request.
    if want_audio_only:
        au = best_audio()
        if au and au.get("url"):
            video_url = au["url"]
            is_audio_only = True


    # RELIABILITY FIRST (blocked residential IPs):
    # Progressive/muxed streams (especially android 360p) often work when DASH
    # 720/1080 tracks return 403. Prefer progressive for normal playback so
    # videos actually play. Split is only used when no progressive exists.
    if not video_url:
        if single_element:
            p = best_progressive(target_h) or best_progressive()
            if p:
                video_url = p.get("url")
            elif info.get("url"):
                video_url = info.get("url")
        else:
            p = best_progressive(target_h) or best_progressive()
            if p:
                video_url = p.get("url")
            else:
                vo = best_video_only(target_h) if target_h else None
                if not vo and heights:
                    vo = best_video_only(heights[0])
                au = best_audio()
                if vo and au:
                    video_url = vo.get("url")
                    audio_url = au.get("url")
                    is_split = True
                elif info.get("url"):
                    video_url = info.get("url")

    if not video_url:
        raise HTTPException(status_code=404, detail="No playable stream found")

    thumbnail = info.get("thumbnail") or f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    channel = info.get("channel") or info.get("uploader") or ""
    upload_date = info.get("upload_date") or ""
    if len(upload_date) == 8:
        upload_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    result = {
        "id": video_id,
        "title": info.get("title"),
        # Absolute when RIPPL_PUBLIC_BASE is set: a <video src> on GitHub Pages would
        # otherwise resolve "/proxy?..." against github.io and 404.
        "video_url": f"{PUBLIC_BASE}/proxy?url={quote(video_url, safe='')}",
        "video_url_direct": video_url,
        "audio_url": (f"{PUBLIC_BASE}/proxy?url={quote(audio_url, safe='')}" if audio_url else None),
        # The frontend is served from GitHub Pages (a DIFFERENT origin than this API).
        # A relative link like <a href="/download?..."> resolves against GitHub Pages,
        # not this server, and 404s there. Sending public_base lets the client build an
        # ABSOLUTE download link the same way video_url already is — no Cloudflare/DNS
        # changes needed, just reusing the mechanism that already makes streaming work.
        "public_base": PUBLIC_BASE,
        "is_split": is_split,
        "is_audio_only": is_audio_only,
        "is_hls": False,
        "thumbnail": thumbnail,
        "channel": channel,
        "view_count": info.get("view_count"),
        "upload_date": upload_date,
        "duration": info.get("duration"),
        "description": (info.get("description") or "")[:1000],
        "available_qualities": available_qualities,
        "tags": (info.get("tags") or [])[:15],
        "categories": info.get("categories") or [],
    }
    cache_set(resp_key, result, 1500)
    return result


# ── Comments ──────────────────────────────────────────────────────────────────

@app.get("/comments/{video_id}")
def get_comments(video_id: str, limit: int = Query(200, ge=20, le=500)):
    cache_key = f"comments::{video_id}::{limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"https://www.youtube.com/watch?v={video_id}"

    def fetch(client):
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "getcomments": True,
            # max_comments: total, per-thread, replies-per-thread(0=none), pages
            "extractor_args": {"youtube": {
                "comment_sort": ["top"],
                "max_comments": [str(limit), "all", "0", "all"],
            }},
        }
        if client:
            opts["extractor_args"]["youtube"]["player_client"] = [client]
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    info = None
    for client in [None, "web", "android"]:
        try:
            info = fetch(client)
            if info and info.get("comments"):
                break
        except Exception:
            continue
    try:
        comments = (info or {}).get("comments") or []
        top = [c for c in comments if not c.get("parent") or c.get("parent") == "root"]
        top.sort(key=lambda c: c.get("like_count") or 0, reverse=True)
        result = [
            {
                "id": c.get("id"),
                "author": c.get("author"),
                "author_thumbnail": c.get("author_thumbnail"),
                "text": (c.get("text") or "")[:600],
                "like_count": c.get("like_count") or 0,
                "timestamp": c.get("timestamp"),
            }
            for c in top[:limit]
        ]
        if result:
            cache_set(cache_key, result, 1200)  # 20 min
        return result
    except Exception:
        return []


# ── Stream proxy ──────────────────────────────────────────────────────────────

@app.get("/prefetch")
def prefetch(video_id: str = Query(...)):
    """Fire-and-forget: warm the _raw_extract cache for this video so that when the
    user actually taps it, /video/{id} returns almost instantly. Returns immediately;
    the extraction happens on a background thread. Safe to call many times — _raw_extract
    is idempotent and short-circuits on cache hit."""
    if not _VALID_ID.match(video_id or ""):
        return {"status": "skip"}
    # Skip if already cached — no thread, no wasted work.
    if cache_get(f"info::{video_id}"):
        return {"status": "cached"}
    def _warm():
        try: _raw_extract(video_id)
        except Exception: pass
    threading.Thread(target=_warm, daemon=True).start()
    return {"status": "warming"}


@app.get("/sponsorblock")
def sponsorblock(video_id: str = Query(...)):
    """Fetch skippable segments from the SponsorBlock community DB. Cached 1h.
    Categories: sponsor, self-promo, interaction reminders, intros/outros, previews,
    and non-music sections (great for skipping the talking bits on music videos)."""
    if not _VALID_ID.match(video_id or ""):
        return {"segments": []}
    ckey = f"sb::{video_id}"
    cached = cache_get(ckey)
    if cached is not None:
        return {"segments": cached}
    cats = ["sponsor", "selfpromo", "interaction", "intro", "outro", "preview", "music_offtopic"]
    try:
        params = {"videoID": video_id, "categories": json.dumps(cats)}
        with httpx.Client(timeout=8.0) as client:
            r = client.get("https://sponsor.ajay.app/api/skipSegments", params=params)
            if r.status_code == 404:
                cache_set(ckey, [], 3600)
                return {"segments": []}
            r.raise_for_status()
            data = r.json()
        segs = []
        for it in data:
            s = it.get("segment") or []
            if len(s) == 2 and s[1] > s[0]:
                segs.append([round(float(s[0]), 2), round(float(s[1]), 2), it.get("category", "")])
        segs.sort(key=lambda x: x[0])
        cache_set(ckey, segs, 3600)
        return {"segments": segs}
    except Exception:
        return {"segments": []}


def _safe_filename(name, ext):
    name = (name or "rippl").strip()
    # Strip filesystem-illegal chars but keep unicode (emoji, accents, etc.)
    name = _re.sub(r'[\\/:*?"<>|\r\n\t]+', "", name)[:120].strip() or "rippl"
    return f"{name}.{ext}"

@app.get("/download")
async def download(video_id: str = Query(...), kind: str = Query("video"),
                   url: str = Query(""), name: str = Query(""), request: Request = None):
    """Download whatever the user is currently watching, as a file.

    The reliable path (used by the frontend) is to pass `url=` — the SAME direct stream
    URL the player is already playing — plus an optional `name=`. We then relay it with
    the exact async + retry mechanism /proxy uses (which works on this host), only adding
    a Content-Disposition so the browser saves it instead of playing it. This means: if
    it streams, it downloads. No format re-picking, no separate sync client (that sync
    path was what made downloads fail even when streaming worked).

    If `url=` is omitted we fall back to extracting a format server-side (kind=audio →
    best m4a, else best progressive), so old links / direct hits still work.
    """
    if not _VALID_ID.match(video_id or ""):
        raise HTTPException(status_code=400, detail="bad id")

    src_url = url or ""
    # Only accept googlevideo stream URLs for the pass-through path (don't become an open proxy).
    if src_url and "googlevideo.com" not in src_url:
        src_url = ""

    # Decide filename + extension.
    ext = "m4a" if kind == "audio" else "mp4"
    title = name or video_id

    # Fallback: no url given → pick a format ourselves (kept for compatibility).
    if not src_url:
        info = _raw_extract(video_id)
        title = name or info.get("title") or video_id
        fmts = info.get("formats") or []
        def _direct(f):
            return f.get("url") and "m3u8" not in (f.get("protocol") or "")
        if kind == "audio":
            cands = [f for f in fmts if f.get("acodec") not in (None, "none")
                     and f.get("vcodec") in (None, "none") and _direct(f)]
            cands.sort(key=lambda f: (0 if f.get("ext") in ("m4a", "mp4") else 1, -(f.get("abr") or 0)))
            ext = "m4a"
        else:
            cands = [f for f in fmts if f.get("acodec") not in (None, "none")
                     and f.get("vcodec") not in (None, "none") and _direct(f)]
            cands.sort(key=lambda f: (0 if f.get("ext") == "mp4" else 1,
                                      abs((f.get("height") or 0) - 360), f.get("height") or 0))
            ext = "mp4"
        if not cands:
            raise HTTPException(status_code=404, detail="no downloadable format available")
        src_url = cands[0]["url"]

    filename = _safe_filename(title, ext)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://www.youtube.com/",
        "Origin": "https://www.youtube.com",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    # Honour a Range header if the download manager sends one (enables resume/progress).
    range_header = request.headers.get("range") if request else None
    if range_header:
        headers["Range"] = range_header

    # SAME async + retry path as /proxy (this is the mechanism that actually works here).
    client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(None, connect=15.0))
    upstream = None
    for attempt in range(4):
        req = client.build_request("GET", src_url, headers=headers)
        upstream = await client.send(req, stream=True)
        if upstream.status_code in (403, 500, 502, 503, 504) and attempt < 3:
            await upstream.aclose()
            continue
        break

    if upstream is None or upstream.status_code >= 400:
        code = upstream.status_code if upstream is not None else 502
        if upstream is not None:
            await upstream.aclose()
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"download upstream failed ({code})")

    async def body():
        try:
            async for chunk in upstream.aiter_raw(chunk_size=262144):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    ascii_fallback = _re.sub(r'[^\x20-\x7e]', '_', filename) or f"rippl.{ext}"
    quoted_utf8 = quote(filename, safe="")
    out_headers = {
        "Content-Disposition": f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quoted_utf8}",
        "Cache-Control": "no-store",
        "Accept-Ranges": "bytes",
    }
    if "content-length" in upstream.headers:
        out_headers["Content-Length"] = upstream.headers["content-length"]
    if "content-range" in upstream.headers:
        out_headers["Content-Range"] = upstream.headers["content-range"]
    media_type = upstream.headers.get("content-type") or ("audio/mp4" if ext == "m4a" else "video/mp4")
    status_code = upstream.status_code if upstream.status_code in (200, 206) else 200
    return StreamingResponse(body(), status_code=status_code, media_type=media_type, headers=out_headers)



@app.get("/proxy")
async def proxy_stream(url: str = Query(...), request: Request = None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://www.youtube.com/",
        "Origin": "https://www.youtube.com",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    range_header = request.headers.get("range") if request else None
    if range_header:
        headers["Range"] = range_header

    client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(None, connect=15.0))

    # Open upstream, retrying a couple of times on transient 403/5xx (googlevideo URLs
    # sometimes reject the first hit or die briefly). No HEAD — many reject it.
    upstream = None
    for attempt in range(3):
        req = client.build_request("GET", url, headers=headers)
        upstream = await client.send(req, stream=True)
        if upstream.status_code in (403, 500, 502, 503, 504) and attempt < 2:
            await upstream.aclose()
            continue
        break

    async def stream_body():
        try:
            async for chunk in upstream.aiter_raw(chunk_size=262144):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    passthrough = {}
    for h in ("content-type", "content-length", "content-range", "accept-ranges"):
        if h in upstream.headers:
            passthrough[h] = upstream.headers[h]
    passthrough.setdefault("Accept-Ranges", "bytes")
    media_type = upstream.headers.get("content-type", "video/mp4")

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        media_type=media_type,
        headers=passthrough,
    )


@app.get("/config")
def get_config():
    """Lets the frontend learn the server's real public origin once at load time, so it
    can build absolute links (e.g. downloads) the same way /video already builds
    video_url/proxy — needed because the frontend is served from a different origin
    (GitHub Pages) than this API, so a relative link/fetch would resolve against the
    wrong host and 404 there instead of reaching this server."""
    return {"public_base": PUBLIC_BASE}


@app.get("/health")
def health():
    return {"status": "ok", "cache_entries": len(_CACHE)}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/history")
def get_history(request: Request):
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    if uid:
        rows = conn.execute("SELECT * FROM history WHERE user_id=? ORDER BY watched_at DESC LIMIT 100", (uid,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM history WHERE anon_id=? ORDER BY watched_at DESC LIMIT 100", (anon,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/history/add")
def add_history(request: Request, video_id: str, title: str = "", thumbnail: str = "", channel: str = ""):
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    if uid:
        conn.execute("DELETE FROM history WHERE video_id=? AND user_id=?", (video_id, uid))
        conn.execute("INSERT INTO history (video_id, title, thumbnail, channel, user_id) VALUES (?,?,?,?,?)",
                     (video_id, title, thumbnail, channel, uid))
    else:
        conn.execute("DELETE FROM history WHERE video_id=? AND anon_id=?", (video_id, anon))
        conn.execute("INSERT INTO history (video_id, title, thumbnail, channel, anon_id) VALUES (?,?,?,?,?)",
                     (video_id, title, thumbnail, channel, anon))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.delete("/history/clear")
def clear_history(request: Request):
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    if uid:
        conn.execute("DELETE FROM history WHERE user_id=?", (uid,))
    else:
        conn.execute("DELETE FROM history WHERE anon_id=?", (anon,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/history/{video_id}")
def remove_history_item(request: Request, video_id: str):
    # Declared AFTER /history/clear so the literal "clear" route matches first.
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    if uid:
        conn.execute("DELETE FROM history WHERE video_id=? AND user_id=?", (video_id, uid))
    else:
        conn.execute("DELETE FROM history WHERE video_id=? AND anon_id=?", (video_id, anon))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ── Likes ─────────────────────────────────────────────────────────────────────

@app.get("/likes")
def get_likes(request: Request):
    uid = _uid(request)
    if not uid:
        return []   # logged out: no personal library
    conn = get_db()
    rows = conn.execute("SELECT * FROM likes WHERE user_id=? ORDER BY liked_at DESC", (uid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/likes/add")
def add_like(request: Request, video_id: str, title: str = "", thumbnail: str = "", channel: str = ""):
    uid = _require_login(request)
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO likes (user_id, video_id, title, thumbnail, channel) VALUES (?,?,?,?,?)",
                 (uid, video_id, title, thumbnail, channel))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.delete("/likes/remove")
def remove_like(request: Request, video_id: str):
    uid = _require_login(request)
    conn = get_db()
    conn.execute("DELETE FROM likes WHERE video_id=? AND user_id=?", (video_id, uid))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/likes/check/{video_id}")
def check_like(request: Request, video_id: str):
    uid = _uid(request)
    if not uid:
        return {"liked": False}
    conn = get_db()
    row = conn.execute("SELECT id FROM likes WHERE video_id=? AND user_id=?", (video_id, uid)).fetchone()
    conn.close()
    return {"liked": row is not None}


# ── Blocked videos ("Don't recommend") ────────────────────────────────────────

@app.get("/blocked")
def get_blocked(request: Request):
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    if uid:
        rows = conn.execute("SELECT video_id FROM blocked_videos WHERE user_id=?", (uid,)).fetchall()
    else:
        rows = conn.execute("SELECT video_id FROM blocked_videos WHERE anon_id=?", (anon,)).fetchall()
    conn.close()
    return {"ids": [r["video_id"] for r in rows]}

@app.post("/blocked/add")
def add_blocked(request: Request, video_id: str, title: str = "", reason: str = ""):
    uid, anon = _uid(request), _anon(request)
    conn = get_db()
    conn.execute("INSERT INTO blocked_videos (user_id, anon_id, video_id, title, reason) VALUES (?,?,?,?,?)",
                 (uid, None if uid else anon, video_id, title, reason))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/blocked/add_bulk")
async def add_blocked_bulk(request: Request):
    data = await request.json()
    ids = data.get("ids", [])
    reason = data.get("reason", "")
    uid, anon = _uid(request), _anon(request)
    owner_u = uid
    owner_a = None if uid else anon
    conn = get_db()
    for it in ids:
        vid = it.get("video_id") if isinstance(it, dict) else it
        title = it.get("title", "") if isinstance(it, dict) else ""
        if vid:
            conn.execute("INSERT INTO blocked_videos (user_id, anon_id, video_id, title, reason) VALUES (?,?,?,?,?)",
                         (owner_u, owner_a, vid, title, reason))
    conn.commit()
    conn.close()
    return {"status": "ok", "count": len(ids)}


# ── Playlists ─────────────────────────────────────────────────────────────────

def _owns_playlist(conn, uid, playlist_id):
    row = conn.execute("SELECT user_id FROM playlists WHERE id=?", (playlist_id,)).fetchone()
    return row is not None and row["user_id"] == uid

@app.get("/playlists")
def get_playlists(request: Request):
    uid = _uid(request)
    if not uid:
        return []
    conn = get_db()
    rows = conn.execute("SELECT * FROM playlists WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/playlists")
def create_playlist(request: Request, name: str):
    uid = _require_login(request)
    conn = get_db()
    cursor = conn.execute("INSERT INTO playlists (name, user_id) VALUES (?,?)", (name, uid))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return {"status": "ok", "id": new_id}

@app.delete("/playlists/{playlist_id}")
def delete_playlist(request: Request, playlist_id: int):
    uid = _require_login(request)
    conn = get_db()
    if not _owns_playlist(conn, uid, playlist_id):
        conn.close(); raise HTTPException(status_code=403, detail="not your playlist")
    conn.execute("DELETE FROM playlist_videos WHERE playlist_id=?", (playlist_id,))
    conn.execute("DELETE FROM playlists WHERE id=?", (playlist_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/playlists/{playlist_id}/videos")
def get_playlist_videos(request: Request, playlist_id: int):
    uid = _require_login(request)
    conn = get_db()
    if not _owns_playlist(conn, uid, playlist_id):
        conn.close(); raise HTTPException(status_code=403, detail="not your playlist")
    rows = conn.execute("SELECT * FROM playlist_videos WHERE playlist_id=? ORDER BY id",
                        (playlist_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/playlists/{playlist_id}/add")
def add_to_playlist(request: Request, playlist_id: int, video_id: str, title: str = "", thumbnail: str = "", channel: str = ""):
    uid = _require_login(request)
    conn = get_db()
    if not _owns_playlist(conn, uid, playlist_id):
        conn.close(); raise HTTPException(status_code=403, detail="not your playlist")
    conn.execute("INSERT INTO playlist_videos (playlist_id, video_id, title, thumbnail, channel) VALUES (?,?,?,?,?)",
                 (playlist_id, video_id, title, thumbnail, channel))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/playlists/{playlist_id}/add_bulk")
async def add_to_playlist_bulk(playlist_id: int, request: Request):
    uid = _require_login(request)
    data = await request.json()
    videos = data.get("videos", [])
    conn = get_db()
    if not _owns_playlist(conn, uid, playlist_id):
        conn.close(); raise HTTPException(status_code=403, detail="not your playlist")
    conn.executemany(
        "INSERT INTO playlist_videos (playlist_id, video_id, title, thumbnail, channel) VALUES (?,?,?,?,?)",
        [(playlist_id, v.get("video_id"), v.get("title",""), v.get("thumbnail",""), v.get("channel",""))
         for v in videos if v.get("video_id")]
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "count": len(videos)}

@app.delete("/playlists/{playlist_id}/remove/{video_id}")
def remove_from_playlist(request: Request, playlist_id: int, video_id: str):
    uid = _require_login(request)
    conn = get_db()
    if not _owns_playlist(conn, uid, playlist_id):
        conn.close(); raise HTTPException(status_code=403, detail="not your playlist")
    conn.execute("DELETE FROM playlist_videos WHERE playlist_id=? AND video_id=?",
                 (playlist_id, video_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}