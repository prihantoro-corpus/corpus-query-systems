import os
import json
import time
import requests
import subprocess
import tempfile
import streamlit as st

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYTICS_FILE = os.path.join(_ROOT_DIR, "cortex_analytics.json")

def init_analytics():
    """Ensures the analytics store exists and is valid."""
    if not os.path.exists(ANALYTICS_FILE):
        data = {
            "sessions": [],
            "total_access_count": 0,
            "created_at": "2026-03-01 00:00:00"
        }
        _write_analytics(data)

def _read_analytics():
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"sessions": [], "total_access_count": 0, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}

def _write_analytics(data):
    try:
        with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Analytics write warning: {e}")

def get_client_ip_geo():
    """
    Returns zero-cost local session info without network calls or IP geotracking.
    """
    return {
        "ip": "Local/Private",
        "city": "Local",
        "region": "Local",
        "country": "Local",
        "org": "Local Machine"
    }

def track_session_access():
    """Lightweight session access tracker."""
    pass

def update_session_duration():
    """Lightweight session duration tracker."""
    pass

@st.cache_data(ttl=3600, show_spinner=False)
def get_analytics_summary():
    """Returns analytics summary data cached for 1 hour."""
    data = _read_analytics()
    sessions = data.get("sessions", [])
    now = time.time()
    for s in sessions:
        start = s.get("start_timestamp", now)
        heartbeat = s.get("last_heartbeat", start)
        duration_sec = max(0, heartbeat - start)
        s["duration_formatted"] = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s" if duration_sec >= 60 else f"{int(duration_sec)}s"

    return {
        "total_accesses": data.get("total_access_count", len(sessions)),
        "active_sessions_count": len(sessions),
        "recent_sessions": sessions,
        "tracked_since": data.get("created_at", "2026-03-01 00:00:00")[:10]
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_git_maintenance_log(limit=15):
    """
    Fetches maintenance log cached for 1 hour to prevent Git CLI subprocess overhead on reruns.
    """
    try:
        cmd = ["git", "log", f"-n{limit}", "--pretty=format:%h|%an|%ar|%s"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = []
        for line in res.stdout.splitlines():
            if "|" in line:
                h, author, time_ago, subject = line.split("|", 3)
                logs.append({
                    "commit": h,
                    "author": author,
                    "time": time_ago,
                    "message": subject
                })
        if logs:
            return logs
    except Exception:
        pass

    return []
