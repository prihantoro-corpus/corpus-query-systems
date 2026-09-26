import os
import json
import time
import requests
import subprocess
import tempfile
import streamlit as st

ANALYTICS_FILE = os.path.join(tempfile.gettempdir(), "cortex_analytics.json")

def init_analytics():
    """Ensures the analytics store exists and is valid."""
    if not os.path.exists(ANALYTICS_FILE):
        data = {
            "sessions": [],
            "total_access_count": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
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
    Attempts to retrieve geographic information for the active user session.
    Uses multi-provider fallback (ipwho.is, ip-api.com) for high reliability.
    """
    try:
        # Check Streamlit headers if available
        user_ip = "Local/Private"
        try:
            from streamlit import context
            if hasattr(context, "headers"):
                forwarded = context.headers.get("x-forwarded-for")
                if forwarded:
                    user_ip = forwarded.split(",")[0].strip()
                elif context.headers.get("host"):
                    user_ip = context.headers.get("host")
        except Exception:
            pass

        # Perform geo lookup (cached per session if valid)
        if "geo_info" in st.session_state and st.session_state["geo_info"].get("city") not in ("Unknown", "Unknown / Localhost"):
            return st.session_state["geo_info"]

        geo_data = {
            "ip": user_ip,
            "city": "Unknown",
            "region": "Unknown",
            "country": "Unknown",
            "org": "Unknown ISP"
        }

        # 1. Provider 1: ipwho.is (fast, JSON, works with IPv4/v6)
        try:
            target_url = f"https://ipwho.is/{user_ip}" if user_ip not in ("Local/Private", "localhost", "127.0.0.1") else "https://ipwho.is/"
            res = requests.get(target_url, timeout=3, headers={"User-Agent": "CortexCorpusApp/1.0"})
            if res.status_code == 200:
                d = res.json()
                if d.get("success", True):
                    geo_data = {
                        "ip": d.get("ip", user_ip),
                        "city": d.get("city", "Unknown City"),
                        "region": d.get("region", "Unknown Region"),
                        "country": d.get("country", "Unknown Country"),
                        "org": d.get("connection", {}).get("isp") or d.get("connection", {}).get("org") or "Unknown ISP"
                    }
                    st.session_state["geo_info"] = geo_data
                    return geo_data
        except Exception:
            pass

        # 2. Provider 2: ip-api.com (fallback)
        try:
            target_url = f"http://ip-api.com/json/{user_ip}" if user_ip not in ("Local/Private", "localhost", "127.0.0.1") else "http://ip-api.com/json/"
            res = requests.get(target_url, timeout=3)
            if res.status_code == 200:
                d = res.json()
                if d.get("status") == "success":
                    geo_data = {
                        "ip": d.get("query", user_ip),
                        "city": d.get("city", "Unknown City"),
                        "region": d.get("regionName", "Unknown Region"),
                        "country": d.get("country", "Unknown Country"),
                        "org": d.get("isp") or d.get("org") or "Unknown ISP"
                    }
                    st.session_state["geo_info"] = geo_data
                    return geo_data
        except Exception:
            pass

        st.session_state["geo_info"] = geo_data
        return geo_data
    except Exception as e:
        return {
            "ip": "Unknown",
            "city": "Unknown",
            "region": "Unknown",
            "country": "Unknown",
            "org": str(e)
        }

def track_session_access():
    """
    Tracks session access, total access count, and session duration.
    Should be called once per session start.
    """
    init_analytics()
    
    if "session_tracked" not in st.session_state:
        st.session_state["session_tracked"] = True
        st.session_state["session_start_time"] = time.time()

        geo = get_client_ip_geo()
        data = _read_analytics()
        data["total_access_count"] = data.get("total_access_count", 0) + 1

        session_entry = {
            "id": f"sess_{int(time.time())}",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_timestamp": time.time(),
            "city": geo.get("city", "Unknown"),
            "region": geo.get("region", "Unknown"),
            "country": geo.get("country", "Unknown"),
            "org": geo.get("org", "Unknown"),
            "last_heartbeat": time.time()
        }
        
        # Keep last 100 sessions
        data["sessions"].insert(0, session_entry)
        data["sessions"] = data["sessions"][:100]
        st.session_state["session_id"] = session_entry["id"]
        _write_analytics(data)

def update_session_duration():
    """Updates the active session's duration/heartbeat."""
    if "session_id" in st.session_state:
        data = _read_analytics()
        for sess in data.get("sessions", []):
            if sess.get("id") == st.session_state["session_id"]:
                sess["last_heartbeat"] = time.time()
                break
        _write_analytics(data)

def get_analytics_summary():
    """Returns analytics summary data for the UI."""
    data = _read_analytics()
    sessions = data.get("sessions", [])
    
    # Calculate duration for sessions
    now = time.time()
    for s in sessions:
        start = s.get("start_timestamp", now)
        heartbeat = s.get("last_heartbeat", start)
        duration_sec = max(0, heartbeat - start)
        s["duration_formatted"] = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s" if duration_sec >= 60 else f"{int(duration_sec)}s"

    return {
        "total_accesses": data.get("total_access_count", len(sessions)),
        "active_sessions_count": len(sessions),
        "recent_sessions": sessions
    }

def get_git_maintenance_log(limit=15):
    """
    Fetches automated maintenance log from Git commits (local or GitHub REST API).
    """
    # 1. Local Git CLI
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

    # 2. GitHub REST API (Online Fallback)
    try:
        url = f"https://api.github.com/repos/prihantoro-corpus/cortex/commits?per_page={limit}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            items = res.json()
            logs = []
            for item in items:
                logs.append({
                    "commit": item['sha'][:7],
                    "author": item['commit']['author']['name'],
                    "time": item['commit']['author']['date'][:10],
                    "message": item['commit']['message'].split('\n')[0]
                })
            return logs
    except Exception as e:
        print(f"GitHub API Error: {e}")

    return []
