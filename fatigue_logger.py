"""Локальные логи эпизодов сонливости (JSONL + экспорт)."""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

LOG_DIR = Path(__file__).parent / "data"
LOG_FILE = LOG_DIR / "fatigue_log.jsonl"
DEFAULT_LOG_COOLDOWN_SEC = 10.0


def ensure_session_id() -> str:
    if "fatigue_session_id" not in st.session_state:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.fatigue_session_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    return st.session_state.fatigue_session_id


def _build_event(
    result: Any,
    mode: str,
    event_type: str,
    nearest_stop: dict | None,
) -> dict:
    event = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "session_id": ensure_session_id(),
        "event": event_type,
        "mode": mode,
        "status": result.status,
        "ear": result.ear,
        "left_ear": result.left_ear,
        "right_ear": result.right_ear,
        "confidence": result.confidence,
        "closed_frames": result.closed_frames,
    }
    if nearest_stop:
        event["nearest_stop_name"] = nearest_stop.get("name")
        event["nearest_stop_km"] = nearest_stop.get("distance_km")
    return event


def append_event(event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(limit: int = 500, session_id: str | None = None) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    events: list[dict] = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if session_id:
        events = [e for e in events if e.get("session_id") == session_id]
    return events[-limit:]


def events_to_csv(events: list[dict]) -> str:
    if not events:
        return "logged_at,session_id,event,mode,status,ear,left_ear,right_ear,confidence,closed_frames,nearest_stop_name,nearest_stop_km\n"
    fields = [
        "logged_at",
        "session_id",
        "event",
        "mode",
        "status",
        "ear",
        "left_ear",
        "right_ear",
        "confidence",
        "closed_frames",
        "nearest_stop_name",
        "nearest_stop_km",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for event in events:
        writer.writerow(event)
    return buf.getvalue()


def handle_fatigue_log(
    result: Any,
    mode: str,
    nearest_stop: dict | None = None,
    cooldown_sec: float | None = None,
) -> None:
    """Пишет эпизоды сонливости: start / continue / end."""
    if not st.session_state.get("fatigue_log_enabled", True):
        return

    pause = cooldown_sec if cooldown_sec is not None else st.session_state.get(
        "fatigue_log_cooldown_sec", DEFAULT_LOG_COOLDOWN_SEC
    )
    was_drowsy = bool(st.session_state.get("_fatigue_episode_active"))

    if result.status != "Спит":
        if was_drowsy:
            append_event(_build_event(result, mode, "drowsy_end", nearest_stop))
            st.session_state._fatigue_episode_active = False
        return

    now = time.time()
    if not was_drowsy:
        append_event(_build_event(result, mode, "drowsy_start", nearest_stop))
        st.session_state._fatigue_episode_active = True
        st.session_state._fatigue_log_last_ts = now
        return

    last = st.session_state.get("_fatigue_log_last_ts", 0.0)
    if now - last < pause:
        return

    st.session_state._fatigue_log_last_ts = now
    append_event(_build_event(result, mode, "drowsy_continue", nearest_stop))


def render_fatigue_log_sidebar() -> None:
    """UI логов в sidebar."""
    st.markdown("---")
    st.subheader("📝 Логи усталости")
    st.toggle("Записывать эпизоды", value=True, key="fatigue_log_enabled")
    st.slider(
        "Пауза между записями (сек)",
        min_value=5,
        max_value=60,
        value=10,
        step=1,
        key="fatigue_log_cooldown_sec",
    )

    session_id = ensure_session_id()
    st.caption(f"Сессия: `{session_id}`")

    session_events = read_events(limit=200, session_id=session_id)
    with st.expander(f"Записи сессии ({len(session_events)})"):
        if not session_events:
            st.caption("Пока пусто — закрой глаза ~1 сек при включённой камере.")
        for event in reversed(session_events[-12:]):
            ts = event.get("logged_at", "")[:19].replace("T", " ")
            label = event.get("event", "?")
            ear = event.get("ear", "—")
            stop = event.get("nearest_stop_name")
            extra = f" · {stop}" if stop else ""
            st.caption(f"{ts} · {label} · EAR {ear}{extra}")

    all_events = read_events(limit=500)
    if LOG_FILE.exists() and all_events:
        st.download_button(
            "⬇️ Скачать JSONL",
            data=LOG_FILE.read_bytes(),
            file_name="fatigue_log.jsonl",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Скачать CSV",
            data=events_to_csv(all_events).encode("utf-8-sig"),
            file_name="fatigue_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
