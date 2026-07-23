"""Звуковая тревога при детекции сонливости (браузер, Web Audio API)."""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

DEFAULT_COOLDOWN_SEC = 5.0


def play_drowsiness_alert_sound() -> None:
    """Короткая сирена в браузере (без внешних файлов)."""
    nonce = st.session_state.get("_alert_sound_nonce", 0) + 1
    st.session_state["_alert_sound_nonce"] = nonce
    components.html(
        f"""
        <script>
        (async function() {{
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            if (ctx.state === "suspended") {{
                try {{ await ctx.resume(); }} catch (e) {{ return; }}
            }}
            function tone(freq, start, dur, volume) {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = "square";
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0.0001, start);
                gain.gain.exponentialRampToValueAtTime(volume, start + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(start);
                osc.stop(start + dur + 0.05);
            }}
            const t = ctx.currentTime;
            tone(880, t, 0.22, 0.22);
            tone(660, t + 0.28, 0.22, 0.22);
            tone(880, t + 0.56, 0.32, 0.26);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def handle_drowsiness_alert(status: str, cooldown_sec: float | None = None) -> None:
    """Сигнал при статусе «Спит», не чаще cooldown_sec."""
    if not st.session_state.get("sound_alert_enabled", True):
        return

    if status != "Спит":
        return

    pause = cooldown_sec if cooldown_sec is not None else st.session_state.get(
        "alert_cooldown_sec", DEFAULT_COOLDOWN_SEC
    )
    now = time.time()
    last = st.session_state.get("_drowsy_alert_last_ts", 0.0)
    if now - last < pause:
        return

    st.session_state["_drowsy_alert_last_ts"] = now
    play_drowsiness_alert_sound()
