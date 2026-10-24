"""Core metrics from the submission, tracked per turn and per call, with threshold alerts."""
import time

# Thresholds taken from the submission's core metrics (section 1).
TTFA_LIMIT_MS = 600
E2E_LIMIT_MS = 1200
TTFA_WARN_MS = 575  # section 3 ramp pause: half of the 50 ms margin

TURNS, CALLS, ALERTS = [], [], []


def record_turn(call_id, stt_ms, llm_ms, tts_first_ms, tts_full_ms, tools, source):
    ttfa = stt_ms + llm_ms + tts_first_ms  # caller stops talking -> first audio plays
    e2e = stt_ms + llm_ms + tts_full_ms    # caller stops talking -> full reply ready
    turn = {"call": call_id, "at": time.strftime("%H:%M:%S"), "stt_ms": round(stt_ms), "llm_ms": round(llm_ms),
            "tts_ms": round(tts_first_ms), "ttfa_ms": round(ttfa), "e2e_ms": round(e2e),
            "tools": ", ".join(t["tool"] for t in tools) or "-", "source": source}
    TURNS.append(turn)
    if ttfa > TTFA_LIMIT_MS:
        _alert("TTFA", f"{round(ttfa)} ms against the 600 ms target")
    elif ttfa > TTFA_WARN_MS:
        _alert("TTFA warning", f"{round(ttfa)} ms passed the 575 ms ramp-pause line")
    if e2e > E2E_LIMIT_MS:
        _alert("E2E", f"{round(e2e)} ms against the 1.2 s target")
    return turn


def close_call(call_id, escalated, booked):
    CALLS.append({"call": call_id, "escalated": escalated, "booked": booked})


def _alert(metric, detail):
    ALERTS.append({"at": time.strftime("%H:%M:%S"), "metric": metric, "detail": detail,
                   "action": "TPM notified (simulated)"})


def _p95(values):
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def summary():
    n = len(CALLS)
    contained = sum(1 for c in CALLS if not c["escalated"])
    return {
        "Calls": n,
        "Containment (resolved by the agent)": f"{100 * contained / n:.0f}%" if n else "-",
        "Handed to a human": sum(1 for c in CALLS if c["escalated"]),
        "Bookings": sum(1 for c in CALLS if c["booked"]),
        "P95 TTFA (target under 600 ms)": f"{_p95([t['ttfa_ms'] for t in TURNS])} ms" if TURNS else "-",
        "P95 E2E (target under 1.2 s)": f"{_p95([t['e2e_ms'] for t in TURNS])} ms" if TURNS else "-",
        "Alerts": len(ALERTS),
    }
