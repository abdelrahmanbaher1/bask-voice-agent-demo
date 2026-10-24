"""Bask Health voice intake agent demo: speech in, tool-calling agent, speech out, core metrics live."""
import asyncio
import os
import tempfile
import time
import uuid

import gradio as gr

import agent
import metrics
from personas import PERSONAS

STT_MODEL = "whisper-large-v3-turbo"
VOICE = "en-US-AriaNeural"


def transcribe(path):
    client = agent._client()
    if client is None or not path:
        return None, 0.0
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        text = client.audio.transcriptions.create(file=f, model=STT_MODEL).text
    return text.strip(), (time.perf_counter() - t0) * 1000


async def _stream_tts(text, out):
    import edge_tts
    t0 = time.perf_counter()
    first = None
    with open(out, "wb") as f:
        async for chunk in edge_tts.Communicate(text, VOICE).stream():
            if chunk["type"] == "audio":
                if first is None:
                    first = (time.perf_counter() - t0) * 1000  # first audio chunk: what TTFA counts
                f.write(chunk["data"])
    return first or 0.0, (time.perf_counter() - t0) * 1000


def speak(text):
    """Returns (audio_path, ms to first audio chunk, ms to full audio)."""
    try:
        out = os.path.join(tempfile.gettempdir(), f"reply-{uuid.uuid4().hex[:8]}.mp3")
        first, full = asyncio.run(_stream_tts(text, out))
        return out, first, full
    except Exception as e:
        print("TTS unavailable:", e)
        return None, 0.0, 0.0


def new_call():
    return {"id": "call-" + uuid.uuid4().hex[:5], "history": [], "escalated": False, "booked": False}


def _tool_note(tools):
    lines = []
    for t in tools:
        r = t["result"]
        if t["tool"] == "book_appointment" and r.get("ok"):
            lines.append(f"🛠 book_appointment → {r['ref']}, {r['doctor']} {r['slot']}, confirmation email sent (simulated)")
        elif t["tool"] == "escalate_to_human":
            lines.append(f"🛠 escalate_to_human → {r['handoff']} ({t['args'].get('reason', '')})")
        else:
            lines.append(f"🛠 {t['tool']}")
    return "\n".join(lines)


def agent_step(state, user_text, stt_ms=0.0, with_voice=True):
    state["history"].append({"role": "user", "content": user_text})
    reply, tools, llm_ms, source = agent.run_turn(state["history"])
    state["history"].append({"role": "assistant", "content": reply})
    state["escalated"] |= any(t["tool"] == "escalate_to_human" for t in tools)
    state["booked"] |= any(t["tool"] == "book_appointment" and t["result"].get("ok") for t in tools)
    audio, tts_first, tts_full = speak(reply) if with_voice else (None, 0.0, 0.0)
    turn = metrics.record_turn(state["id"], stt_ms, llm_ms, tts_first, tts_full, tools, source)
    return reply, tools, audio, turn


def _turn_md(turn):
    return ('<div class="grid">'
            + _card("Speech-to-text", f"{turn['stt_ms']} ms" if turn["stt_ms"] else "-", "Whisper" if turn["stt_ms"] else "Typed or scripted input")
            + _card("LLM", f"{turn['llm_ms']} ms", turn["source"])
            + _card("First audio chunk", f"{turn['tts_ms']} ms", "Text-to-speech")
            + _card("TTFA", f"{turn['ttfa_ms']} ms", "Target under 600 ms", turn["ttfa_ms"] <= 600)
            + _card("E2E", f"{turn['e2e_ms']} ms", "Target under 1.2 s", turn["e2e_ms"] <= 1200)
            + "</div>")


def on_talk(audio_path, text, state, chat):
    state = state or new_call()
    stt_ms = 0.0
    if audio_path:
        heard, stt_ms = transcribe(audio_path)
        if heard is None:
            return chat, None, state, '<p class="muted">Voice needs GROQ_API_KEY. Type your message instead.</p>', None, ""
        text = heard
    if not (text or "").strip():
        return chat, None, state, '<p class="muted">Say or type something first.</p>', None, ""
    reply, tools, audio, turn = agent_step(state, text, stt_ms)
    chat = chat + [{"role": "user", "content": text}]
    note = _tool_note(tools)
    chat = chat + [{"role": "assistant", "content": reply + (f"\n\n_{note}_" if note else "")}]
    return chat, audio, state, _turn_md(turn), None, ""


def on_end_call(state):
    if state and state["history"]:
        metrics.close_call(state["id"], state["escalated"], state["booked"])
    return [], None, new_call(), '<p class="muted">Call ended and logged. Start a new one any time.</p>'


def on_persona(name):
    state = new_call()
    chat, last = [], None
    for line in PERSONAS[name]:
        reply, tools, audio, turn = agent_step(state, line)
        note = _tool_note(tools)
        chat += [{"role": "user", "content": line},
                 {"role": "assistant", "content": reply + (f"\n\n_{note}_" if note else "")}]
        last = audio or last
    metrics.close_call(state["id"], state["escalated"], state["booked"])
    outcome = "handed to a human" if state["escalated"] else ("booked" if state["booked"] else "resolved by the agent")
    return chat, last, f'<p class="muted">Test call {state["id"]} finished: <b>{outcome}</b>. Last turn:</p>' + _turn_md(turn)


def on_dashboard():
    s = metrics.summary()
    t = [x["ttfa_ms"] for x in metrics.TURNS]
    e = [x["e2e_ms"] for x in metrics.TURNS]
    p95t, p95e = metrics._p95(t), metrics._p95(e)
    md = ('<div class="grid">'
          + _card("Calls", s["Calls"], "Logged when a call ends")
          + _card("Containment", s["Containment (resolved by the agent)"], "Resolved by the agent")
          + _card("Handed to a human", s["Handed to a human"], "Angry callers, clinical questions")
          + _card("Bookings", s["Bookings"], "Confirmation email sent")
          + _card("P95 TTFA", f"{p95t} ms" if t else "-", "Target under 600 ms", (p95t <= 600) if t else None)
          + _card("P95 E2E", f"{p95e} ms" if e else "-", "Target under 1.2 s", (p95e <= 1200) if e else None)
          + _card("Alerts to the TPM", s["Alerts"], "One per missed target")
          + "</div>")
    turns = [[t["at"], t["call"], t["stt_ms"], t["llm_ms"], t["tts_ms"], t["ttfa_ms"], t["e2e_ms"], t["tools"], t["source"]]
             for t in reversed(metrics.TURNS[-50:])]
    alerts = [[a["at"], a["metric"], a["detail"], a["action"]] for a in reversed(metrics.ALERTS[-50:])]
    emails = "\n\n---\n\n".join(f"```\n{b['email']}\n```" for b in reversed(agent.BOOKINGS[-5:])) or "_No bookings yet._"
    return md, turns, alerts, emails


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.zinc, neutral_hue=gr.themes.colors.zinc, radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#ffffff", block_background_fill="#ffffff", block_border_color="#e4e4e7",
    block_border_width="1px", block_shadow="0 1px 2px rgba(0,0,0,0.04)", block_label_text_color="#71717a",
    block_title_text_color="#09090b", border_color_primary="#e4e4e7", input_background_fill="#ffffff",
    input_border_color="#e4e4e7", button_primary_background_fill="#18181b", button_primary_background_fill_hover="#27272a",
    button_primary_text_color="#fafafa", button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f4f4f5", button_secondary_border_color="#e4e4e7",
    button_secondary_text_color="#18181b", body_text_color="#09090b", body_text_color_subdued="#71717a",
)

CSS = """
.gradio-container {max-width: 1100px !important; margin: 0 auto !important;}
.hdr h1 {font-size: 28px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px;}
.hdr p {color: #71717a; font-size: 14px; margin: 0;}
.pill {display: inline-block; font-size: 12px; font-weight: 500; padding: 2px 10px; border-radius: 9999px;
       border: 1px solid #e4e4e7; background: #f4f4f5; color: #18181b; margin-top: 8px;}
.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;}
.card {border: 1px solid #e4e4e7; border-radius: 12px; padding: 14px 16px; background: #fff;
       box-shadow: 0 1px 2px rgba(0,0,0,0.04);}
.card .label {font-size: 13px; color: #71717a;}
.card .value {font-size: 24px; font-weight: 600; color: #09090b; margin-top: 2px; letter-spacing: -0.01em;}
.card .sub {font-size: 12px; color: #71717a; margin-top: 4px;}
.badge {display: inline-block; font-size: 11px; font-weight: 500; padding: 1px 8px; border-radius: 9999px; border: 1px solid;}
.ok {color: #15803d; background: #f0fdf4; border-color: #bbf7d0;}
.bad {color: #b91c1c; background: #fef2f2; border-color: #fecaca;}
.muted {color: #71717a; font-size: 13px;}
"""


def _card(label, value, sub="", status=None):
    badge = "" if status is None else f' <span class="badge {"ok" if status else "bad"}">{"on target" if status else "missed"}</span>'
    return (f'<div class="card"><div class="label">{label}{badge}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


MODE = "Live: Groq Whisper + gpt-oss" if os.getenv("GROQ_API_KEY") else "Offline: scripted replies (set GROQ_API_KEY for the live model)"

with gr.Blocks(title="Bask Health voice agent demo") as demo:
    gr.HTML(
        '<div class="hdr"><h1>Bask Health voice intake agent</h1>'
        "<p>A working prototype for the TPM submission: speech-to-text, an LLM with tools and text-to-speech, "
        "measured live against the submission's core metrics. All patients, doctors and bookings are fictional.</p>"
        f'<span class="pill">{MODE}</span></div>')
    state = gr.State(new_call())
    with gr.Tab("Talk to the agent"):
        chat = gr.Chatbot(height=360, label="Call")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
            txt = gr.Textbox(label="Or type", placeholder="e.g. Is Dr. Malik free on Thursday?")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            end = gr.Button("End call")
        voice = gr.Audio(label="Agent voice", autoplay=True)
        last_turn = gr.HTML()
        send.click(on_talk, [mic, txt, state, chat], [chat, voice, state, last_turn, mic, txt])
        txt.submit(on_talk, [mic, txt, state, chat], [chat, voice, state, last_turn, mic, txt])
        mic.stop_recording(on_talk, [mic, txt, state, chat], [chat, voice, state, last_turn, mic, txt])
        end.click(on_end_call, [state], [chat, voice, state, last_turn])
    with gr.Tab("Run a test patient"):
        persona = gr.Dropdown(list(PERSONAS), value=list(PERSONAS)[0], label="Test patient")
        run = gr.Button("Run call", variant="primary")
        p_chat = gr.Chatbot(height=360, label="Transcript")
        p_voice = gr.Audio(label="Agent voice (last reply)")
        p_out = gr.HTML()
        run.click(on_persona, [persona], [p_chat, p_voice, p_out])
    with gr.Tab("Core metrics dashboard"):
        refresh = gr.Button("Refresh", variant="primary")
        summary = gr.HTML()
        turns = gr.Dataframe(headers=["Time", "Call", "STT ms", "LLM ms", "TTS first chunk ms", "TTFA ms", "E2E ms", "Tools", "Source"],
                             label="Turns, newest first")
        alerts = gr.Dataframe(headers=["Time", "Metric", "Detail", "Action"], label="Alerts sent to the TPM")
        emails = gr.Markdown(label="Confirmation emails")
        refresh.click(on_dashboard, None, [summary, turns, alerts, emails])
        demo.load(on_dashboard, None, [summary, turns, alerts, emails])

if __name__ == "__main__":
    port = os.getenv("PORT")  # set by Render and similar hosts
    demo.launch(theme=THEME, css=CSS, server_name="0.0.0.0" if port else None,
                server_port=int(port) if port else None)
