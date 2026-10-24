"""Bask Health intake agent: one LLM turn with tool calls, plus a scripted fallback.

The LLM runs on Groq's free tier when GROQ_API_KEY is set. Without a key, or if the
API fails, a rule-based fallback answers so the demo never breaks.
"""
import json
import os
import re
import time
import uuid

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You are the voice intake agent for Bask Health, a telehealth company.
Speak like a friendly receptionist on a phone call: at most two short sentences per reply.

You can:
- check_availability: look up open slots. Never invent a doctor or a time, always call the tool.
- book_appointment: book once you have the patient's name, email, doctor and slot.
- escalate_to_human: hand the call to a person.

Hard rules:
- Never give medical, medication, dosage or drug-interaction advice. Escalate to a clinician instead.
- If the caller is angry, upset, or asks for a person, escalate to a human agent.
- After booking, read back the doctor, the time and that a confirmation email is on its way."""

# Fictional schedule for the demo.
DOCTORS = {
    "Dr. Sara Malik": {"specialty": "primary care", "slots": ["Tue 10:00", "Thu 14:00", "Thu 16:30"]},
    "Dr. James Lee": {"specialty": "dermatology", "slots": ["Wed 09:30", "Thu 15:00"]},
    "Dr. Amina Patel": {"specialty": "mental health", "slots": ["Mon 11:00", "Fri 13:00"]},
}

BOOKINGS = []
ESCALATIONS = []

TOOLS = [
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "List open appointment slots, optionally for one doctor or specialty.",
        "parameters": {"type": "object", "properties": {
            "doctor_or_specialty": {"type": "string", "description": "Doctor name or specialty, empty for all"}}}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Book a slot and send the confirmation email.",
        "parameters": {"type": "object", "properties": {
            "patient_name": {"type": "string"}, "email": {"type": "string"},
            "doctor": {"type": "string"}, "slot": {"type": "string"}},
            "required": ["patient_name", "email", "doctor", "slot"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the call to a human agent or clinician.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "enum": ["angry_caller", "medical_advice", "caller_request", "other"]},
            "summary": {"type": "string"}},
            "required": ["reason", "summary"]}}},
]


STEMS = {"Dr. Sara Malik": ["malik", "primary", "family doctor", "gp"],
         "Dr. James Lee": ["lee", "dermatolog", "skin"],
         "Dr. Amina Patel": ["patel", "mental", "therap", "psych"]}


def _match_doctor(text):
    text = (text or "").lower()
    for name, stems in STEMS.items():
        if any(s in text for s in stems):
            return name
    return None


DAYS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}


def _filter_slots(slots, text):
    text = (text or "").lower()
    day = next((v for k, v in DAYS.items() if k in text), None)
    out = [s for s in slots if not day or s.startswith(day)]
    if "afternoon" in text:
        out = [s for s in out if int(s.split()[1][:2]) >= 12]
    elif "morning" in text:
        out = [s for s in out if int(s.split()[1][:2]) < 12]
    return out


def check_availability(doctor_or_specialty=""):
    name = _match_doctor(doctor_or_specialty)
    docs = {name: DOCTORS[name]} if name else DOCTORS
    return {n: {"specialty": d["specialty"], "open_slots": d["slots"]} for n, d in docs.items()}


def book_appointment(patient_name, email, doctor, slot):
    name = _match_doctor(doctor)
    if not name:
        return {"ok": False, "error": f"No doctor matching '{doctor}'"}
    if slot not in DOCTORS[name]["slots"]:
        return {"ok": False, "error": f"{slot} is not open for {name}", "open_slots": DOCTORS[name]["slots"]}
    DOCTORS[name]["slots"].remove(slot)
    ref = "BSK-" + uuid.uuid4().hex[:6].upper()
    email_text = (f"To: {email}\nSubject: Your Bask Health appointment {ref}\n\n"
                  f"Hi {patient_name}, you're booked with {name} on {slot}. "
                  f"Join from the Bask app 5 minutes early. Reply to reschedule.")
    BOOKINGS.append({"ref": ref, "patient": patient_name, "doctor": name, "slot": slot, "email": email_text})
    return {"ok": True, "ref": ref, "doctor": name, "slot": slot, "confirmation_email": "sent (simulated)"}


def escalate_to_human(reason, summary):
    ESCALATIONS.append({"reason": reason, "summary": summary, "at": time.strftime("%H:%M:%S")})
    return {"ok": True, "handoff": "clinician" if reason == "medical_advice" else "human agent"}


IMPLS = {"check_availability": check_availability, "book_appointment": book_appointment,
         "escalate_to_human": escalate_to_human}


def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq
    return Groq(api_key=key)


def run_turn(history):
    """Run one agent turn. history: OpenAI-style messages without the system prompt.

    Returns (reply_text, tools_used, llm_ms, source)."""
    client = _client()
    if client is not None:
        try:
            return _llm_turn(client, history)
        except Exception as e:  # rate limit, network: fall back rather than fail on stage
            print("LLM error, using fallback:", e)
    t0 = time.perf_counter()
    reply, tools = _fallback_turn(history)
    return reply, tools, (time.perf_counter() - t0) * 1000, "scripted fallback"


def _llm_turn(client, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    tools_used = []
    t0 = time.perf_counter()
    for _ in range(4):  # at most a few tool rounds per turn
        resp = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS,
                                              tool_choice="auto", temperature=0.3, max_tokens=400,
                                              reasoning_effort="low")
        m = resp.choices[0].message
        if not m.tool_calls:
            return m.content or "", tools_used, (time.perf_counter() - t0) * 1000, f"Groq {MODEL}"
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                    for c in m.tool_calls]})
        for c in m.tool_calls:
            args = json.loads(c.function.arguments or "{}")
            result = IMPLS[c.function.name](**args)
            tools_used.append({"tool": c.function.name, "args": args, "result": result})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    return "Let me get someone to help you with that.", tools_used, (time.perf_counter() - t0) * 1000, f"Groq {MODEL}"


ANGRY = re.compile(r"\b(angry|ridiculous|third time|unacceptable|real person|human|manager|furious)\b", re.I)
MEDICAL = re.compile(r"\b(mg|dose|dosage|ibuprofen|sertraline|medication|drug|pill|take .* with)\b", re.I)
BOOK = re.compile(r"\b(book|schedule|works|appointment)\b", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _fallback_turn(history):
    text = history[-1]["content"]
    if MEDICAL.search(text):
        r = escalate_to_human("medical_advice", text[:120])
        return "I can't advise on medication, but I'm connecting you to a clinician right now.", \
            [{"tool": "escalate_to_human", "args": {"reason": "medical_advice"}, "result": r}]
    if ANGRY.search(text):
        r = escalate_to_human("angry_caller", text[:120])
        return "I'm sorry this has been so frustrating. I'm transferring you to a team member now.", \
            [{"tool": "escalate_to_human", "args": {"reason": "angry_caller"}, "result": r}]
    email = EMAIL.search(text)
    doctor = _match_doctor(" ".join(m["content"] for m in history if m["role"] == "user"))
    if email and doctor and DOCTORS[doctor]["slots"]:
        name = re.search(r"name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)
        slot = (_filter_slots(DOCTORS[doctor]["slots"], text) or DOCTORS[doctor]["slots"])[0]
        r = book_appointment(name.group(1) if name else "Patient", email.group(0), doctor, slot)
        return f"You're booked with {doctor} on {slot}. Your confirmation email is on its way.", \
            [{"tool": "book_appointment", "args": {"doctor": doctor, "slot": slot}, "result": r}]
    if doctor or "available" in text.lower() or BOOK.search(text):
        r = check_availability(doctor or "")
        slots = "; ".join(f"{n}: {', '.join(_filter_slots(d['open_slots'], text) or ['nothing that matches'])}"
                          for n, d in r.items())
        ask = " What name and email should I book under?" if BOOK.search(text) else ""
        return f"Here's what's open: {slots}.{ask}", [{"tool": "check_availability", "args": {}, "result": r}]
    return "I can help you find a doctor or book an appointment. What do you need today?", []
