---
title: Bask Health Voice Agent Demo
emoji: 📞
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Bask Health voice intake agent, a working prototype

A small companion to my TPM submission on the real-time AI voice agent. It runs the same pipeline shape as the recommended architecture (speech-to-text, an LLM with tools, text-to-speech) and measures every turn against the submission's core metrics, with alerts when a target is missed.

**All patients, doctors, bookings and emails are fictional. No real patient data is used or stored.**

## What it does

| Tab | What you see |
|---|---|
| Talk to the agent | Speak or type. Speech-to-text (Groq Whisper), an LLM agent with tools (Groq gpt-oss-20b), spoken reply (Edge TTS). Each turn shows its timing. |
| Run a test patient | Four scripted callers: availability question, booking, angry caller, drug question. |
| Core metrics dashboard | Calls, containment, handoffs, bookings, P95 TTFA and E2E, every turn's timing, alerts sent to the TPM, and the confirmation emails. |

The agent has three tools: `check_availability`, `book_appointment` (writes the booking and drafts the confirmation email) and `escalate_to_human`. Two guardrails from the submission are hard rules in the prompt: it never gives medication advice (it hands off to a clinician), and it hands angry callers to a person.

## How it maps to the submission

| Submission | Prototype |
|---|---|
| Core metrics (section 1) | TTFA under 600 ms, E2E under 1.2 s, containment, handoffs, bookings, tracked per turn and per call |
| Guardrail rollback (section 1) | Every miss raises an alert to the TPM (simulated notification) |
| Ramp pause at 575 ms TTFA (section 3) | A warning alert fires between 575 and 600 ms |
| Clinical safety (core metric: no wrong or invented clinical answers) | Drug questions are always escalated, never answered |
| Vendor fallback (section 4) | If the LLM API fails or hits a limit, a scripted fallback answers, so a call never breaks |

**About the latency numbers:** this runs on free services over the public internet with a non-streaming LLM call, so TTFA here is around 1 second and the alerts fire. That is the point of the dashboard: the targets are wired in and a miss is visible immediately. The production stack in the submission (Deepgram streaming with a tuned LLM) is what the 550 ms TTFA figure refers to.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GROQ_API_KEY=...   # free at console.groq.com, no card needed. Without it, scripted replies are used.
.venv/bin/python app.py
```

To host it free, deploy the repo on Render (the included `render.yaml` sets it up) and add `GROQ_API_KEY` as an environment variable in the Render dashboard. It is never stored in the code.

## Data

Test patients are scripted for this demo. Open datasets worth using to extend it, after checking each dataset card's license:

- [Verox132/Medical_Conversational_Dataset](https://huggingface.co/datasets/Verox132/Medical_Conversational_Dataset) for appointment-handling dialogue
- [FunDialogues/healthcare-minor-consultation](https://huggingface.co/datasets/FunDialogues/healthcare-minor-consultation) for fictitious patient questions
- [united-we-care/United-Syn-Med](https://huggingface.co/datasets/united-we-care/United-Syn-Med) for synthetic medical speech, to measure word error rate
