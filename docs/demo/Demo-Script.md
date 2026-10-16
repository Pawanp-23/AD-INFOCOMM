# RAMEI demo — runbook and video script

Two uses: the **live demo** in front of the guide (Part A + Part C), and the
**recorded demo video**, about 6 minutes (Part B). Rehearse once end-to-end before recording.

## Part A — Preparation (do this the day before)

| # | Step | Check |
|---|---|---|
| 1 | Model bundle exists: `ml/artifacts/bundles/ramei-rgcmt-1.0.0/manifest.json` | System page shows "bundle found" |
| 2 | Users exist: `python -m app.cli create-user --email admin@ramei.local --role admin` (and an `agent`) | Can log in |
| 3 | Record **your own** 20-40 s clip: face clearly lit, speaking with a clear emotion (e.g. an annoyed complaint, then calming down) | Plays in a browser |
| 4 | `python scripts/make_demo_clips.py my_clip.mp4 --out demo_clips/` | 4 videos + 1 wav |
| 5 | Pre-run every clip once so models are warm and the reports exist | Sessions list has 5 completed sessions |
| 6 | Start: `uvicorn app.main:app --port 8000` (backend) and `npm run dev` (frontend) | http://localhost:5173 loads |
| 7 | Close other GPU apps; plug in the laptop; silence notifications | — |
| 8 | Browser: allow camera + microphone for localhost; zoom 110-125% for recording | — |

Use only recordings of yourself (or people who agreed). Do not use dataset clips in a public video
unless their license allows it.

## Part B — Video script (~6 min)

| Time | Screen | Say (summarised — speak naturally) |
|---|---|---|
| 0:00-0:30 | Title slide: RAMEI name + one-line tagline | "Emotion recognition from voice, words and face — that knows when not to trust a signal." Who you are, internship at AD-INFOCOMM. |
| 0:30-1:15 | Slide: baseline gaps (G1-G7) + leakage table (Final Report §5.1) | The baseline was audio-only and claimed ~70%. On speakers it has never heard, it gets ~51-55% — the 70% came from the same actors in train and test. That honest number is what we improve on. |
| 1:15-1:45 | Slide: architecture diagram (Final Report §4.1) | Three branches → a reliability score per modality → fusion that shifts trust → calibrated confidence, "uncertain" when unsure, and an explanation. |
| 1:45-2:45 | **Upload** page → upload `*_clean.mp4` → consent → processing → **Session report** | Timeline, dominant emotion, transcript. Click a segment: per-modality predictions, reliability bars, fusion weights, explanation lines. |
| 2:45-3:45 | Open the reports for `*_noisy.mp4`, then `*_dark.mp4` (pre-computed) side by side with clean | Same person, same words. With noise, voice reliability drops and the weight moves to the face; in the dark video the face is down-weighted ("Face is poorly lit"). This is the core idea, visible. |
| 3:45-4:15 | Report for `*_no_video.wav` | Camera off — the model still works on voice and words; the face is shown as unavailable rather than guessed. |
| 4:15-5:00 | **Live** page: start with camera + mic, speak for ~20 s, then cover the camera lens with your hand | Live segments every ~3 s. With the lens covered, the face weight falls in real time. Show an "uncertain" segment if one appears and explain abstention. |
| 5:00-5:30 | Report: alerts list, summary card, Export → CSV | Alerts (e.g. sustained anger), session summary, export for analysts. |
| 5:30-6:00 | Slide: results table (baseline → audio → fusion) + limitations | Headline numbers, the honest limitations (acted data, small speaker pools), what comes next. Thank the guide. |

Recording tips: 1080p, record the browser window only, a quiet room, a separate microphone if you
have one. Record the screen first, then narrate over it if live narration is hard. Trim dead time
while models process (or show it once and cut the rest).

## Part C — Questions to prepare for

| Likely question | Short answer |
|---|---|
| Why is your accuracy lower than the original 70%? | We measure on unseen speakers. The 70% includes speaker leakage (+15-21 points, M0 report). |
| How do you know reliability helps? | Ablation E9 removes the reliability bias; compare it with E8 per scenario (M2 report). |
| What does "uncertain" mean? | Calibrated top probability below the threshold — the system abstains instead of guessing. |
| Is the confidence trustworthy? | Temperature scaling fitted on validation speakers; ECE is reported per fold. |
| Does it work on real calls? | Trained/evaluated on acted data; real conversational data (IEMOCAP/MSP-Podcast) is the next step. |
| Privacy? | Consent per session; raw media deleted after analysis by default; the optional AI summary sees only aggregate numbers. |
| Can it be misused? | It must not be used for decisions about individuals or covert monitoring (Final Report §7). |
| Why not fine-tune everything end to end? | 4 GB GPU; frozen WavLM matched LoRA on 24 speakers (M1), so frozen features were the better trade-off. |
