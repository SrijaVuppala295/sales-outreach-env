# Sales Outreach Sequencing Environment

An OpenEnv-compatible RL environment for training AI agents to write effective B2B sales outreach.

## Environment Description

Simulates the real-world workflow of a B2B Sales Development Representative (SDR). The AI agent reads a lead profile and writes personalized outreach messages. The environment scores messages deterministically and simulates realistic lead responses.

**Why this matters:** Every B2B company trains SDRs. Teaching agents to write non-generic, context-aware outreach has immediate enterprise value.

---

## Tasks

### Task 1 — Cold Email (Easy, 1 step)
Write one personalized cold email. Scored on personalization, CTA, non-generic language, length, subject quality.

### Task 2 — Full Sequence (Medium, 3 steps)
Send 3 messages in sequence: `email → linkedin → followup`. Scored per step on channel correctness, no repetition, personalization, CTA, value add.

### Task 3 — Objection Handling (Hard, 2 steps)
Step 1: Cold email. Step 2: Handle a realistic objection (timing/budget/competitor/relevance). Scored on empathy, recovery strategy, re-engagement.

---

## Action Space

```python
SalesOutreachAction(
    subject: str   # Email subject (empty for linkedin/followup)
    body: str      # Message content
    channel: str   # "email" | "linkedin" | "followup"
)
```

## Observation Space

```python
SalesOutreachObservation(
    done: bool
    reward: float              # Step reward 0.0–1.0
    lead_profile: dict         # Full lead context
    lead_response: str         # Simulated lead response
    current_step: int
    max_steps: int
    task_name: str
    feedback: str              # Scoring feedback
    score_breakdown: dict      # Per-criterion scores
)
```

## Reward Function

Partial progress rewarded at every step (0.0–1.0). Deterministic graders — no LLM calls in scoring.

| Criterion | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Personalization | 40% | 30% | 20% |
| CTA | 25% | 15% | 25% |
| Non-generic | 20% | — | — |
| Length | 10% | — | — |
| Subject quality | 5% | — | — |
| Correct channel | — | 20% | — |
| No repetition | — | 20% | — |
| Value add | — | 15% | — |
| Acknowledges objection | — | — | 25% |
| Recovery strategy | — | — | 30% |

---

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Quick HTTP Test

Use the same top-level `episode_id` in both `/reset` and `/step`.

```powershell
$episodeId = "episode-001"

$resetBody = @{
    episode_id = $episodeId
    seed = 42
    task = "objection_handling"
} | ConvertTo-Json

$reset = Invoke-RestMethod -Uri "http://localhost:8000/reset" `
    -Method Post `
    -ContentType "application/json" `
    -Body $resetBody

$reset | ConvertTo-Json -Depth 6

$action = @{
    subject = "Priya, faster releases at FinStack"
    body = "Hi Priya, I saw FinStack's Series B and your post about microservices migration. We help engineering teams reduce deployment friction and improve visibility into technical debt during scale-ups. Would you be open to a 15-minute conversation next week to compare approaches?"
    channel = "email"
}

$stepBody = @{
    episode_id = $episodeId
    action = $action
} | ConvertTo-Json -Depth 6

$step = Invoke-RestMethod -Uri "http://localhost:8000/step" `
    -Method Post `
    -ContentType "application/json" `
    -Body $stepBody

$step | ConvertTo-Json -Depth 6
```

Notes:
- `episode_id` belongs at the top level of the `/step` request, not inside `action`.
- The message should match the lead returned by `/reset`; after `seed = 42` and `task = "objection_handling"`, the lead is `Priya Sharma` at `FinStack Technologies`.

## Docker

```bash
docker build -t sales-outreach-env .
docker run -p 8000:8000 sales-outreach-env
```

## Run Baseline

```bash
# Set in .env:
# API_BASE_URL=https://api.groq.com/openai/v1
# MODEL_NAME=llama-3.3-70b-versatile
# HF_TOKEN=your_groq_api_key

python inference.py
```

## Validate

```bash
openenv validate
```

## Baseline Scores (llama-3.3-70b-versatile via Groq)

| Task | Score |
|------|-------|
| cold_email | ~0.70 |
| full_sequence | ~0.58 |
| objection_handling | ~0.55 |
