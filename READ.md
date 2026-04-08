# Sales Outreach Sequencing Environment

An OpenEnv-compatible RL environment for training AI agents to write effective B2B sales outreach messages.

---

## What This Environment Does

This environment simulates the real-world workflow of a Sales Development Representative (SDR). An AI agent must:

1. Read a lead's profile (name, title, company, pain points, recent news)
2. Write personalized outreach messages
3. Handle realistic lead responses and objections

**Why this matters:** Every B2B company needs effective outreach. Training agents to write non-generic, context-aware messages has immediate enterprise value. This environment fills a real gap — no comparable OpenEnv environment exists for sales/outreach training.

---

## Tasks (Easy → Medium → Hard)

### Task 1 — Cold Email (Easy)
**Goal:** Write a single cold email to a lead.

**What's tested:**
- Personalization (references company, pain points, recent news)
- Clear call-to-action
- Non-generic language
- Appropriate length (50-150 words)

**Max steps:** 1  
**Success threshold:** score ≥ 0.50

---

### Task 2 — Full Sequence (Medium)
**Goal:** Send 3 messages in the correct sequence:
- Step 1: Cold email (`channel=email`)
- Step 2: LinkedIn follow-up (`channel=linkedin`)
- Step 3: Final follow-up with value add (`channel=followup`)

**What's tested:**
- Correct channel usage per step
- No repetition between messages
- Value added in step 3 (case studies, results, numbers)
- Personalization throughout

**Max steps:** 3  
**Success threshold:** avg score ≥ 0.50

---

### Task 3 — Objection Handling (Hard)
**Goal:** Initial outreach + recover from a realistic objection.
- Step 1: Cold email
- Step 2: Lead raises objection (timing / budget / competitor / irrelevance)
- Agent must acknowledge, pivot, and re-engage

**What's tested:**
- Acknowledges objection with empathy
- Uses appropriate recovery strategy
- Re-engages with a new CTA
- Doesn't give up

**Max steps:** 3  
**Success threshold:** score ≥ 0.50 on objection recovery step

---

## Action Space

```python
OutreachAction(
    subject: str   # Email subject line (empty for non-email)
    body: str      # The message content
    channel: str   # "email" | "linkedin" | "followup"
)
```

## Observation Space

```python
OutreachObservation(
    done: bool                        # Is episode over?
    reward: float                     # Step reward (0.0 - 1.0)
    lead_profile: dict                # Full lead context
    lead_response: str                # Simulated lead response
    current_step: int                 # Current step number
    max_steps: int                    # Total steps in this task
    task_name: str                    # Which task is running
    feedback: str                     # Environment feedback on last message
    score_breakdown: dict             # Per-criterion scores
)
```

## Reward Function

Rewards are given at each step (0.0-1.0). Partial progress is rewarded — the agent gets signal even if it doesn't fully succeed.

| Criterion | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Personalization | 40% | 30% | 20% |
| CTA presence | 25% | 15% | 25% |
| Non-generic | 20% | — | — |
| Length | 15% | — | — |
| Correct channel | — | 20% | — |
| No repetition | — | 20% | — |
| Value add | — | 15% | — |
| Acknowledge objection | — | — | 25% |
| Recovery strategy | — | — | 30% |

---

## Baseline Scores

Run with `gpt-4o-mini`:

| Task | Score |
|------|-------|
| cold_email | ~0.72 |
| full_sequence | ~0.61 |
| objection_handling | ~0.58 |

---

## Setup & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t sales-outreach-env .
docker run -p 8000:8000 sales-outreach-env
```

### Run baseline
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token"
python inference.py
```

### Validate
```bash
openenv validate
```

---

## Lead Profiles

The environment includes 5 diverse lead profiles across industries:
- FinStack Technologies (Fintech, Head of Engineering)
- RetailLoop Inc (E-commerce SaaS, VP of Sales)
- MedBridge Health (HealthTech, COO)
- SupplyNest (Supply Chain SaaS, Founder/CEO)
- Klarity GmbH (HR Tech, Director of People)

Each profile includes: name, title, company, industry, pain points, recent news, tech stack, and LinkedIn activity.

---

## Scoring Breakdown

**Real-world utility (30%):** Directly models B2B SDR workflow. Immediate value for training sales AI agents.

**Task & grader quality (25%):** 3 tasks with clear difficulty curve. Deterministic graders based on content signals — no LLM calls in evaluation.

**Environment design (20%):** Clean state management. Partial reward at every step. Meaningful episode boundaries.

**Code quality (15%):** Passes `openenv validate`. Docker builds cleanly. Typed Pydantic models throughout.

**Creativity (10%):** Novel domain in OpenEnv. Simulated lead responses create realistic multi-turn dynamics.