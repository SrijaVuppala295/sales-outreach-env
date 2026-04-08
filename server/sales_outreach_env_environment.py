# server/sales_outreach_env_environment.py
# Full Sales Outreach Sequencing Environment
# Replaces the auto-generated echo stub.
# Compatible with openenv-core 0.2.3

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from openenv.core.env_server.types import State

# Global session store for environment state (for local/testing only)
ENV_SESSIONS = {}

try:
    from ..models import SalesOutreachAction, SalesOutreachObservation
except ImportError:
    from models import SalesOutreachAction, SalesOutreachObservation

try:
    from .leads import LEADS, OBJECTIONS
    from .graders import grade_cold_email, grade_sequence_step, grade_objection_recovery
except ImportError:
    from server.leads import LEADS, OBJECTIONS
    from server.graders import grade_cold_email, grade_sequence_step, grade_objection_recovery


# ─────────────────────────────────────────────────────────────
# SIMULATED LEAD RESPONSES
# Deterministic templates triggered by score range.
# ─────────────────────────────────────────────────────────────

def _simulate_response(score: float, lead: dict) -> str:
    name = lead["name"].split()[0]
    company = lead["company"]

    if score >= 0.75:
        options = [
            f"Hi, thanks for reaching out! This actually sounds relevant to what we're working on at {company}. Can we set up a quick call?",
            f"Hey {name[:1]}... wait, you're writing to me! But yes, this caught my attention. When are you free this week?",
            f"Appreciate the personalized note. I'm open to learning more — send me a calendar link.",
        ]
    elif score >= 0.50:
        options = [
            f"Thanks for the email. I'm pretty busy right now but send me more details and I'll take a look.",
            f"Interesting — I'd need to understand better how this fits our situation at {company}.",
            f"Got your message. Not sure if the timing is right but I'll keep it in mind.",
        ]
    elif score >= 0.25:
        options = [
            f"Thanks for reaching out. We're not looking at new solutions right now.",
            f"Appreciate the note but this doesn't seem like the right fit at the moment.",
            f"Hi — I get a lot of these. Not interested currently.",
        ]
    else:
        options = [
            "[No response — lead did not reply]",
            "[Marked as spam — message was too generic]",
            "[No response — message did not stand out]",
        ]

    # Use score to deterministically pick (avoids random in graders)
    idx = int(score * len(options)) % len(options)
    return options[idx]


def _get_objection_response(objection: dict) -> str:
    return objection["response"]


# ─────────────────────────────────────────────────────────────
# TASK INSTRUCTIONS
# ─────────────────────────────────────────────────────────────

def _get_instructions(task_name: str, lead: dict, objection: Optional[dict] = None) -> str:
    n = lead["name"]
    t = lead["title"]
    c = lead["company"]

    if task_name == "cold_email":
        return (
            f"TASK 1 — Cold Email (Easy)\n"
            f"Write a single personalized cold email to {n}, {t} at {c}.\n\n"
            f"Requirements:\n"
            f"  - Use channel='email', fill subject and body\n"
            f"  - Reference their company, pain points, or recent news\n"
            f"  - End with a clear call-to-action (CTA)\n"
            f"  - Keep body 50-150 words\n"
            f"  - Do NOT use generic openers like 'I hope this email finds you well'\n\n"
            f"Scoring: personalization(40%) + CTA(25%) + non-generic(20%) + length(10%) + subject(5%)"
        )
    elif task_name == "full_sequence":
        return (
            f"TASK 2 — Full Outreach Sequence (Medium)\n"
            f"Send 3 messages in sequence to {n} at {c}:\n\n"
            f"  Step 1: Cold email       → channel='email'\n"
            f"  Step 2: LinkedIn message → channel='linkedin'\n"
            f"  Step 3: Final follow-up  → channel='followup' (include a value add: case study, result, number)\n\n"
            f"Requirements per step:\n"
            f"  - Use the EXACT channel name shown above\n"
            f"  - Do NOT repeat content from previous steps\n"
            f"  - Always include a CTA\n\n"
            f"Scoring per step: personalization(30%) + correct_channel(20%) + no_repetition(20%) + CTA(15%) + value_add(15%)"
        )
    elif task_name == "objection_handling":
        obj_hint = f"\nThe lead will raise a '{objection['type']}' objection after Step 1." if objection else ""
        return (
            f"TASK 3 — Objection Handling (Hard)\n"
            f"Step 1: Send a cold email to {n} at {c}.\n"
            f"Step 2: The lead will raise an objection. You must recover and re-engage.{obj_hint}\n\n"
            f"Recovery requirements:\n"
            f"  - Acknowledge the objection with empathy\n"
            f"  - Use a strategy specific to the objection type\n"
            f"  - Re-engage with a new CTA\n"
            f"  - DO NOT give up or say 'no problem, I'll leave you alone'\n\n"
            f"Recovery scoring: acknowledges(25%) + recovery_strategy(30%) + re_engagement(25%) + personalization(20%)"
        )
    return "Unknown task."


# ─────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────

class SalesOutreachEnvironment(Environment):
    """
    Sales Outreach Sequencing Environment.

    3 Tasks with increasing difficulty:
      Task 1 (Easy)   - cold_email:          1-step, write a cold email
      Task 2 (Medium) - full_sequence:        3-step, email → linkedin → followup
      Task 3 (Hard)   - objection_handling:   2-step, cold email + handle objection

    Rewards: 0.0–1.0 per step. Partial progress rewarded at every step.
    Graders: fully deterministic, no LLM calls.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASK_NAMES = ["cold_email", "full_sequence", "objection_handling"]
    TASK_MAX_STEPS = {
        "cold_email": 1,
        "full_sequence": 3,
        "objection_handling": 2,
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._lead: dict = {}
        self._task_name: str = ""
        self._current_step: int = 0
        self._history: list = []
        self._objection: Optional[dict] = None
        self._total_reward: float = 0.0
        self._reset_count: int = 0

    # ─────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> SalesOutreachObservation:
        """
        Start a new episode.
        Picks a random lead and random task.
        Returns initial observation with full lead context and task instructions.
        """
        if seed is not None:
            random.seed(seed)

        self._lead = random.choice(LEADS)
        self._task_name = task if task in self.TASK_NAMES else random.choice(self.TASK_NAMES)
        self._current_step = 0
        self._history = []
        self._total_reward = 0.0
        self._reset_count += 1

        # Pre-select objection for Task 3
        self._objection = random.choice(OBJECTIONS) if self._task_name == "objection_handling" else None

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        # Store state in global session dict
        ENV_SESSIONS[self._state.episode_id] = {
            "task_name": self._task_name,
            "lead": self._lead,
            "current_step": self._current_step,
            "history": self._history,
            "objection": self._objection,
            "total_reward": self._total_reward,
            "reset_count": self._reset_count,
        }

        max_steps = self.TASK_MAX_STEPS[self._task_name]
        instructions = _get_instructions(self._task_name, self._lead, self._objection)

        return SalesOutreachObservation(
            done=False,
            reward=0.0,
            echoed_message="Sales Outreach Env environment ready!",
            message_length=0,
            lead_profile=self._lead,
            lead_response="[Episode started — write your first outreach message]",
            current_step=0,
            max_steps=max_steps,
            task_name=self._task_name,
            feedback=instructions,
            score_breakdown={},
        )

    # ─────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────
    def step(
        self,
        action: SalesOutreachAction,
        timeout_s: Optional[float] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SalesOutreachObservation:  # type: ignore[override]
        """
        Process agent's outreach message.
        Score it deterministically, simulate lead response, check if done.
        """
        request_episode_id = episode_id or getattr(self._state, "episode_id", None)
        session = ENV_SESSIONS.get(request_episode_id)
        if not session:
            raise Exception("Session not found. Please reset first and provide episode_id in your step request.")

        # Restore state from session
        self._task_name = session["task_name"]
        self._lead = session["lead"]
        self._current_step = session["current_step"]
        self._history = session["history"]
        self._objection = session["objection"]
        self._total_reward = session["total_reward"]
        self._reset_count = session["reset_count"]
        self._state = State(episode_id=request_episode_id, step_count=self._current_step)

        self._current_step += 1
        self._state.step_count += 1

        # Use body field; fallback to message for backward compat
        body = action.body if hasattr(action, 'body') and action.body else getattr(action, 'message', None)
        subject = getattr(action, 'subject', None)
        channel = getattr(action, 'channel', None)
        max_steps = self.TASK_MAX_STEPS[self._task_name]

        # ── Grade based on task and step ──
        score = 0.0
        breakdown: dict = {}
        lead_response = ""
        done = False

        if self._task_name == "cold_email":
            score, breakdown = grade_cold_email(
                subject=subject, body=body, lead=self._lead
            )
            lead_response = _simulate_response(score, self._lead)
            done = True

        elif self._task_name == "full_sequence":
            score, breakdown = grade_sequence_step(
                step_num=self._current_step,
                channel=channel,
                subject=subject,
                body=body,
                lead=self._lead,
                history=self._history,
            )
            lead_response = _simulate_response(score, self._lead)
            done = self._current_step >= max_steps

        elif self._task_name == "objection_handling":
            if self._current_step == 1:
                # Step 1: cold outreach — grade as cold email
                score, breakdown = grade_cold_email(
                    subject=subject, body=body, lead=self._lead
                )
                # After step 1, lead raises the pre-selected objection
                lead_response = _get_objection_response(self._objection)
                done = False
            else:
                # Step 2: agent must handle the objection
                score, breakdown = grade_objection_recovery(
                    body=body, lead=self._lead, objection=self._objection
                )
                lead_response = _simulate_response(score, self._lead)
                done = True

        else:
            score = 0.0
            breakdown = {}
            lead_response = "[Unknown task — episode ended]"
            done = True

        # Accumulate total reward
        self._total_reward += score

        # Save step to history
        self._history.append({"step": self._current_step, "channel": channel, "body": body, "score": score})

        # Update session state
        ENV_SESSIONS[request_episode_id] = {
            "task_name": self._task_name,
            "lead": self._lead,
            "current_step": self._current_step,
            "history": self._history,
            "objection": self._objection,
            "total_reward": self._total_reward,
            "reset_count": self._reset_count,
        }

        # Build feedback string
        feedback = self._build_feedback(score, breakdown)

        return SalesOutreachObservation(
            done=done,
            reward=round(score, 3),
            echoed_message=body[:100] if body else "",
            message_length=len(body) if body else 0,
            lead_profile=self._lead,
            lead_response=lead_response,
            current_step=self._current_step,
            max_steps=max_steps,
            task_name=self._task_name,
            feedback=feedback,
            score_breakdown=breakdown,
        )

    # ─────────────────────────────────────────────
    # state property
    # ─────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    # ─────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────
    def _build_feedback(self, score: float, breakdown: dict) -> str:
        lines = [f"Step score: {score:.3f}"]
        for k, v in breakdown.items():
            lines.append(f"  {k}: {v:.3f}")
        if score >= 0.75:
            lines.append("✓ Strong message — lead engaged positively")
        elif score >= 0.50:
            lines.append("~ Decent message — lead is lukewarm")
        elif score >= 0.25:
            lines.append("✗ Weak — too generic or missing key elements")
        else:
            lines.append("✗✗ Poor — no response or spam-flagged")
        return "\n".join(lines)
