# server/environment.py
# Core environment logic for the Sales Outreach Sequencing Environment.
# Implements reset(), step(), and state property as required by OpenEnv spec.

import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment

# Use relative imports since this runs inside server/ package
from .leads import LEADS, OBJECTIONS
from .graders import (
    grade_cold_email,
    grade_sequence_step,
    grade_objection_recovery,
)

# We need to import models from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import OutreachAction, OutreachObservation, OutreachState


# ─────────────────────────────────────────────────────────────
# SIMULATED LEAD RESPONSES
# These are deterministic response templates triggered by score ranges.
# High score = positive response. Low score = ghosted / negative.
# ─────────────────────────────────────────────────────────────

def _simulate_lead_response(score: float, task: str, step: int, lead: dict) -> str:
    """Simulate the lead's response based on how good the agent's message was."""
    name = lead["name"].split()[0]

    if score >= 0.75:
        responses = [
            f"Hi, thanks for reaching out! This actually sounds relevant. Can we set up a quick call?",
            f"Hey, appreciate the personalized note. I'm open to learning more — send me a calendar link.",
            f"Thanks {name[:1]}... wait, you're writing to me, not the other way! But yes, this caught my attention. When are you free?",
        ]
    elif score >= 0.50:
        responses = [
            f"Thanks for the email. I'm a bit busy right now but send me more details and I'll take a look.",
            f"Interesting, though I'd need to understand better how this fits our situation at {lead['company']}.",
            f"Got your message. Not sure if the timing is right but I'll keep it in mind.",
        ]
    elif score >= 0.25:
        responses = [
            f"Thanks for reaching out. We're not looking at new solutions right now.",
            f"Appreciate the note but this doesn't seem like a fit at the moment.",
            f"Hi — I get a lot of these. Not interested currently.",
        ]
    else:
        responses = [
            f"[No response — lead did not reply]",
            f"[Marked as spam — message was too generic]",
            f"[No response]",
        ]

    return random.choice(responses)


def _get_objection_response(objection: dict) -> str:
    return objection["response"]


# ─────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────

class SalesOutreachEnvironment(Environment):
    """
    Sales Outreach Sequencing Environment.

    3 Tasks:
      Task 1 (Easy)   - cold_email:          Write a single cold email
      Task 2 (Medium) - full_sequence:        Write a 3-step outreach sequence
      Task 3 (Hard)   - objection_handling:   Handle a realistic lead objection

    Reward signal: 0.0-1.0 at each step (partial progress rewards).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    TASK_NAMES = ["cold_email", "full_sequence", "objection_handling"]
    TASK_MAX_STEPS = {
        "cold_email": 1,
        "full_sequence": 3,
        "objection_handling": 3,
    }

    def __init__(self):
        self._state = OutreachState()
        self._lead = None
        self._task_name = ""
        self._current_step = 0
        self._history = []        # list of {channel, subject, body, score}
        self._objection = None    # only used in Task 3
        self._total_reward = 0.0

    # ─────────────────────────────────────────────
    # reset() — called at start of every episode
    # ─────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> OutreachObservation:
        """
        Start a new episode.
        - Picks a random lead profile
        - Picks a task (or uses the one provided)
        - Returns the initial observation with full lead context
        """
        if seed is not None:
            random.seed(seed)

        # Pick lead and task
        self._lead = random.choice(LEADS)
        self._task_name = task if task in self.TASK_NAMES else random.choice(self.TASK_NAMES)
        self._current_step = 0
        self._history = []
        self._total_reward = 0.0
        self._objection = None

        # For Task 3, pre-select the objection the lead will raise at step 2
        if self._task_name == "objection_handling":
            self._objection = random.choice(OBJECTIONS)

        ep_id = episode_id or str(uuid.uuid4())
        self._state = OutreachState(
            episode_id=ep_id,
            step_count=0,
            task_name=self._task_name,
            total_reward=0.0,
            lead_id=self._lead["id"],
        )

        max_steps = self.TASK_MAX_STEPS[self._task_name]

        return OutreachObservation(
            done=False,
            reward=None,
            lead_profile=self._lead,
            lead_response="[Episode started — write your first outreach message]",
            current_step=0,
            max_steps=max_steps,
            task_name=self._task_name,
            feedback=self._get_task_instructions(),
            score_breakdown={},
        )

    # ─────────────────────────────────────────────
    # step() — called every time agent sends a message
    # ─────────────────────────────────────────────
    def step(
        self,
        action: OutreachAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> OutreachObservation:
        """
        Process the agent's outreach message.
        Score it, simulate lead response, check if episode is done.
        """
        self._current_step += 1
        self._state.step_count += 1

        max_steps = self.TASK_MAX_STEPS[self._task_name]

        # ── Grade the action based on current task and step ──
        if self._task_name == "cold_email":
            score, breakdown = grade_cold_email(
                subject=action.subject,
                body=action.body,
                lead=self._lead,
            )
            lead_response = _simulate_lead_response(score, self._task_name, self._current_step, self._lead)
            done = True  # Task 1 is 1 step

        elif self._task_name == "full_sequence":
            score, breakdown = grade_sequence_step(
                step_num=self._current_step,
                channel=action.channel,
                subject=action.subject,
                body=action.body,
                lead=self._lead,
                history=self._history,
            )
            lead_response = _simulate_lead_response(score, self._task_name, self._current_step, self._lead)
            done = self._current_step >= max_steps

        elif self._task_name == "objection_handling":
            if self._current_step == 1:
                # Step 1: initial cold outreach — grade as cold email
                score, breakdown = grade_cold_email(
                    subject=action.subject,
                    body=action.body,
                    lead=self._lead,
                )
                # After step 1, lead raises the objection
                lead_response = _get_objection_response(self._objection)
                done = False

            elif self._current_step == 2:
                # Step 2: agent responds to the objection
                score, breakdown = grade_objection_recovery(
                    body=action.body,
                    lead=self._lead,
                    objection=self._objection,
                )
                lead_response = _simulate_lead_response(score, self._task_name, self._current_step, self._lead)
                done = True  # Task 3 ends after objection handling

            else:
                # Extra steps beyond task scope — penalize
                score = 0.0
                breakdown = {"out_of_bounds": 0.0}
                lead_response = "[Task complete — no more steps needed]"
                done = True

        else:
            score = 0.0
            breakdown = {}
            lead_response = "[Unknown task]"
            done = True

        # Accumulate reward
        self._total_reward += score
        self._state.total_reward = self._total_reward

        # Store step in history (for sequence repetition checks)
        self._history.append({
            "step": self._current_step,
            "channel": action.channel,
            "subject": action.subject,
            "body": action.body,
            "score": score,
        })

        # Build feedback string
        feedback = self._build_feedback(score, breakdown)

        return OutreachObservation(
            done=done,
            reward=round(score, 3),
            lead_profile=self._lead,
            lead_response=lead_response,
            current_step=self._current_step,
            max_steps=max_steps,
            task_name=self._task_name,
            feedback=feedback,
            score_breakdown=breakdown,
        )

    # ─────────────────────────────────────────────
    # state property — returns episode metadata
    # ─────────────────────────────────────────────
    @property
    def state(self) -> OutreachState:
        return self._state

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    def _get_task_instructions(self) -> str:
        lead = self._lead
        if self._task_name == "cold_email":
            return (
                f"TASK 1 (Easy) — Cold Email\n"
                f"Write a single cold email to {lead['name']}, {lead['title']} at {lead['company']}.\n"
                f"Your message must include: a personalized subject, reference to their context, and a clear CTA.\n"
                f"Use channel='email'. Fill subject and body fields.\n"
                f"Scoring: personalization (40%) + CTA (25%) + non-generic (20%) + length (15%)"
            )
        elif self._task_name == "full_sequence":
            return (
                f"TASK 2 (Medium) — Full Outreach Sequence\n"
                f"Send 3 messages in sequence to {lead['name']} at {lead['company']}:\n"
                f"  Step 1: Cold email (channel='email')\n"
                f"  Step 2: LinkedIn follow-up (channel='linkedin')\n"
                f"  Step 3: Final follow-up with value add (channel='followup')\n"
                f"Each step must use the correct channel and not repeat previous messages.\n"
                f"Scoring per step: personalization (30%) + correct channel (20%) + no repetition (20%) + CTA (15%) + value add (15% step 3 only)"
            )
        elif self._task_name == "objection_handling":
            return (
                f"TASK 3 (Hard) — Objection Handling\n"
                f"Step 1: Send a cold email to {lead['name']} at {lead['company']}.\n"
                f"Step 2: The lead will raise an objection. You must recover and re-engage.\n"
                f"Recovery scoring: acknowledge objection (25%) + recovery strategy (30%) + re-engagement CTA (25%) + personalization (20%)\n"
                f"Warning: giving up or going silent results in 0.0 score."
            )
        return "Unknown task."

    def _build_feedback(self, score: float, breakdown: dict) -> str:
        lines = [f"Step score: {score:.3f}"]
        for key, val in breakdown.items():
            lines.append(f"  {key}: {val:.3f}")
        if score >= 0.75:
            lines.append("✓ Strong message — lead engaged positively")
        elif score >= 0.50:
            lines.append("~ Decent message — lead is lukewarm")
        elif score >= 0.25:
            lines.append("✗ Weak message — too generic or missing key elements")
        else:
            lines.append("✗✗ Very poor message — no response or spam-flagged")
        return "\n".join(lines)
