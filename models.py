# models.py
# Typed Pydantic models for the Sales Outreach Sequencing Environment
# Compatible with openenv-core 0.2.3
# Action and Observation inherit from openenv base classes

from typing import Any, Dict
from openenv.core.env_server.interfaces import Action, Observation


# ─────────────────────────────────────────────
# ACTION — what the LLM agent sends each step
# ─────────────────────────────────────────────
class SalesOutreachAction(Action):
    """
    The agent writes an outreach message to a lead.

    Fields:
        subject  - email subject line (empty string for linkedin/followup)
        body     - the actual message content (required)
        channel  - 'email' | 'linkedin' | 'followup'
    
    NOTE: 'message' field kept for backward compat with auto-generated app.py
    """
    message: str = ""     # kept for schema compat
    subject: str = ""
    body: str = ""
    channel: str = "email"


# ─────────────────────────────────────────────
# OBSERVATION — what the agent sees after each step
# done and reward are inherited from Observation base class
# ─────────────────────────────────────────────
class SalesOutreachObservation(Observation):
    """
    After each agent action, the environment returns this.

    Fields:
        lead_profile    - full lead context dict
        lead_response   - simulated response from the lead
        current_step    - which step in the sequence (0, 1, 2, 3)
        max_steps       - total steps allowed in this task
        task_name       - 'cold_email' | 'full_sequence' | 'objection_handling'
        feedback        - environment feedback on agent's last message
        score_breakdown - per-criterion partial scores

    NOTE: echoed_message and message_length kept for backward compat
          with the auto-generated app.py schema check
    """
    # Backward compat with echo schema (auto-generated app.py uses these)
    echoed_message: str = ""
    message_length: int = 0

    # Core outreach fields
    lead_profile: Dict[str, Any] = {}
    lead_response: str = ""
    current_step: int = 0
    max_steps: int = 1
    task_name: str = ""
    feedback: str = ""
    score_breakdown: Dict[str, float] = {}