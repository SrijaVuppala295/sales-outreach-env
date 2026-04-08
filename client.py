# client.py
# WebSocket client for the Sales Outreach Sequencing Environment.
# Compatible with openenv-core 0.2.3
# This is what agent code imports to talk to the environment server.

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from models import SalesOutreachAction, SalesOutreachObservation
except ImportError:
    from .models import SalesOutreachAction, SalesOutreachObservation


class SalesOutreachEnv(EnvClient[SalesOutreachAction, SalesOutreachObservation, dict]):
    """
    Client for the Sales Outreach Sequencing Environment.

    Usage (async):
        async with SalesOutreachEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(SalesOutreachAction(body="Hi...", channel="email"))

    Usage (sync, for scripts):
        with SalesOutreachEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(SalesOutreachAction(body="Hi..."))

    From Docker (inference.py):
        env = await SalesOutreachEnv.from_docker_image("sales-outreach-env:latest")
    """

    def _step_payload(self, action: SalesOutreachAction) -> dict:
        """Convert typed action → JSON dict sent over WebSocket."""
        return {
            "message": action.body or action.message,  # backward compat
            "subject": action.subject,
            "body": action.body,
            "channel": action.channel,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Convert JSON WebSocket response → typed StepResult."""
        obs_data = payload.get("observation", payload)

        observation = SalesOutreachObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            # Backward compat fields
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            # Core outreach fields
            lead_profile=obs_data.get("lead_profile", {}),
            lead_response=obs_data.get("lead_response", ""),
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 1),
            task_name=obs_data.get("task_name", ""),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> dict:
        """Parse state response — returns raw dict for compatibility."""
        return payload
