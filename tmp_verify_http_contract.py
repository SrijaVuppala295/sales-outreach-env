from models import SalesOutreachAction
from server.sales_outreach_env_environment import SalesOutreachEnvironment


episode_id = "episode-001"

reset_env = SalesOutreachEnvironment()
reset_obs = reset_env.reset(seed=42, episode_id=episode_id, task="objection_handling")

step_env = SalesOutreachEnvironment()
step_obs = step_env.step(
    SalesOutreachAction(
        subject="Priya, faster releases at FinStack",
        body=(
            "Hi Priya, saw FinStack's Series B and your post about microservices migration. "
            "We help engineering leaders reduce deployment friction and technical debt visibility "
            "during scale-ups without adding reporting overhead. Would it be worth a 15-minute "
            "chat next week to compare how teams like yours shortened release cycles?"
        ),
        channel="email",
    ),
    episode_id=episode_id,
)

print("reset_task=", reset_obs.task_name)
print("reset_lead=", reset_obs.lead_profile.get("name"))
print("step_done=", step_obs.done)
print("step_current_step=", step_obs.current_step)
print("step_task=", step_obs.task_name)
print("step_response=", step_obs.lead_response)
