# server/app.py
# FastAPI server — uses auto-generated create_app pattern from openenv-core 0.2.3
# DO NOT change the create_app import — this is the correct API for your version.

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e


from models import SalesOutreachAction, SalesOutreachObservation
from server.sales_outreach_env_environment import SalesOutreachEnvironment


# Create the FastAPI app with all endpoints auto-generated:
# POST /reset, POST /step, GET /state, WS /ws, GET /health, GET /docs
app = create_app(
    SalesOutreachEnvironment,
    SalesOutreachAction,
    SalesOutreachObservation,
    env_name="sales_outreach_env",
    max_concurrent_envs=10,
)


def main():
    """Run the FastAPI app with command-line-configurable port."""
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
