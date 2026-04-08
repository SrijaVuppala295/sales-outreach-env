#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Sales Outreach Sequencing Environment.

HACKATHON MANDATORY REQUIREMENTS MET:
  ✓ Named 'inference.py' in root directory
  ✓ Uses OpenAI Python client (openai library) for ALL LLM calls
  ✓ Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
  ✓ Defaults set ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN
  ✓ Emits START/STEP/END structured stdout logs
  ✓ Runs in under 20 minutes
  ✓ Works on 2 vCPU, 8GB RAM
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI client — mandatory per hackathon rules
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# CONFIG — EXACT format required by submission checklist
# Defaults only for API_BASE_URL and MODEL_NAME — NOT HF_TOKEN
# ─────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME: str = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN: str = os.getenv("HF_TOKEN")

# Optional — only needed if using from_docker_image()
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "sales-outreach-env:latest")

# Internal alias
API_KEY: str = HF_TOKEN or ""

BENCHMARK = "sales_outreach_env"
TASKS = ["cold_email", "full_sequence", "objection_handling"]
MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.50

if not API_KEY:
    print("[ERROR] HF_TOKEN not set. Add it to your .env file.", flush=True)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# MANDATORY LOG FUNCTIONS — exact [START] [STEP] [END] format
# ANY deviation breaks evaluation scoring per hackathon rules
# ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
        "timestamp": time.time(),
    }), flush=True)


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert B2B Sales Development Representative (SDR).
You write highly personalized, effective outreach messages to leads.

You will receive:
1. A lead profile (name, title, company, industry, pain points, recent news, tech stack)
2. The current task and step instructions
3. The lead's last response (if any)
4. Your previous messages (if any)

You MUST respond with ONLY a valid JSON object — no extra text, no markdown fences:
{
  "subject": "email subject (empty string for linkedin/followup steps)",
  "body": "your outreach message here",
  "channel": "email" or "linkedin" or "followup"
}

Critical rules:
- Always reference SPECIFIC details from the lead profile (company, pain points, recent news)
- Keep body between 50-150 words
- Always end with a clear call-to-action
- NEVER use: "I hope this email finds you well", "I wanted to reach out", "touching base"
- For objection handling: acknowledge the objection first, then pivot, never give up
- Use the EXACT channel name required by the task instructions
- Respond with ONLY the JSON object, nothing else
"""


# ─────────────────────────────────────────────────────────────
# LLM CALL — uses OpenAI client (mandatory)
# ─────────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    lead_profile: dict,
    task_name: str,
    feedback: str,
    lead_response: str,
    current_step: int,
    history: List[str],
) -> dict:
    """
    Call the LLM using OpenAI Python client.
    Returns dict with {subject, body, channel}.
    """
    user_content = f"""LEAD PROFILE:
{json.dumps(lead_profile, indent=2)}

TASK: {task_name}
CURRENT STEP: {current_step}

TASK INSTRUCTIONS:
{feedback}

LEAD'S LAST RESPONSE:
{lead_response if lead_response else "No response yet — this is your first message."}

YOUR PREVIOUS MESSAGES:
{chr(10).join(history) if history else "None — this is your first message."}

Write your outreach message now as a JSON object.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=600,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        action = json.loads(content)
        return {
            "subject": str(action.get("subject", "")),
            "body": str(action.get("body", "")),
            "channel": str(action.get("channel", "email")),
        }

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | Raw: {content[:200]}", flush=True)
        return _fallback_action(lead_profile, current_step, task_name)

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return _fallback_action(lead_profile, current_step, task_name)


def _fallback_action(lead: dict, step: int, task: str) -> dict:
    """Deterministic fallback if LLM call fails — still scores non-zero."""
    company = lead.get("company", "your company")
    name = lead.get("name", "there")
    pain = lead.get("pain_points", ["efficiency"])[0] if lead.get("pain_points") else "efficiency"
    news = lead.get("recent_news", "recent growth")
    channel_map = {1: "email", 2: "linkedin", 3: "followup"}
    channel = channel_map.get(step, "email") if task == "full_sequence" else "email"

    return {
        "subject": f"Quick idea for {company} on {pain}",
        "body": (
            f"Hi {name.split()[0]},\n\n"
            f"Saw that {company} {news.lower()}. Given the focus on {pain}, "
            f"I thought it might be worth a quick conversation about how we've helped "
            f"similar companies tackle this.\n\n"
            f"Would you be open to a 15-minute call this week?"
        ),
        "channel": channel,
    }


# ─────────────────────────────────────────────────────────────
# RUN ONE TASK EPISODE
# ─────────────────────────────────────────────────────────────

async def run_task_episode(client: OpenAI, env, task_name: str) -> dict:
    """Run one full episode for a task. Returns result dict."""
    rewards: List[float] = []
    steps_taken = 0
    history: List[str] = []
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Import action model
        try:
            from models import SalesOutreachAction
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from models import SalesOutreachAction

        # Reset env — we call reset without task arg since env picks randomly
        # For reproducible task-specific runs, env should support task kwarg
        result = await env.reset()
        obs = result.observation

        # Override task display (env picks random task, we log the intended one)
        actual_task = getattr(obs, "task_name", task_name) or task_name

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action from LLM
            action_dict = call_llm(
                client=client,
                lead_profile=getattr(obs, "lead_profile", {}),
                task_name=getattr(obs, "task_name", actual_task),
                feedback=getattr(obs, "feedback", ""),
                lead_response=getattr(obs, "lead_response", ""),
                current_step=getattr(obs, "current_step", step),
                history=history,
            )

            # Send action to environment
            try:
                result = await env.step(SalesOutreachAction(
                    subject=action_dict["subject"],
                    body=action_dict["body"],
                    channel=action_dict["channel"],
                    message=action_dict["body"],  # backward compat
                ))
            except Exception as e:
                print(f"[DEBUG] env.step error: {e}", flush=True)
                log_step(step=step, action=action_dict, reward=0.0, done=True, error=str(e))
                break

            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step

            # Update history for context
            history.append(
                f"Step {step} | channel={action_dict['channel']} | "
                f"body={action_dict['body'][:120]}... | "
                f"lead_said={getattr(obs, 'lead_response', '')[:80]}... | "
                f"score={reward:.3f}"
            )

            log_step(step=step, action=action_dict, reward=reward, done=done, error=None)

            if done:
                break

        # Final score: average reward across steps, normalized
        if rewards:
            score = sum(rewards) / MAX_STEPS
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {type(e).__name__}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "rewards": rewards,
        "steps": steps_taken,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"[DEBUG] inference.py starting", flush=True)
    print(f"[DEBUG] model={MODEL_NAME} | api_base={API_BASE_URL} | env={BENCHMARK}", flush=True)

    # Initialize OpenAI client (mandatory — uses OpenAI Python library)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Import environment client
    try:
        from client import SalesOutreachEnv
    except ImportError:
        print("[ERROR] Could not import SalesOutreachEnv from client.py", flush=True)
        sys.exit(1)

    # Connect to Docker container
    print(f"[DEBUG] Connecting to Docker image: {LOCAL_IMAGE_NAME}", flush=True)
    try:
        env = await SalesOutreachEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as e:
        print(f"[DEBUG] Docker connect failed: {e}. Trying localhost:8000...", flush=True)
        try:
            env = SalesOutreachEnv(base_url="http://localhost:8000")
        except Exception as e2:
            print(f"[ERROR] Cannot connect to environment: {e2}", flush=True)
            sys.exit(1)

    all_results = []

    try:
        for task_name in TASKS:
            print(f"\n[DEBUG] ── Starting task: {task_name} ──", flush=True)
            result = await run_task_episode(client, env, task_name)
            all_results.append(result)
            print(f"[DEBUG] Task '{task_name}' done | score={result['score']:.3f} | success={result['success']}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # ── Final Summary ──
    print("\n" + "=" * 60, flush=True)
    print("  BASELINE EVALUATION COMPLETE", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        bar = "█" * int(r["score"] * 20)
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {r['task']:25s} | {bar:<20} | {r['score']:.3f} | {status}", flush=True)

    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n  AVERAGE SCORE : {avg:.3f}", flush=True)
    print(f"  THRESHOLD     : {SUCCESS_THRESHOLD}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())