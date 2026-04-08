from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import uvicorn

from environment import HiringEnv
from models import (
    AskInterviewQuestionAction,
    FlagBiasAction,
    Observation,
    RankAction,
    RejectAction,
    ScheduleAction,
    ScreenAction,
    SendMessageAction,
    ShortlistAction,
    SubmitAction,
)


ENV_NAME = "hiring-coordinator-env"

ACTION_MAP = {
    "screen": ScreenAction,
    "shortlist": ShortlistAction,
    "reject": RejectAction,
    "rank": RankAction,
    "schedule": ScheduleAction,
    "send_message": SendMessageAction,
    "flag_bias": FlagBiasAction,
    "ask_question": AskInterviewQuestionAction,
    "submit": SubmitAction,
}


class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=4)
    seed: int = 42
    domain: str = "engineering"


app = FastAPI(
    title="Hiring Coordinator OpenEnv",
    description="Policy-aware hiring workflow simulation for agent evaluation.",
    version="1.0.0",
)


def parse_action(data: dict[str, Any]):
    action_type = data.get("action_type")
    cls = ACTION_MAP.get(action_type)
    if not cls:
        raise ValueError(f"Unknown action_type: {action_type}")
    return cls.model_validate(data)


def get_env() -> HiringEnv:
    env = getattr(app.state, "env", None)
    if env is None:
        env = HiringEnv(task_id=1, seed=42)
        app.state.env = env
    return env


def public_observation(observation: Observation) -> dict[str, Any]:
    data = observation.model_dump(mode="json")
    data.get("jd", {}).pop("biased_clauses", None)
    return data


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    data = dict(state)
    if "jd" in data:
        data["jd"] = dict(data["jd"])
        data["jd"].pop("biased_clauses", None)
    data["candidates"] = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"true_fit_score", "true_label", "os_proficiency", "dbms_proficiency"}
        }
        for candidate in data.get("candidates", [])
    ]
    data["hidden_fields_omitted"] = True
    return data


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": ENV_NAME}


@app.post("/reset")
def reset(payload: ResetRequest | None = Body(default=None)) -> dict[str, Any]:
    config = payload or ResetRequest()
    env = HiringEnv(task_id=config.task_id, seed=config.seed, domain=config.domain)
    app.state.env = env
    return public_observation(env._build_observation())


@app.post("/step")
def step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    raw_action = payload.get("action", payload)
    if not isinstance(raw_action, dict):
        raise HTTPException(status_code=400, detail="Action payload must be a JSON object")

    try:
        action = parse_action(raw_action)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    env = get_env()
    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": public_observation(observation),
        "reward": reward.step_reward,
        "reward_details": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = get_env()
    return public_state(env.state())


def main() -> None:
    uvicorn.run("server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
