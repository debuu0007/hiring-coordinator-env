## Context
I have already written the core logic for an OpenEnv-compliant hiring coordinator environment. 
The files are:
- `models.py`       — All Pydantic models (Observation, Action, Reward, domain objects)
- `data_generator.py` — Deterministic synthetic data generation
- `graders.py`      — Deterministic graders for all 3 tasks (returns float 0.0–1.0)
- `environment.py`  — Core HiringEnv class with reset(), step(), state()

DO NOT modify these 4 files. Build everything else around them.

---

## Project structure to create

```
hiring-env/
├── models.py               ← already written
├── data_generator.py       ← already written
├── graders.py              ← already written
├── environment.py          ← already written
├── openenv.yaml            ← create this
├── server.py               ← create this (FastAPI HTTP wrapper)
├── baseline.py             ← create this (OpenAI API baseline agent)
├── validate.py             ← create this (openenv validate runner)
├── tests/
│   └── test_env.py         ← create this
├── Dockerfile              ← create this
├── requirements.txt        ← create this
└── README.md               ← create this
```

---

## 1. openenv.yaml

```yaml
name: hiring-coordinator-env
version: "1.0.0"
description: >
  A hiring coordinator environment where an agent screens candidates,
  shortlists, ranks, schedules interviews, detects JD bias, and manages
  candidate communications — across 3 tasks of increasing difficulty.
author: "<your name>"
tags: [hiring, hr, recruitment, nlp, classification]
tasks:
  - id: 1
    name: Resume Screening
    difficulty: easy
    description: Screen 10 candidates for a job description. Classify each as fit/maybe/no_fit.
    max_steps: 20
  - id: 2
    name: Shortlist and Rank
    difficulty: medium
    description: From 15 candidates, shortlist top 3 and rank them correctly.
    max_steps: 35
  - id: 3
    name: Full Hiring Pipeline
    difficulty: hard
    description: >
      End-to-end: screen 20 candidates, detect biased JD clauses,
      shortlist and rank, schedule interviews, and send communications.
    max_steps: 60
observation_type: json
action_type: json
reward_range: [-1.0, 1.0]
```

---

## 2. server.py — FastAPI HTTP wrapper

Create a FastAPI app with these routes:

```
POST /reset         body: {task_id: int, seed: int}   → returns Observation JSON
POST /step          body: {action: dict}               → returns {observation, reward, done, info}
GET  /state         → returns full internal state dict
GET  /health        → returns {"status": "ok", "env": "hiring-coordinator-env"}
```

- Store the env instance in app state (single-instance server is fine for hackathon)
- Use `from environment import HiringEnv`
- Parse actions using `pydantic` model_validate with a discriminator on `action_type`
- Return proper HTTP 400 errors for invalid actions with an error message

Action parsing pattern (since Action is a Union type):
```python
from models import ScreenAction, ShortlistAction, RejectAction, RankAction, ScheduleAction, SendMessageAction, FlagBiasAction, SubmitAction

ACTION_MAP = {
    "screen": ScreenAction,
    "shortlist": ShortlistAction,
    "reject": RejectAction,
    "rank": RankAction,
    "schedule": ScheduleAction,
    "send_message": SendMessageAction,
    "flag_bias": FlagBiasAction,
    "submit": SubmitAction,
}

def parse_action(data: dict):
    action_type = data.get("action_type")
    cls = ACTION_MAP.get(action_type)
    if not cls:
        raise ValueError(f"Unknown action_type: {action_type}")
    return cls.model_validate(data)
```

---

## 3. baseline.py — OpenAI API baseline agent

Create a script that:
1. Reads `OPENAI_API_KEY` from environment variables (fail with clear error if missing)
2. Runs gpt-4o-mini (not gpt-4) against all 3 tasks sequentially
3. Uses the OpenAI client to call the model, passing the observation as a JSON string in the prompt
4. Parses the model's response as a JSON action dict, submits it via `env.step()`
5. Loops until `done=True` or max steps reached
6. Prints a final score table at the end

System prompt to use for the agent:
```
You are a professional hiring coordinator. You will receive a job description and a pool of candidates.
Your goal is to evaluate candidates and take hiring actions.

You must respond ONLY with a valid JSON object matching one of these action types:
- {"action_type": "screen", "candidate_id": "c_000", "decision": "fit|no_fit|maybe"}
- {"action_type": "shortlist", "candidate_id": "c_000"}
- {"action_type": "reject", "candidate_id": "c_000", "reason": "..."}
- {"action_type": "rank", "ordered_ids": ["c_000", "c_001", ...]}
- {"action_type": "schedule", "candidate_id": "c_000", "slot_id": "slot_000"}
- {"action_type": "send_message", "candidate_id": "c_000", "message_type": "invite|reject|schedule_confirm|waitlist"}
- {"action_type": "flag_bias", "clause": "...", "bias_type": "gender|age|nationality|disability|vague"}
- {"action_type": "submit"}

Current observation (JSON):
{observation}
```

Error handling: if model returns invalid JSON, retry once, then default to {"action_type": "submit"}.

Output format at the end:
```
==================== BASELINE RESULTS ====================
Task 1 (Easy)   — Resume Screening:     0.73
Task 2 (Medium) — Shortlist and Rank:   0.61
Task 3 (Hard)   — Full Pipeline:        0.48
---------------------------------------------------------
Average score:                          0.61
==========================================================
```

---

## 4. tests/test_env.py

Write pytest tests covering:
- `test_reset_returns_observation`: reset() returns valid Observation with correct candidate count
- `test_step_screen_correct`: screen a known strong_fit candidate as "fit" → positive reward
- `test_step_screen_wrong`: screen a known no_fit candidate as "fit" → negative reward
- `test_double_booking_penalty`: scheduling same slot twice → reward < 0
- `test_submit_ends_episode`: after SubmitAction, done=True
- `test_max_actions_auto_submit`: taking max_actions steps without submitting → episode ends
- `test_task1_grader_perfect`: agent screens all correctly → grade > 0.9
- `test_task2_shortlist_top3`: agent shortlists exactly the top 3 → score > 0.8
- `test_task3_full_pipeline`: smoke test — run task 3, submit, get a score
- `test_deterministic`: same seed → same observation every time

Use `from environment import HiringEnv` and the action models from `models.py`.

---

## 5. Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. requirements.txt

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
openai>=1.30.0
pytest>=8.0.0
httpx>=0.27.0       # for FastAPI test client
```

---

## 7. README.md

Write a clean README with:
- Project overview (2-3 sentences)
- Architecture diagram (ASCII is fine) showing: Agent → server.py → HiringEnv → graders.py
- Quick start: `docker build`, `docker run`, `curl /health`
- How to run the baseline: `OPENAI_API_KEY=... python baseline.py`
- How to run tests: `pytest tests/`
- Task descriptions table (task id, name, difficulty, description, max_steps)
- Reward function explanation (brief, bullet points)
- OpenEnv spec compliance checklist

---

## Important constraints
- All randomness uses fixed seeds — same seed = same episode
- Ground truth (true_fit_score, true_label) is NEVER sent in Observation — only in state()
- Do not use asyncio in environment.py — keep it synchronous
- server.py can be async (FastAPI) but env.step() calls are synchronous
- Python 3.11+, Pydantic v2
