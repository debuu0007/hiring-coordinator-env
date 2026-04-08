---
title: Hiring Coordinator Env
sdk: docker
app_port: 8000
tags:
  - openenv
  - hiring
  - recruitment
  - responsible-ai
---

# Hiring Coordinator Env

Hiring Coordinator Env is an OpenEnv-style real-world task simulation for evaluating agents on structured hiring coordination. The agent screens candidates, builds shortlists, ranks finalists, detects biased job-posting clauses, schedules interviews, sends candidate communications, and selects tailored interview questions across four tasks of increasing difficulty.

The design is intentionally deterministic: fixtures are generated from fixed seeds, hidden ground truth is used only by graders, and every grader returns a score in `[0.0, 1.0]`.

## Motivation

Hiring workflows are a practical agent-evaluation domain because real coordinators must balance ranking quality, process consistency, candidate communication, and policy-aware decision making. This environment models the work as a structured selection workflow rather than a generic recommender: it rewards job-related criteria and penalizes brittle process failures such as double booking, false bias flags, or premature submission.

Research inputs used to shape the environment:

- OpenEnv standardizes Gym-style `reset()`, `step()`, and `state()` APIs and supports HTTP/Docker deployment: https://meta-pytorch.org/OpenEnv/
- NYC Local Law 144 highlights automated employment decision-tool concerns such as bias audits and candidate notice: https://intro.nyc/local-laws/2021-144
- OPM guidance on structured interviews emphasizes standardized, job-related evaluation: https://piv.opm.gov/policy-data-oversight/assessment-and-selection/other-assessment-methods/structured-interviews/
- EEOC AI materials emphasize employment-selection risk, adverse impact, and accommodation concerns: https://www.eeoc.gov/ai

## Architecture

```text
Agent / inference.py
        |
        v
FastAPI server.py
        |
        v
HiringEnv environment.py
        |
        v
Deterministic graders.py + seeded data_generator.py
```

## Tasks

| Task | Difficulty | Description | Max steps | Grader |
| --- | --- | --- | ---: | --- |
| 1. Resume Screening | Easy | Screen 8 candidates as `fit`, `maybe`, or `no_fit`. | 16 | Positive-class F1 for strong-fit screening, with small partial credit for not rejecting strong fits. |
| 2. Screen, Shortlist, and Rank | Medium | Screen 12 candidates, shortlist the top 3, and rank them best to worst. | 30 | 35% screening F1, 25% shortlist F1, and 40% NDCG@3. |
| 3. Full Hiring Pipeline | Hard | Screen 15 candidates, flag biased job-posting clauses, shortlist/rank, schedule finalist interviews, and message finalists. | 45 | 30% screening, 25% ranking, 20% bias detection, 15% scheduling, and 10% finalist communications. |
| 4. Interview Question Selection | Hard | Run the full pipeline, then select tailored interview questions (language, OS, DBMS) for each shortlisted candidate from a fixed question bank. | 55 | 20% screening, 15% ranking, 15% bias, 10% scheduling, 10% comms, and 30% interview question selection. |

## Observation Space

The public observation is JSON with:

- `jd`: job description with title, required skills, preferred skills, public `posting_clauses`, experience, location, remote policy, and salary band.
- `candidates`: candidate views with ID, name, skills, experience, education, location, remote preference, salary expectation, and ATS-style match signals.
- Candidate match signals include `required_skills_matched`, `required_skills_missing`, `preferred_skills_matched`, `experience_meets_minimum`, `experience_delta`, `location_compatible`, and `salary_in_range`.
- `shortlist`: candidate IDs already shortlisted.
- `rejections`: candidate IDs mapped to rejection reasons.
- `interview_schedule`: candidate IDs mapped to slot IDs.
- `available_slots`: unbooked interview slots.
- `messages_sent`: candidate communication history.
- `question_bank`: fixed list of interview questions (Task 4 only), each with `question_id`, `domain`, `topic`, and `text`.
- `interview_questions_asked`: questions already submitted by the agent.
- `task_description`, `actions_taken`, `max_actions`, `task_id`, `done`.

Hidden labels such as `true_fit_score`, `true_label`, and `biased_clauses` are not returned in public observations or public HTTP state. The internal Python `HiringEnv.state()` method retains hidden fields for deterministic local debugging and grading.

## Action Space

Actions are JSON objects with an `action_type` discriminator:

```json
{"action_type":"screen","candidate_id":"c_000","decision":"fit"}
{"action_type":"shortlist","candidate_id":"c_000"}
{"action_type":"reject","candidate_id":"c_000","reason":"Missing required SQL experience"}
{"action_type":"rank","ordered_ids":["c_000","c_001","c_002"]}
{"action_type":"schedule","candidate_id":"c_000","slot_id":"slot_000"}
{"action_type":"send_message","candidate_id":"c_000","message_type":"schedule_confirm"}
{"action_type":"flag_bias","clause":"Recent graduate preferred","bias_type":"age"}
{"action_type":"ask_question","candidate_id":"c_000","question_id":"q_lang_python"}
{"action_type":"submit"}
```

## Reward Function

The reward is shaped throughout the trajectory, not only at terminal submission:

- Positive rewards for correct screening, strong shortlists, useful rankings, correct bias flags, valid scheduling, and appropriate messages.
- Partial rewards for borderline choices, such as marking a strong fit as `maybe`.
- Penalties for false positives, rejecting strong fits, duplicate work, invalid IDs, double-booked interview slots, scheduling non-shortlisted candidates, false bias flags, and wrong message types.
- Immediate shaping rewards are scaled so each step stays within the declared `[-1.0, 1.0]` reward range.
- Terminal reward aligns cumulative trajectory reward with the deterministic final grader score.

## HTTP API

```bash
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": 1, "seed": 42}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action": {"action_type": "submit"}}'
curl -X GET http://localhost:8000/state
```

Routes:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Quick Start

Install locally:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests
uvicorn server:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker build -t hiring-coordinator-env .
docker run --rm -p 8000:8000 hiring-coordinator-env
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

## Baseline Inference

The required baseline script is `inference.py` in the project root. It uses the OpenAI Python client against an OpenAI-compatible endpoint.

Required environment variables for judged inference:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct:fastest"
export HF_TOKEN="<your-token>"
python3 inference.py
```

The script emits one block per task using `[START]`, `[STEP]`, and `[END]` lines. With an API token, it uses a two-phase baseline: first it asks the model for structured candidate analysis from a compact ATS-style summary table, then it asks the model to convert that analysis into action JSON. If no API token is configured, it uses a deterministic but intentionally imperfect local heuristic only for smoke testing.

Reference smoke-test scores with seed `42` on this workspace:

| Task | Score |
| --- | ---: |
| Resume Screening | 0.80 |
| Screen, Shortlist, and Rank | 0.83 |
| Full Hiring Pipeline | 0.928 |
| Average | 0.853 |

## Validation

```bash
python3 validate.py
openenv validate
```

Local status from this workspace:

- `python3 -m pytest tests`: passing, 11 tests.
- FastAPI `/health`, `/reset`, and `/step`: smoke checked with `TestClient`.
- `openenv validate`: not run locally because the `openenv` CLI is not installed in this shell.
- `docker build`: not run locally because Docker is not installed in this shell.

## OpenEnv Compliance Checklist

- Typed Pydantic `Observation`, action models, and `Reward` model.
- `HiringEnv.reset()` returns the initial observation.
- `HiringEnv.step(action)` returns `(observation, reward, done, info)`.
- `HiringEnv.state()` returns current environment state.
- `openenv.yaml` is present with metadata, task list, reward range, and Docker/HTTP entrypoint.
- Root-level `inference.py` is present and uses the OpenAI client.
- `Dockerfile` starts the FastAPI server on port `8000`.
- 4 tasks include deterministic programmatic graders returning scores in `[0.0, 1.0]`.
