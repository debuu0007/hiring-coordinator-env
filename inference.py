from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from environment import HiringEnv
from models import SubmitAction
from server import ACTION_MAP, public_observation


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
BENCHMARK = os.getenv("HIRING_ENV_BENCHMARK") or "hiring-coordinator-env"
SEED = int(os.getenv("HIRING_ENV_SEED") or "42")
TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "3500")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or "0.5")

TASKS = {
    1: "resume-screening",
    2: "screen-shortlist-and-rank",
    3: "full-hiring-pipeline",
}

ANALYSIS_SYSTEM_PROMPT = """
You are a professional hiring coordinator using structured, job-related criteria.
Analyze candidate match evidence only. Do not infer demographic traits from names, education,
or location. Return ONLY a valid JSON object with this shape:
{"candidate_assessments":[{"candidate_id":"c_000","label":"fit|maybe|no_fit","rank_score":0.0,"rationale":"brief job-related rationale"}],"top_3":["c_000","c_001","c_002"],"bias_flags":[{"clause":"exact clause","bias_type":"age|nationality|disability|vague"}]}

Use these rules:
- Prioritize required skills, then preferred skills, then experience, then location/remote compatibility, then salary range.
- Favor candidates with fewer missing required skills.
- Use "fit" for strong overall matches, "maybe" for borderline matches, and "no_fit" for weak matches.
- For bias detection, flag only exact job-posting clauses that are not job-related or imply protected-class preferences.
- Do not flag neutral compensation, accommodation, rubric, or role-requirement clauses.
""".strip()

ACTION_SYSTEM_PROMPT = """
You are a professional hiring coordinator using structured, job-related criteria.
Return ONLY a valid JSON array of action objects. Do not use markdown or explanations.

Valid action schemas:
{"action_type":"screen","candidate_id":"c_000","decision":"fit|no_fit|maybe"}
{"action_type":"shortlist","candidate_id":"c_000"}
{"action_type":"reject","candidate_id":"c_000","reason":"job-related reason"}
{"action_type":"rank","ordered_ids":["c_000","c_001","c_002"]}
{"action_type":"schedule","candidate_id":"c_000","slot_id":"slot_000"}
{"action_type":"send_message","candidate_id":"c_000","message_type":"invite|reject|schedule_confirm|waitlist"}
{"action_type":"flag_bias","clause":"exact clause from jd.posting_clauses","bias_type":"gender|age|nationality|disability|vague"}
{"action_type":"submit"}

Use candidate names only as identifiers. Do not infer demographic traits from names, education,
or location. Base all hiring decisions only on these visible job-related fields:

Candidate fit scoring rubric:
- Required skills match: 40% of fit.
- Preferred skills match: 20% of fit.
- Experience match: 20% of fit. Full credit when years_experience >= jd.min_experience_years;
  partial credit for close misses; extra years help only up to a reasonable cap.
- Location/remote compatibility: 10% of fit. Give credit if candidate.remote_ok is true,
  jd.remote_ok is true, or candidate.location equals jd.location.
- Salary band compatibility: 10% of fit. Give credit if salary_expectation is between
  jd.salary_min and jd.salary_max; small partial credit if expectation is below the band.

Decision thresholds:
- Use "fit" only for candidates whose estimated score is at least 0.70.
- Use "maybe" for candidates around 0.40 to 0.69.
- Use "no_fit" for candidates below 0.40.

Ranking instructions:
- Estimate the score for every candidate before shortlisting.
- Shortlist exactly the top 3 candidates for tasks that ask for shortlisting.
- Rank ordered_ids from highest estimated score to lowest estimated score.
- Do not shortlist candidates just because they have many years of experience if they miss
  required skills or salary/location compatibility.

Bias detection instructions:
- Only flag exact clauses from jd.posting_clauses.
- Flag "young", "recent graduate", or age-coded preferences as age.
- Flag "Native English speaker" as nationality.
- Flag "rockstar" combined with long-hours expectations as disability.
- Flag vague non-job-related constraints such as immediate relocation or culture fit as vague.
- Do not flag neutral, job-related screening, compensation, accommodation, or rubric clauses.

Process instructions:
- In the full pipeline, schedule only shortlisted candidates into unique available slots.
- Send schedule_confirm or invite messages only to shortlisted candidates.
- Use submit as the final action.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
    error_val = "null" if not error else re.sub(r"\s+", " ", error)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def parse_action(data: dict[str, Any]):
    action_type = data.get("action_type")
    cls = ACTION_MAP.get(action_type)
    if cls is None:
        raise ValueError(f"Unknown action_type: {action_type}")
    return cls.model_validate(data)


def public_observation_json(env: HiringEnv) -> dict[str, Any]:
    return public_observation(env._build_observation())


def build_user_prompt(task_id: int, observation: dict[str, Any]) -> str:
    task_guidance = {
        1: (
            "Screen every candidate using the exact scoring rubric. Output one screen action "
            "per candidate, then submit."
        ),
        2: (
            "Estimate the score for every candidate using the exact scoring rubric. Output "
            "shortlist actions for exactly the best 3 candidates, one rank action ordered best "
            "to worst, then submit. Do not screen candidates unless needed."
        ),
        3: (
            "Screen every candidate using the exact scoring rubric, flag biased clauses exactly "
            "from posting_clauses, shortlist and rank exactly the best 3 candidates, schedule "
            "those shortlisted candidates into unique slots, send schedule_confirm messages to "
            "scheduled candidates, then submit."
        ),
    }
    return json.dumps(
        {
            "task_id": task_id,
            "goal": task_guidance[task_id],
            "important": (
                "Before choosing actions, calculate an estimated candidate fit score from the "
                "visible fields: required skills 0.40, preferred skills 0.20, experience 0.20, "
                "location/remote 0.10, salary 0.10. Return only the final action JSON array."
            ),
            "observation": observation,
        },
        ensure_ascii=True,
    )


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def build_observation_summary(observation: dict[str, Any]) -> str:
    jd = observation["jd"]
    candidates = observation["candidates"]
    lines = [
        f"Task: {observation['task_description']}",
        f"Job: {jd['title']}",
        f"Required skills: {format_list(jd['required_skills'])}",
        f"Preferred skills: {format_list(jd['preferred_skills'])}",
        f"Minimum experience: {jd['min_experience_years']} years",
        f"Location: {jd['location']} | Remote OK: {jd['remote_ok']}",
        f"Salary band: {jd['salary_min']} - {jd['salary_max']}",
        "",
        "Posting clauses:",
    ]
    lines.extend(f"- {clause}" for clause in jd.get("posting_clauses", []))
    lines.extend([
        "",
        "Candidates:",
        "| ID | Req Match | Req Missing | Pref Match | Exp Delta | Loc OK | Salary OK | Salary |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: |",
    ])
    for candidate in candidates:
        req_total = len(jd["required_skills"])
        pref_total = len(jd["preferred_skills"])
        lines.append(
            "| {id} | {req_count}/{req_total}: {req_match} | {req_missing} | "
            "{pref_count}/{pref_total}: {pref_match} | {exp_delta:+d} | {loc_ok} | {salary_ok} | {salary} |".format(
                id=candidate["id"],
                req_count=len(candidate["required_skills_matched"]),
                req_total=req_total,
                req_match=format_list(candidate["required_skills_matched"]),
                req_missing=format_list(candidate["required_skills_missing"]),
                pref_count=len(candidate["preferred_skills_matched"]),
                pref_total=pref_total,
                pref_match=format_list(candidate["preferred_skills_matched"]),
                exp_delta=candidate["experience_delta"],
                loc_ok="yes" if candidate["location_compatible"] else "no",
                salary_ok="yes" if candidate["salary_in_range"] else "no",
                salary=candidate["salary_expectation"],
            )
        )

    if observation.get("available_slots"):
        lines.extend(["", "Available interview slots:"])
        for slot in observation["available_slots"]:
            lines.append(f"- {slot['slot_id']}: {slot['datetime_str']} with {slot['interviewer_name']}")

    return "\n".join(lines)


def build_analysis_prompt(task_id: int, observation: dict[str, Any]) -> str:
    return (
        f"Analyze task {task_id}. Use the evidence table below. Return only the requested JSON object.\n\n"
        f"{build_observation_summary(observation)}"
    )


def build_action_prompt(task_id: int, observation: dict[str, Any], analysis: dict[str, Any]) -> str:
    action_goals = {
        1: "Create one screen action for every candidate, then submit.",
        2: "Create screen actions for every candidate, shortlist exactly the top 3, rank them best to worst, then submit.",
        3: (
            "Create screen actions for every candidate, flag biased clauses, shortlist exactly the top 3, "
            "rank them best to worst, schedule those 3 into unique slots, send schedule_confirm messages "
            "to those 3, then submit."
        ),
    }
    return json.dumps(
        {
            "task_id": task_id,
            "goal": action_goals[task_id],
            "analysis": analysis,
            "available_slots": observation.get("available_slots", []),
            "valid_candidate_ids": [candidate["id"] for candidate in observation["candidates"]],
            "instruction": "Return only a JSON array of action objects. Do not include rationale.",
        },
        ensure_ascii=True,
    )


def extract_json_actions(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start_positions = [pos for pos in [cleaned.find("["), cleaned.find("{")] if pos != -1]
    if start_positions:
        cleaned = cleaned[min(start_positions):]

    parsed = json.loads(cleaned)
    if isinstance(parsed, dict) and "actions" in parsed:
        parsed = parsed["actions"]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Model response was not a JSON action list")
    return [item for item in parsed if isinstance(item, dict)]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed


def get_model_actions(client: OpenAI | None, task_id: int, observation: dict[str, Any]) -> list[dict[str, Any]]:
    if client is None:
        return []
    analysis_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": build_analysis_prompt(task_id, observation)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    analysis = extract_json_object(analysis_completion.choices[0].message.content or "")

    action_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_action_prompt(task_id, observation, analysis)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return extract_json_actions(action_completion.choices[0].message.content or "")


def score_candidate(candidate: dict[str, Any], jd: dict[str, Any]) -> float:
    req_match = len(candidate["required_skills_matched"]) / max(len(jd["required_skills"]), 1)
    pref_match = len(candidate["preferred_skills_matched"]) / max(len(jd["preferred_skills"]), 1)
    exp_ok = 1.0 if candidate["experience_meets_minimum"] else 0.0
    loc_ok = 1.0 if candidate["location_compatible"] else 0.0
    salary_ok = 1.0 if candidate["salary_in_range"] else 0.0
    # Deliberately imperfect heuristic: useful for smoke tests, not a copy of hidden ground truth.
    return round(0.55 * req_match + 0.15 * pref_match + 0.15 * exp_ok + 0.10 * loc_ok + 0.05 * salary_ok, 4)


def score_to_decision(score: float) -> str:
    if score >= 0.68:
        return "fit"
    if score >= 0.38:
        return "maybe"
    return "no_fit"


def classify_bias_clause(clause: str) -> str | None:
    lowered = clause.lower()
    if "young" in lowered or "recent graduate" in lowered:
        return "age"
    if "native english" in lowered:
        return "nationality"
    if "long hours" in lowered or "rockstar" in lowered:
        return "disability"
    if "relocate immediately" in lowered or "cultural fit" in lowered:
        return "vague"
    return None


def fallback_actions(task_id: int, observation: dict[str, Any]) -> list[dict[str, Any]]:
    jd = observation["jd"]
    candidates = observation["candidates"]
    scored = sorted(
        ((candidate, score_candidate(candidate, jd)) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    top_ids = [candidate["id"] for candidate, _ in scored[:3]]

    actions: list[dict[str, Any]] = []
    if task_id in (1, 2, 3):
        for candidate, score in scored:
            actions.append(
                {
                    "action_type": "screen",
                    "candidate_id": candidate["id"],
                    "decision": score_to_decision(score),
                }
            )

    if task_id == 3:
        for clause in jd.get("posting_clauses", []):
            bias_type = classify_bias_clause(clause)
            if bias_type:
                actions.append({"action_type": "flag_bias", "clause": clause, "bias_type": bias_type})

    if task_id in (2, 3):
        for candidate_id in top_ids:
            actions.append({"action_type": "shortlist", "candidate_id": candidate_id})
        actions.append({"action_type": "rank", "ordered_ids": top_ids})

    if task_id == 3:
        for candidate_id, slot in zip(top_ids, observation.get("available_slots", [])):
            actions.append({"action_type": "schedule", "candidate_id": candidate_id, "slot_id": slot["slot_id"]})
        for candidate_id in top_ids:
            actions.append(
                {
                    "action_type": "send_message",
                    "candidate_id": candidate_id,
                    "message_type": "schedule_confirm",
                }
            )

    actions.append({"action_type": "submit"})
    return actions


ALLOWED_ACTIONS_BY_TASK = {
    1: {"screen", "submit"},
    2: {"screen", "shortlist", "rank", "submit"},
    3: {"screen", "flag_bias", "shortlist", "rank", "schedule", "send_message", "submit"},
}


def normalize_actions(task_id: int, actions: list[dict[str, Any]], observation: dict[str, Any]) -> list[dict[str, Any]]:
    valid_candidate_ids = {candidate["id"] for candidate in observation["candidates"]}
    valid_slot_ids = {slot["slot_id"] for slot in observation.get("available_slots", [])}
    allowed_actions = ALLOWED_ACTIONS_BY_TASK[task_id]
    normalized: list[dict[str, Any]] = []
    seen_shortlist: set[str] = set()

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = action.get("action_type")
        if action_type not in allowed_actions:
            continue

        candidate_id = action.get("candidate_id")
        if action_type in {"screen", "shortlist", "schedule", "send_message"} and candidate_id not in valid_candidate_ids:
            continue
        if action_type == "schedule" and action.get("slot_id") not in valid_slot_ids:
            continue
        if action_type == "shortlist":
            if candidate_id in seen_shortlist or len(seen_shortlist) >= 3:
                continue
            seen_shortlist.add(candidate_id)
        if action_type == "rank":
            ordered_ids = [
                candidate_id
                for candidate_id in action.get("ordered_ids", [])
                if candidate_id in valid_candidate_ids
            ]
            action = {"action_type": "rank", "ordered_ids": ordered_ids[:3]}
            if not action["ordered_ids"]:
                continue

        normalized.append(action)

    if not normalized or normalized[-1].get("action_type") != "submit":
        normalized.append({"action_type": "submit"})
    return normalized[: observation["max_actions"]]


def run_task(task_id: int, client: OpenAI | None) -> tuple[float, list[float]]:
    env = HiringEnv(task_id=task_id, seed=SEED)
    observation = public_observation_json(env)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    done = False

    log_start(task=TASKS[task_id], env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            model_actions = get_model_actions(client, task_id, observation)
        except Exception:
            model_actions = []
        actions = normalize_actions(task_id, model_actions or fallback_actions(task_id, observation), observation)

        for step_number, action_data in enumerate(actions, start=1):
            action_error = None
            try:
                action = parse_action(action_data)
                obs, reward_model, done, info = env.step(action)
                reward = reward_model.step_reward
                observation = public_observation(obs)
                score = float(info.get("final_score", score)) if info else score
            except (ValueError, ValidationError, RuntimeError) as exc:
                action_error = str(exc)
                action_data = {"action_type": "submit"}
                action = SubmitAction()
                obs, reward_model, done, info = env.step(action)
                reward = reward_model.step_reward
                observation = public_observation(obs)
                score = float(info.get("final_score", score)) if info else score

            rewards.append(reward)
            steps_taken = step_number
            log_step(step_number, action_data, reward, done, action_error)
            if done:
                break

        if not done:
            obs, reward_model, done, info = env.step(SubmitAction())
            rewards.append(reward_model.step_reward)
            steps_taken += 1
            score = float(info.get("final_score", score)) if info else score
            log_step(steps_taken, {"action_type": "submit"}, reward_model.step_reward, done, None)

        if score == 0.0:
            score, _ = env._compute_final_score()
    finally:
        log_end(success=score >= SUCCESS_SCORE_THRESHOLD, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
