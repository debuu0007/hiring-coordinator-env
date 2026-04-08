from __future__ import annotations

from data_generator import BIASED_CLAUSE_TYPES
from environment import HiringEnv
from models import (
    FlagBiasAction,
    RankAction,
    ScheduleAction,
    SendMessageAction,
    ScreenAction,
    ShortlistAction,
    SubmitAction,
)
from server import public_observation


def _strong_fit(env: HiringEnv):
    return next(candidate for candidate in env._candidates if candidate.true_label == "strong_fit")


def _not_strong_fit(env: HiringEnv):
    return next(candidate for candidate in env._candidates if candidate.true_label != "strong_fit")


def _top_ids(env: HiringEnv, k: int = 3) -> list[str]:
    return [
        candidate.id
        for candidate in sorted(env._candidates, key=lambda item: item.true_fit_score, reverse=True)[:k]
    ]


def test_reset_returns_observation():
    env = HiringEnv(task_id=1, seed=42)
    observation = env.reset()

    assert observation.task_id == 1
    assert observation.task_description
    assert len(observation.candidates) == 8
    assert observation.candidates[0].required_skills_matched is not None
    assert observation.candidates[0].required_skills_missing is not None
    assert isinstance(observation.candidates[0].location_compatible, bool)
    assert observation.actions_taken == 0
    assert not observation.done


def test_step_screen_correct():
    env = HiringEnv(task_id=1, seed=42)
    candidate = _strong_fit(env)

    _, reward, done, _ = env.step(ScreenAction(candidate_id=candidate.id, decision="fit"))

    assert reward.step_reward > 0
    assert not done


def test_step_screen_wrong():
    env = HiringEnv(task_id=1, seed=42)
    candidate = _not_strong_fit(env)

    _, reward, _, _ = env.step(ScreenAction(candidate_id=candidate.id, decision="fit"))

    assert reward.step_reward < 0


def test_double_booking_penalty():
    env = HiringEnv(task_id=3, seed=42)
    candidate_a, candidate_b = _top_ids(env, k=2)
    slot_id = env._slots[0].slot_id

    env.step(ShortlistAction(candidate_id=candidate_a))
    env.step(ShortlistAction(candidate_id=candidate_b))
    env.step(ScheduleAction(candidate_id=candidate_a, slot_id=slot_id))
    _, reward, _, _ = env.step(ScheduleAction(candidate_id=candidate_b, slot_id=slot_id))

    assert reward.step_reward < 0


def test_submit_ends_episode():
    env = HiringEnv(task_id=1, seed=42)

    observation, _, done, info = env.step(SubmitAction())

    assert done
    assert observation.done
    assert "final_score" in info


def test_max_actions_auto_submit():
    env = HiringEnv(task_id=1, seed=42)
    candidate_id = env._candidates[0].id
    done = False

    for _ in range(env._max_actions):
        _, _, done, _ = env.step(ScreenAction(candidate_id=candidate_id, decision="maybe"))
        if done:
            break

    assert done


def test_task1_grader_perfect():
    env = HiringEnv(task_id=1, seed=42)

    for candidate in env._candidates:
        decision = "fit" if candidate.true_label == "strong_fit" else "no_fit"
        env.step(ScreenAction(candidate_id=candidate.id, decision=decision))
    _, _, done, info = env.step(SubmitAction())

    assert done
    assert info["final_score"] > 0.9


def test_task2_shortlist_top3():
    env = HiringEnv(task_id=2, seed=42)
    top_ids = _top_ids(env, k=3)

    for candidate in env._candidates:
        decision = "fit" if candidate.true_label == "strong_fit" else "no_fit"
        env.step(ScreenAction(candidate_id=candidate.id, decision=decision))
    for candidate_id in top_ids:
        env.step(ShortlistAction(candidate_id=candidate_id))
    env.step(RankAction(ordered_ids=top_ids))
    _, _, done, info = env.step(SubmitAction())

    assert done
    assert info["final_score"] > 0.8


def test_task3_full_pipeline():
    env = HiringEnv(task_id=3, seed=42)

    top_ids = _top_ids(env, k=3)
    for candidate in env._candidates:
        decision = "fit" if candidate.true_label == "strong_fit" else "no_fit"
        env.step(ScreenAction(candidate_id=candidate.id, decision=decision))
    for clause in env._jd.biased_clauses:
        env.step(FlagBiasAction(clause=clause, bias_type=BIASED_CLAUSE_TYPES[clause]))
    for candidate_id in top_ids:
        env.step(ShortlistAction(candidate_id=candidate_id))
    env.step(RankAction(ordered_ids=top_ids))
    for candidate_id, slot in zip(top_ids, env._slots):
        env.step(ScheduleAction(candidate_id=candidate_id, slot_id=slot.slot_id))
        env.step(SendMessageAction(candidate_id=candidate_id, message_type="schedule_confirm"))
    _, _, done, info = env.step(SubmitAction())

    assert done
    assert info["final_score"] > 0.8


def test_deterministic():
    env_a = HiringEnv(task_id=3, seed=123)
    env_b = HiringEnv(task_id=3, seed=123)

    assert env_a.reset().model_dump() == env_b.reset().model_dump()


def test_public_observation_hides_ground_truth_bias_flags():
    env = HiringEnv(task_id=3, seed=42)
    public = public_observation(env.reset())

    assert "posting_clauses" in public["jd"]
    assert "biased_clauses" not in public["jd"]
    assert "true_fit_score" not in public["candidates"][0]
