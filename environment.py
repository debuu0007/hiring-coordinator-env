"""
HiringEnv — OpenEnv-compliant hiring coordinator environment.
Implements: reset(), step(), state()
"""
from __future__ import annotations
from typing import Any
from models import (
    Action, Observation, Reward, CandidateView,
    ScreenAction, ShortlistAction, RejectAction, RankAction,
    ScheduleAction, SendMessageAction, FlagBiasAction, SubmitAction,
)
from data_generator import (
    generate_jd, generate_candidate_pool, generate_interview_slots,
    BIASED_CLAUSE_TYPES,
)
from graders import grade_task1, grade_task2, grade_task3


# Task config: pool_size, max_actions, plant_bias, require_schedule
TASK_CONFIG = {
    1: {"pool_size": 8, "max_actions": 16, "plant_bias": False, "require_schedule": False},
    2: {"pool_size": 12, "max_actions": 30, "plant_bias": False, "require_schedule": False},
    3: {"pool_size": 15, "max_actions": 45, "plant_bias": True,  "require_schedule": True},
}

TASK_DESCRIPTIONS = {
    1: "Screen 8 candidates as fit, maybe, or no_fit using job-related match evidence.",
    2: "Screen 12 candidates, shortlist exactly the top 3, and rank them best to worst.",
    3: (
        "Run the full hiring pipeline: screen 15 candidates, flag biased job-posting clauses, "
        "shortlist and rank the top 3, schedule them into unique slots, and send finalist messages."
    ),
}

SHAPING_REWARD_SCALE = 0.2


class HiringEnv:
    def __init__(self, task_id: int = 1, seed: int = 42, domain: str = "engineering"):
        assert task_id in (1, 2, 3), "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.seed = seed
        self.domain = domain
        self._state: dict[str, Any] = {}
        self._cumulative_reward = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        cfg = TASK_CONFIG[self.task_id]
        self._jd = generate_jd(
            domain=self.domain,
            seed=self.seed,
            plant_bias=cfg["plant_bias"],
        )
        self._candidates = generate_candidate_pool(
            jd=self._jd,
            pool_size=cfg["pool_size"],
            seed=self.seed,
            domain=self.domain,
        )
        self._slots = generate_interview_slots(n=6, seed=self.seed) if cfg["require_schedule"] else []
        self._max_actions = cfg["max_actions"]

        # Mutable state
        self._screen_decisions: dict[str, str] = {}
        self._shortlist: list[str] = []
        self._rejections: dict[str, str] = {}
        self._ranking: list[str] = []
        self._schedule: dict[str, str] = {}           # candidate_id → slot_id
        self._messages: list[dict] = []
        self._bias_flags: list[dict] = []
        self._actions_taken = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._episode_info: dict = {}

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._actions_taken += 1
        step_reward, breakdown = self._apply_action(action)
        if not isinstance(action, SubmitAction):
            step_reward *= SHAPING_REWARD_SCALE
            breakdown["shaping_scale"] = SHAPING_REWARD_SCALE

        # Action budget exhaustion = forced submit
        if self._actions_taken >= self._max_actions and not self._done:
            self._done = True
            final_score, final_breakdown = self._compute_final_score()
            terminal_reward = final_score - self._cumulative_reward  # delta
            step_reward += terminal_reward
            breakdown.update(final_breakdown)
            breakdown["forced_submit"] = True

        self._cumulative_reward += step_reward
        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            breakdown=breakdown,
        )
        obs = self._build_observation()
        return obs, reward, self._done, self._episode_info

    def state(self) -> dict:
        """Full internal state (for debugging / logging)."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "jd": self._jd.model_dump(),
            "candidates": [c.model_dump() for c in self._candidates],  # includes ground truth
            "screen_decisions": self._screen_decisions,
            "shortlist": self._shortlist,
            "rejections": self._rejections,
            "ranking": self._ranking,
            "schedule": self._schedule,
            "messages": self._messages,
            "bias_flags": self._bias_flags,
            "actions_taken": self._actions_taken,
            "max_actions": self._max_actions,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> tuple[float, dict]:
        if isinstance(action, ScreenAction):
            return self._handle_screen(action)
        elif isinstance(action, ShortlistAction):
            return self._handle_shortlist(action)
        elif isinstance(action, RejectAction):
            return self._handle_reject(action)
        elif isinstance(action, RankAction):
            return self._handle_rank(action)
        elif isinstance(action, ScheduleAction):
            return self._handle_schedule(action)
        elif isinstance(action, SendMessageAction):
            return self._handle_message(action)
        elif isinstance(action, FlagBiasAction):
            return self._handle_bias_flag(action)
        elif isinstance(action, SubmitAction):
            return self._handle_submit(action)
        else:
            return -0.05, {"error": "unknown_action"}

    def _handle_screen(self, action: ScreenAction) -> tuple[float, dict]:
        candidate = self._get_candidate(action.candidate_id)
        if candidate is None:
            return -0.05, {"error": "invalid_candidate_id"}

        # Already screened → small penalty for redundancy
        if action.candidate_id in self._screen_decisions:
            return -0.03, {"penalty": "already_screened"}

        self._screen_decisions[action.candidate_id] = action.decision
        true_label = candidate.true_label

        # Map agent decision to binary
        agent_positive = action.decision == "fit"
        agent_maybe = action.decision == "maybe"
        true_positive = true_label == "strong_fit"

        if agent_positive and true_positive:
            r = +0.15
            tag = "correct_fit"
        elif agent_maybe and true_positive:
            r = +0.07  # partial credit
            tag = "partial_maybe_fit"
        elif not agent_positive and not true_positive:
            r = +0.10
            tag = "correct_reject"
        elif agent_positive and not true_positive:
            r = -0.10
            tag = "false_positive"
        elif agent_maybe and not true_positive:
            r = -0.02
            tag = "unnecessary_maybe"
        else:
            r = -0.08
            tag = "missed_fit"

        return r, {"screen": tag, "candidate_id": action.candidate_id}

    def _handle_shortlist(self, action: ShortlistAction) -> tuple[float, dict]:
        candidate = self._get_candidate(action.candidate_id)
        if candidate is None:
            return -0.05, {"error": "invalid_candidate_id"}
        if action.candidate_id in self._shortlist:
            return -0.03, {"penalty": "already_shortlisted"}

        self._shortlist.append(action.candidate_id)
        true_positive = candidate.true_label == "strong_fit"

        if true_positive:
            r = +0.20
            tag = "good_shortlist"
        elif candidate.true_label == "maybe":
            r = +0.05  # borderline — acceptable
            tag = "borderline_shortlist"
        else:
            r = -0.15  # shortlisted a clear no-fit
            tag = "bad_shortlist"

        return r, {"shortlist": tag, "candidate_id": action.candidate_id}

    def _handle_reject(self, action: RejectAction) -> tuple[float, dict]:
        candidate = self._get_candidate(action.candidate_id)
        if candidate is None:
            return -0.05, {"error": "invalid_candidate_id"}

        self._rejections[action.candidate_id] = action.reason
        true_positive = candidate.true_label == "strong_fit"

        if not true_positive:
            r = +0.08
            tag = "correct_reject"
        else:
            r = -0.20   # wrongly rejected a strong fit — painful
            tag = "wrong_reject"

        return r, {"reject": tag, "candidate_id": action.candidate_id}

    def _handle_rank(self, action: RankAction) -> tuple[float, dict]:
        # Light immediate reward based on how good the ranking looks
        ranked_candidates = [self._get_candidate(cid) for cid in action.ordered_ids if self._get_candidate(cid)]
        if not ranked_candidates:
            return -0.05, {"error": "empty_ranking"}

        self._ranking = action.ordered_ids

        # Quick quality signal: is the top-ranked candidate a strong fit?
        top = ranked_candidates[0]
        r = +0.10 if top.true_label == "strong_fit" else -0.05

        return r, {"rank": "submitted", "top_candidate_fit": top.true_label}

    def _handle_schedule(self, action: ScheduleAction) -> tuple[float, dict]:
        slot = next((s for s in self._slots if s.slot_id == action.slot_id), None)
        if slot is None:
            return -0.05, {"error": "invalid_slot_id"}
        if slot.is_booked:
            return -0.20, {"penalty": "double_booking"}  # hard penalty

        # Check candidate is shortlisted
        if action.candidate_id not in self._shortlist:
            return -0.05, {"penalty": "scheduling_non_shortlisted"}

        slot.is_booked = True
        slot.booked_by_candidate = action.candidate_id
        self._schedule[action.candidate_id] = action.slot_id

        return +0.10, {"schedule": "booked", "slot_id": action.slot_id}

    def _handle_message(self, action: SendMessageAction) -> tuple[float, dict]:
        candidate = self._get_candidate(action.candidate_id)
        if candidate is None:
            return -0.05, {"error": "invalid_candidate_id"}

        # Check if already messaged
        already_messaged = any(
            m["candidate_id"] == action.candidate_id for m in self._messages
        )
        if already_messaged:
            return -0.03, {"penalty": "duplicate_message"}

        self._messages.append({
            "candidate_id": action.candidate_id,
            "message_type": action.message_type,
        })

        is_shortlisted = action.candidate_id in self._shortlist
        is_rejected = action.candidate_id in self._rejections

        # Correct comms:
        if is_shortlisted and action.message_type in ("invite", "schedule_confirm"):
            r = +0.08
        elif is_rejected and action.message_type == "reject":
            r = +0.05
        else:
            r = -0.05   # wrong message type for this candidate's status

        return r, {"message": action.message_type, "candidate_id": action.candidate_id}

    def _handle_bias_flag(self, action: FlagBiasAction) -> tuple[float, dict]:
        planted = self._jd.biased_clauses
        expected_type = BIASED_CLAUSE_TYPES.get(action.clause)
        if action.clause in planted and action.bias_type == expected_type:
            self._bias_flags.append({"clause": action.clause, "bias_type": action.bias_type})
            return +0.15, {"bias_flag": "correct", "clause": action.clause}
        elif action.clause in planted:
            self._bias_flags.append({"clause": action.clause, "bias_type": action.bias_type})
            return +0.05, {"bias_flag": "wrong_bias_type", "clause": action.clause}
        else:
            self._bias_flags.append({"clause": action.clause, "bias_type": action.bias_type})
            return -0.10, {"bias_flag": "false_positive", "clause": action.clause}

    def _handle_submit(self, action: SubmitAction) -> tuple[float, dict]:
        self._done = True
        final_score, breakdown = self._compute_final_score()
        # Align terminal cumulative reward with the deterministic final grader score.
        terminal_r = final_score - self._cumulative_reward
        self._episode_info = {"final_score": final_score, **breakdown}
        return terminal_r, {"submit": True, **breakdown}

    # ------------------------------------------------------------------
    # Final grading
    # ------------------------------------------------------------------

    def _compute_final_score(self) -> tuple[float, dict]:
        if self.task_id == 1:
            score, breakdown = grade_task1(self._screen_decisions, self._candidates)
        elif self.task_id == 2:
            score, breakdown = grade_task2(
                self._screen_decisions, self._shortlist, self._ranking, self._candidates, top_k=3
            )
        else:
            score, breakdown = grade_task3(
                agent_decisions=self._screen_decisions,
                agent_shortlist=self._shortlist,
                agent_ranking=self._ranking,
                agent_bias_flags=self._bias_flags,
                interview_schedule=self._schedule,
                messages_sent=self._messages,
                candidates=self._candidates,
                jd=self._jd,
            )
        return score, breakdown

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_candidate(self, candidate_id: str):
        return next((c for c in self._candidates if c.id == candidate_id), None)

    def _build_candidate_view(self, candidate) -> CandidateView:
        required = set(self._jd.required_skills)
        preferred = set(self._jd.preferred_skills)
        skills = set(candidate.skills)

        return CandidateView(
            id=candidate.id,
            name=candidate.name,
            skills=candidate.skills,
            years_experience=candidate.years_experience,
            education=candidate.education,
            location=candidate.location,
            remote_ok=candidate.remote_ok,
            salary_expectation=candidate.salary_expectation,
            required_skills_matched=sorted(required & skills),
            required_skills_missing=sorted(required - skills),
            preferred_skills_matched=sorted(preferred & skills),
            experience_meets_minimum=candidate.years_experience >= self._jd.min_experience_years,
            experience_delta=candidate.years_experience - self._jd.min_experience_years,
            location_compatible=(
                candidate.remote_ok
                or self._jd.remote_ok
                or candidate.location == self._jd.location
            ),
            salary_in_range=self._jd.salary_min <= candidate.salary_expectation <= self._jd.salary_max,
        )

    def _build_observation(self) -> Observation:
        return Observation(
            task_description=TASK_DESCRIPTIONS[self.task_id],
            jd=self._jd,
            candidates=[self._build_candidate_view(candidate) for candidate in self._candidates],
            shortlist=list(self._shortlist),
            rejections=dict(self._rejections),
            interview_schedule=dict(self._schedule),
            available_slots=[s for s in self._slots if not s.is_booked],
            messages_sent=list(self._messages),
            actions_taken=self._actions_taken,
            max_actions=self._max_actions,
            task_id=self.task_id,
            done=self._done,
        )
