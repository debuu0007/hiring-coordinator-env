from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

class JobDescription(BaseModel):
    id: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    posting_clauses: list[str] = Field(default_factory=list)
    min_experience_years: int
    location: str
    remote_ok: bool
    salary_min: int
    salary_max: int
    # Planted bias flags (hidden from agent, used by grader)
    biased_clauses: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    id: str
    name: str
    skills: list[str]
    years_experience: int
    education: Literal["high_school", "bachelors", "masters", "phd"]
    location: str
    remote_ok: bool
    salary_expectation: int
    # Ground truth (hidden from agent observation)
    true_fit_score: float = Field(ge=0.0, le=1.0)
    true_label: Literal["strong_fit", "maybe", "no_fit"]
    os_proficiency: str = "Linux"
    dbms_proficiency: str = "MySQL"


class InterviewQuestion(BaseModel):
    question_id: str
    domain: Literal["language", "os", "dbms"]
    topic: str
    text: str


class InterviewSlot(BaseModel):
    slot_id: str
    datetime_str: str          # ISO format string
    interviewer_id: str
    interviewer_name: str
    is_booked: bool = False
    booked_by_candidate: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation (what the agent sees each step)
# ---------------------------------------------------------------------------

class CandidateView(BaseModel):
    """Candidate as seen by agent — no ground truth labels."""
    id: str
    name: str
    skills: list[str]
    years_experience: int
    education: str
    location: str
    remote_ok: bool
    salary_expectation: int
    required_skills_matched: list[str]
    required_skills_missing: list[str]
    preferred_skills_matched: list[str]
    experience_meets_minimum: bool
    experience_delta: int
    location_compatible: bool
    salary_in_range: bool


class Observation(BaseModel):
    task_description: str
    jd: JobDescription
    candidates: list[CandidateView]
    shortlist: list[str]                     # candidate ids
    rejections: dict[str, str]               # candidate_id → reason given
    interview_schedule: dict[str, str]       # candidate_id → slot_id
    available_slots: list[InterviewSlot]
    messages_sent: list[dict]                # {candidate_id, message_type}
    question_bank: list[InterviewQuestion] = Field(default_factory=list)
    interview_questions_asked: list[dict] = Field(default_factory=list)
    actions_taken: int
    max_actions: int
    task_id: int
    done: bool


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ScreenAction(BaseModel):
    action_type: Literal["screen"] = "screen"
    candidate_id: str
    decision: Literal["fit", "no_fit", "maybe"]


class ShortlistAction(BaseModel):
    action_type: Literal["shortlist"] = "shortlist"
    candidate_id: str


class RejectAction(BaseModel):
    action_type: Literal["reject"] = "reject"
    candidate_id: str
    reason: str


class RankAction(BaseModel):
    action_type: Literal["rank"] = "rank"
    ordered_ids: list[str]      # best → worst


class ScheduleAction(BaseModel):
    action_type: Literal["schedule"] = "schedule"
    candidate_id: str
    slot_id: str


class SendMessageAction(BaseModel):
    action_type: Literal["send_message"] = "send_message"
    candidate_id: str
    message_type: Literal["invite", "reject", "schedule_confirm", "waitlist"]


class FlagBiasAction(BaseModel):
    action_type: Literal["flag_bias"] = "flag_bias"
    clause: str
    bias_type: Literal["gender", "age", "nationality", "disability", "vague"]


class AskInterviewQuestionAction(BaseModel):
    action_type: Literal["ask_question"] = "ask_question"
    candidate_id: str
    question_id: str


class SubmitAction(BaseModel):
    action_type: Literal["submit"] = "submit"


Action = (
    ScreenAction
    | ShortlistAction
    | RejectAction
    | RankAction
    | ScheduleAction
    | SendMessageAction
    | FlagBiasAction
    | AskInterviewQuestionAction
    | SubmitAction
)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    breakdown: dict[str, Any]   # component -> value, for interpretability
