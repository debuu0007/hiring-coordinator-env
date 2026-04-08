"""
Graders for all hiring tasks.
Each grader returns a float in [0.0, 1.0] and a breakdown dict.
All functions are pure and deterministic.
"""
from __future__ import annotations
import math
from models import Candidate, JobDescription
from data_generator import BIASED_CLAUSE_TYPES, PROGRAMMING_LANGUAGES, QUESTION_BANK_MAP


# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ndcg(predicted_ids: list[str], ground_truth_ids: list[str], k: int = 3) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.
    ground_truth_ids is ordered best → worst.
    """
    relevance = {cid: len(ground_truth_ids) - i for i, cid in enumerate(ground_truth_ids)}

    def dcg(ids):
        return sum(
            relevance.get(cid, 0) / math.log2(rank + 2)
            for rank, cid in enumerate(ids[:k])
        )

    ideal_dcg = dcg(ground_truth_ids[:k])
    if ideal_dcg == 0:
        return 0.0
    return min(1.0, dcg(predicted_ids) / ideal_dcg)


# ---------------------------------------------------------------------------
# Task 1 grader — Screen 10 resumes
# ---------------------------------------------------------------------------

def grade_task1(
    agent_decisions: dict[str, str],       # {candidate_id → "fit"|"no_fit"|"maybe"}
    candidates: list[Candidate],
) -> tuple[float, dict]:
    """
    Scoring:
      - Binary classification: fit (strong_fit) vs not (maybe + no_fit)
      - "maybe" is treated as a partial credit case
      - F1 score on the positive class (fit)
    """
    tp = fp = fn = tn = 0
    partial_credit = 0.0

    for c in candidates:
        pred = agent_decisions.get(c.id, "no_fit")
        true_positive = c.true_label == "strong_fit"

        if true_positive and pred == "fit":
            tp += 1
        elif true_positive and pred == "maybe":
            partial_credit += 0.5       # partial: didn't reject but didn't commit
            fn += 1
        elif true_positive and pred == "no_fit":
            fn += 1
        elif not true_positive and pred == "fit":
            fp += 1
        elif not true_positive and pred == "maybe":
            partial_credit += 0.0       # no credit for maybe on a non-fit
            tn += 1
        else:
            tn += 1

    base_f1 = _f1(tp, fp, fn)
    partial_bonus = partial_credit / max(len(candidates), 1) * 0.1  # max 10% bonus
    final_score = min(1.0, base_f1 + partial_bonus)

    return final_score, {
        "f1_score": round(base_f1, 4),
        "partial_credit_bonus": round(partial_bonus, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "final": round(final_score, 4),
    }


# ---------------------------------------------------------------------------
# Task 2 grader — Shortlist top 3 + rank
# ---------------------------------------------------------------------------

def grade_task2(
    agent_decisions: dict[str, str],        # {candidate_id -> "fit"|"no_fit"|"maybe"}
    agent_shortlist: list[str],            # up to 5 ids agent shortlisted
    agent_ranking: list[str],              # ordered ids best → worst
    candidates: list[Candidate],
    top_k: int = 3,
) -> tuple[float, dict]:
    """
    Scoring:
      - 35%: screening F1
      - 25%: shortlist quality (did they pick the right people?)
      - 40%: ranking quality NDCG@3 (did they order them correctly?)
    """
    screen_score, screen_breakdown = grade_task1(agent_decisions, candidates)

    # Ground truth: sort by true_fit_score descending
    sorted_candidates = sorted(candidates, key=lambda c: c.true_fit_score, reverse=True)
    gt_top_ids = [c.id for c in sorted_candidates[:top_k]]

    # Shortlist quality: what fraction of shortlisted are genuinely good?
    shortlist_set = set(agent_shortlist)
    gt_top_set = set(gt_top_ids)

    shortlist_tp = len(shortlist_set & gt_top_set)
    shortlist_fp = len(shortlist_set - gt_top_set)
    shortlist_fn = len(gt_top_set - shortlist_set)
    shortlist_f1 = _f1(shortlist_tp, shortlist_fp, shortlist_fn)

    # Ranking quality: NDCG on submitted ranking
    # Only evaluate ranking within the shortlisted candidates
    all_sorted_ids = [c.id for c in sorted_candidates]
    ndcg = _ndcg(agent_ranking, all_sorted_ids, k=top_k)

    final_score = 0.35 * screen_score + 0.25 * shortlist_f1 + 0.40 * ndcg

    return final_score, {
        "screening_f1": round(screen_score, 4),
        "screening_tp": screen_breakdown["tp"],
        "screening_fp": screen_breakdown["fp"],
        "screening_fn": screen_breakdown["fn"],
        "shortlist_f1": round(shortlist_f1, 4),
        "ndcg_at_3": round(ndcg, 4),
        "gt_top_ids": gt_top_ids,
        "agent_shortlist": agent_shortlist,
        "final": round(final_score, 4),
    }


# ---------------------------------------------------------------------------
# Task 3 grader — Full pipeline (screen + bias + schedule + comms)
# ---------------------------------------------------------------------------

def grade_task3(
    agent_decisions: dict[str, str],
    agent_shortlist: list[str],
    agent_ranking: list[str],
    agent_bias_flags: list[dict],           # [{clause, bias_type}]
    interview_schedule: dict[str, str],     # {candidate_id → slot_id}
    messages_sent: list[dict],              # [{candidate_id, message_type}]
    candidates: list[Candidate],
    jd: JobDescription,
) -> tuple[float, dict]:
    """
    Composite score:
      30% screening F1
      25% ranking NDCG@3
      20% bias detection
      15% scheduling
      10% finalist communications
    """
    # --- Component 1: Screening ---
    screen_score, screen_breakdown = grade_task1(agent_decisions, candidates)

    # --- Component 2: Ranking ---
    _, rank_breakdown = grade_task2(agent_decisions, agent_shortlist, agent_ranking, candidates)
    rank_score = rank_breakdown["ndcg_at_3"]

    # --- Component 3: Bias detection ---
    planted_clauses = set(jd.biased_clauses)
    planted_pairs = {
        (clause, BIASED_CLAUSE_TYPES.get(clause))
        for clause in planted_clauses
    }
    flagged_pairs = {
        (f["clause"], f.get("bias_type"))
        for f in agent_bias_flags
    }

    bias_tp = len(planted_pairs & flagged_pairs)
    bias_fp = len(flagged_pairs - planted_pairs)
    bias_fn = len(planted_pairs - flagged_pairs)

    # Penalize false positives heavily (over-flagging is problematic in HR)
    bias_precision = bias_tp / (bias_tp + bias_fp) if (bias_tp + bias_fp) > 0 else 0.0
    bias_recall    = bias_tp / (bias_tp + bias_fn) if (bias_tp + bias_fn) > 0 else 0.0
    # Use F-beta with beta=0.5 (precision-weighted)
    beta = 0.5
    if bias_precision + bias_recall > 0:
        bias_score = (1 + beta**2) * bias_precision * bias_recall / (beta**2 * bias_precision + bias_recall)
    else:
        bias_score = 0.0
    bias_score = min(1.0, bias_score)

    # --- Component 4: Process quality ---
    # 4a: Schedule valid shortlisted candidates (no double-booking)
    shortlisted_set = set(agent_shortlist)
    scheduled_set = set(interview_schedule.keys())
    scheduled_shortlisted = scheduled_set & shortlisted_set

    slot_ids_used = list(interview_schedule.values())
    double_booked = len(slot_ids_used) != len(set(slot_ids_used))

    schedule_score = len(scheduled_shortlisted) / max(len(shortlisted_set), 1)
    if double_booked:
        schedule_score *= 0.5   # halve for conflict

    # 4b: Comms — shortlisted candidates got invite or schedule confirmation
    shortlisted_invited = 0

    for msg in messages_sent:
        cid = msg["candidate_id"]
        mtype = msg["message_type"]
        if cid in shortlisted_set and mtype in ("invite", "schedule_confirm"):
            shortlisted_invited += 1

    comms_score = shortlisted_invited / max(len(shortlisted_set), 1)

    # --- Final composite ---
    final_score = (
        0.30 * screen_score +
        0.25 * rank_score +
        0.20 * bias_score +
        0.15 * schedule_score +
        0.10 * comms_score
    )

    return round(final_score, 4), {
        "screening_f1": round(screen_score, 4),
        "ranking_ndcg": round(rank_score, 4),
        "bias_score": round(bias_score, 4),
        "bias_tp": bias_tp, "bias_fp": bias_fp, "bias_fn": bias_fn,
        "schedule_score": round(schedule_score, 4),
        "comms_score": round(comms_score, 4),
        "final": round(final_score, 4),
    }


# ---------------------------------------------------------------------------
# Task 4 grader — Full pipeline + interview question selection
# ---------------------------------------------------------------------------

def _grade_interview_questions(
    questions_asked: list[dict],
    agent_shortlist: list[str],
    candidates: list[Candidate],
    jd: JobDescription,
) -> tuple[float, dict]:
    """
    Grade interview question selection for shortlisted candidates.
    Each shortlisted candidate should receive one question per domain
    (language, os, dbms). Score averages across candidates x domains.
    """
    candidate_map = {c.id: c for c in candidates}
    jd_skills = set(jd.required_skills) | set(jd.preferred_skills)

    per_candidate: dict[str, dict[str, float]] = {cid: {} for cid in agent_shortlist}

    for entry in questions_asked:
        cid = entry.get("candidate_id")
        qid = entry.get("question_id")
        if cid not in per_candidate:
            continue
        question = QUESTION_BANK_MAP.get(qid)
        if question is None:
            continue
        domain = question.domain
        if domain in per_candidate[cid]:
            continue

        candidate = candidate_map.get(cid)
        if candidate is None:
            continue

        if domain == "language":
            candidate_langs = set(candidate.skills) & PROGRAMMING_LANGUAGES
            if question.topic in candidate_langs:
                score = 1.0
                if question.topic in jd_skills:
                    score = min(1.0, score + 0.25)
            else:
                score = 0.0
        elif domain == "os":
            score = 1.0 if question.topic == candidate.os_proficiency else 0.0
        elif domain == "dbms":
            score = 1.0 if question.topic == candidate.dbms_proficiency else 0.0
        else:
            score = 0.0

        per_candidate[cid][domain] = score

    total_checks = max(len(agent_shortlist) * 3, 1)
    total_score = 0.0
    for cid in agent_shortlist:
        for domain in ("language", "os", "dbms"):
            total_score += per_candidate.get(cid, {}).get(domain, 0.0)

    interview_score = total_score / total_checks

    return round(interview_score, 4), {
        "interview_score": round(interview_score, 4),
        "per_candidate": {
            cid: per_candidate.get(cid, {}) for cid in agent_shortlist
        },
    }


def grade_task4(
    agent_decisions: dict[str, str],
    agent_shortlist: list[str],
    agent_ranking: list[str],
    agent_bias_flags: list[dict],
    interview_schedule: dict[str, str],
    messages_sent: list[dict],
    questions_asked: list[dict],
    candidates: list[Candidate],
    jd: JobDescription,
) -> tuple[float, dict]:
    """
    Composite score:
      20% screening F1
      15% ranking NDCG@3
      15% bias detection
      10% scheduling
      10% finalist communications
      30% interview question selection
    """
    pipeline_score, pipeline_breakdown = grade_task3(
        agent_decisions=agent_decisions,
        agent_shortlist=agent_shortlist,
        agent_ranking=agent_ranking,
        agent_bias_flags=agent_bias_flags,
        interview_schedule=interview_schedule,
        messages_sent=messages_sent,
        candidates=candidates,
        jd=jd,
    )

    interview_score, interview_breakdown = _grade_interview_questions(
        questions_asked=questions_asked,
        agent_shortlist=agent_shortlist,
        candidates=candidates,
        jd=jd,
    )

    screen_score = pipeline_breakdown["screening_f1"]
    rank_score = pipeline_breakdown["ranking_ndcg"]
    bias_score = pipeline_breakdown["bias_score"]
    schedule_score = pipeline_breakdown["schedule_score"]
    comms_score = pipeline_breakdown["comms_score"]

    final_score = (
        0.20 * screen_score +
        0.15 * rank_score +
        0.15 * bias_score +
        0.10 * schedule_score +
        0.10 * comms_score +
        0.30 * interview_score
    )

    return round(final_score, 4), {
        "screening_f1": round(screen_score, 4),
        "ranking_ndcg": round(rank_score, 4),
        "bias_score": round(bias_score, 4),
        "bias_tp": pipeline_breakdown["bias_tp"],
        "bias_fp": pipeline_breakdown["bias_fp"],
        "bias_fn": pipeline_breakdown["bias_fn"],
        "schedule_score": round(schedule_score, 4),
        "comms_score": round(comms_score, 4),
        "interview_score": round(interview_score, 4),
        "interview_per_candidate": interview_breakdown["per_candidate"],
        "final": round(final_score, 4),
    }
