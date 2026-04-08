"""
Deterministic synthetic data generator.
All randomness is seeded — same seed = same episode every time.
"""
from __future__ import annotations
import random
from models import Candidate, JobDescription, InterviewSlot

# ---------------------------------------------------------------------------
# Skill ontology
# ---------------------------------------------------------------------------

SKILL_POOL = {
    "engineering": ["Python", "Java", "Go", "Rust", "C++", "SQL", "TypeScript",
                    "Kubernetes", "Docker", "AWS", "React", "PostgreSQL", "Redis"],
    "data":        ["Python", "SQL", "Spark", "Pandas", "ML", "Tableau",
                    "dbt", "Airflow", "Statistics", "R"],
    "pm":          ["Roadmapping", "Jira", "Agile", "SQL", "Figma",
                    "Stakeholder Mgmt", "OKRs", "User Research"],
    "design":      ["Figma", "Sketch", "User Research", "Prototyping",
                    "Accessibility", "Design Systems"],
}

EDUCATION_LEVELS = ["high_school", "bachelors", "masters", "phd"]
LOCATIONS = ["New York", "San Francisco", "Austin", "Remote", "London", "Bangalore"]

# Biased clauses to plant in Task 3 JDs
BIASED_CLAUSE_TYPES = {
    "Must be a cultural fit for our young, energetic team": "age",
    "Native English speaker preferred": "nationality",
    "Looking for a rockstar developer who can work long hours": "disability",
    "Recent graduate preferred": "age",
    "Must be willing to relocate immediately": "vague",
}

BIASED_CLAUSES = list(BIASED_CLAUSE_TYPES.keys())

NEUTRAL_CLAUSES = [
    "Interviewers will use the same job-related rubric for every candidate",
    "Candidates may request reasonable interview accommodations",
    "Screening focuses on role-relevant skills, experience, work authorization, and availability",
    "Compensation discussions should stay within the published salary band",
]


# ---------------------------------------------------------------------------
# Ground truth scoring (hidden from agent)
# ---------------------------------------------------------------------------

def compute_fit_score(candidate: dict, jd: dict) -> float:
    """
    Deterministic fit score based on weighted criteria.
    This is the ground truth the grader uses.
    """
    score = 0.0

    # 1. Required skills match (weight: 0.40)
    required = set(jd["required_skills"])
    candidate_skills = set(candidate["skills"])
    req_overlap = len(required & candidate_skills) / max(len(required), 1)
    score += 0.40 * req_overlap

    # 2. Preferred skills match (weight: 0.20)
    preferred = set(jd["preferred_skills"])
    pref_overlap = len(preferred & candidate_skills) / max(len(preferred), 1)
    score += 0.20 * pref_overlap

    # 3. Experience match (weight: 0.20)
    exp_min = jd["min_experience_years"]
    if candidate["years_experience"] >= exp_min:
        exp_score = min(1.0, candidate["years_experience"] / (exp_min + 3))
    else:
        # Partial credit for close misses
        exp_score = max(0.0, candidate["years_experience"] / exp_min) * 0.5
    score += 0.20 * exp_score

    # 4. Location / remote match (weight: 0.10)
    if candidate["remote_ok"] or jd["remote_ok"] or candidate["location"] == jd["location"]:
        score += 0.10

    # 5. Salary match (weight: 0.10)
    sal_exp = candidate["salary_expectation"]
    if jd["salary_min"] <= sal_exp <= jd["salary_max"]:
        score += 0.10
    elif sal_exp < jd["salary_min"]:
        score += 0.05  # under expectation — acceptable

    return round(score, 4)


def score_to_label(score: float) -> str:
    if score >= 0.70:
        return "strong_fit"
    elif score >= 0.40:
        return "maybe"
    else:
        return "no_fit"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_jd(domain: str = "engineering", seed: int = 42, plant_bias: bool = False) -> JobDescription:
    rng = random.Random(seed)
    pool = SKILL_POOL[domain]

    required = rng.sample(pool, k=rng.randint(3, 5))
    preferred = rng.sample([s for s in pool if s not in required], k=rng.randint(2, 3))
    min_experience_years = rng.randint(2, 6)
    location = rng.choice(LOCATIONS)
    remote_ok = rng.choice([True, False])

    biased = []
    if plant_bias:
        biased = rng.sample(BIASED_CLAUSES, k=rng.randint(2, 3))

    posting_clauses = [
        f"This role requires {', '.join(required)} and values {', '.join(preferred)}",
        f"Target experience is {min_experience_years}+ years in a related role",
        f"Primary work location is {location}; remote work is {'supported' if remote_ok else 'not supported'}",
    ] + rng.sample(NEUTRAL_CLAUSES, k=2) + biased
    rng.shuffle(posting_clauses)

    return JobDescription(
        id=f"jd_{seed}",
        title=f"Senior {domain.title()} Engineer" if domain != "pm" else "Product Manager",
        required_skills=required,
        preferred_skills=preferred,
        posting_clauses=posting_clauses,
        min_experience_years=min_experience_years,
        location=location,
        remote_ok=remote_ok,
        salary_min=100_000,
        salary_max=160_000,
        biased_clauses=biased,
    )


def generate_candidate_pool(
    jd: JobDescription,
    pool_size: int = 10,
    seed: int = 42,
    domain: str = "engineering",
) -> list[Candidate]:
    """
    Generates a pool with a realistic distribution:
      ~20% strong_fit, ~35% maybe, ~45% no_fit
    Ensures at least 2 strong fits so tasks are solvable.
    """
    rng = random.Random(seed + 1000)
    pool = SKILL_POOL[domain]
    candidates = []

    for i in range(pool_size):
        cand_seed = seed + i
        rng_c = random.Random(cand_seed)

        # Vary overlap with JD deliberately
        overlap_factor = rng_c.random()   # 0 = bad match, 1 = great match

        n_skills = rng_c.randint(3, 8)
        if overlap_factor > 0.65:
            # Good match: take mostly from JD skills
            skills = rng_c.sample(jd.required_skills, k=min(len(jd.required_skills), 3))
            extras = rng_c.sample(pool, k=max(0, n_skills - len(skills)))
            skills = list(set(skills + extras))
        else:
            skills = rng_c.sample(pool, k=n_skills)

        years_exp = rng_c.randint(0, 12)
        sal_exp = rng_c.randint(80_000, 200_000)

        raw = {
            "skills": skills,
            "years_experience": years_exp,
            "salary_expectation": sal_exp,
            "remote_ok": rng_c.choice([True, False]),
            "location": rng_c.choice(LOCATIONS),
            "required_skills": jd.required_skills,
            "preferred_skills": jd.preferred_skills,
            "min_experience_years": jd.min_experience_years,
            "salary_min": jd.salary_min,
            "salary_max": jd.salary_max,
        }

        fit_score = compute_fit_score(raw, {
            "required_skills": jd.required_skills,
            "preferred_skills": jd.preferred_skills,
            "min_experience_years": jd.min_experience_years,
            "remote_ok": jd.remote_ok,
            "location": jd.location,
            "salary_min": jd.salary_min,
            "salary_max": jd.salary_max,
        })

        candidates.append(Candidate(
            id=f"c_{i:03d}",
            name=f"Candidate {i+1}",
            skills=skills,
            years_experience=years_exp,
            education=rng_c.choice(EDUCATION_LEVELS),
            location=raw["location"],
            remote_ok=raw["remote_ok"],
            salary_expectation=sal_exp,
            true_fit_score=fit_score,
            true_label=score_to_label(fit_score),
        ))

    # Guarantee solvability: ensure at least 2 strong fits
    strong_fits = [c for c in candidates if c.true_label == "strong_fit"]
    if len(strong_fits) < 2:
        # Patch first 2 candidates to be strong fits
        for j in range(min(2, pool_size)):
            patched = candidates[j].model_copy(update={
                "skills": list(set(jd.required_skills + jd.preferred_skills)),
                "years_experience": max(jd.min_experience_years + 1, 3),
                "salary_expectation": (jd.salary_min + jd.salary_max) // 2,
            })
            recalculated_score = compute_fit_score(patched.model_dump(), jd.model_dump())
            candidates[j] = patched.model_copy(update={
                "true_fit_score": recalculated_score,
                "true_label": score_to_label(recalculated_score),
            })

    return candidates


def generate_interview_slots(n: int = 6, seed: int = 42) -> list[InterviewSlot]:
    """Generate N available interview slots."""
    rng = random.Random(seed + 9999)
    interviewers = [
        ("iv_001", "Alice Chen"),
        ("iv_002", "Bob Martinez"),
        ("iv_003", "Carol Singh"),
    ]
    dates = [
        "2024-06-10T10:00:00", "2024-06-10T14:00:00",
        "2024-06-11T09:00:00", "2024-06-11T15:00:00",
        "2024-06-12T11:00:00", "2024-06-12T16:00:00",
        "2024-06-13T10:00:00", "2024-06-13T14:00:00",
    ]

    slots = []
    used_combos = set()
    for i in range(n):
        while True:
            iv = rng.choice(interviewers)
            dt = rng.choice(dates)
            key = (iv[0], dt)
            if key not in used_combos:
                used_combos.add(key)
                break
        slots.append(InterviewSlot(
            slot_id=f"slot_{i:03d}",
            datetime_str=dt,
            interviewer_id=iv[0],
            interviewer_name=iv[1],
        ))
    return slots
