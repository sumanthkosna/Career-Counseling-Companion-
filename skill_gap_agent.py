"""
skill_gap_agent.py — Skill Gap Analysis Agent
───────────────────────────────────────────────
Compares the student's current skills against the requirements for the
top-matched careers and produces a prioritised skill gap report.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config
from knowledge_base_loader import load_career_requirements


def _build_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.TEMPERATURE: config.TEMPERATURE,
        GenParams.REPETITION_PENALTY: config.REPETITION_PENALTY,
    }
    return ModelInference(
        model_id=config.MODEL_ID,
        credentials=credentials,
        project_id=config.WATSONX_PROJECT_ID,
        params=params,
    )


def analyze_skill_gaps(
    student_profile: dict,
    skill_analysis: dict,
    career_matches: dict,
) -> dict:
    """
    Agent: Skill Gap Analysis Agent (uses RAG knowledge base)
    ──────────────────────────────────────────────────────────
    Input : student profile + skill assessment + career matches
    Output: dict with keys:
        - skill_gap_summary (str)
        - gaps_by_career (list[dict])  — per top career:
            { career_title, missing_technical_skills, missing_soft_skills,
              gap_severity: High/Medium/Low }
        - prioritised_gaps (list[dict])  — [{skill, priority, reason}]
        - student_strengths (list[str])
    """

    # ── RAG: pull requirements for matched careers ──────────────────────
    career_requirements_kb = load_career_requirements()
    requirements_context_lines = []
    top_careers = career_matches.get("top_careers", [])

    for career in top_careers:
        cid = career.get("career_id", "")
        req = career_requirements_kb.get("career_requirements", {}).get(cid, {})
        if req:
            tech_req  = req.get("technical_skills", [])
            soft_req  = req.get("soft_skills", [])
            requirements_context_lines.append(
                f"Career: {career.get('career_title','')}\n"
                f"  Required Technical: {', '.join(tech_req)}\n"
                f"  Required Soft: {', '.join(soft_req)}"
            )

    requirements_context = "\n".join(requirements_context_lines) or "No requirements data found."

    # ── Student skills summary ─────────────────────────────────────────
    tech_skills  = student_profile.get("technical_skills", [])
    soft_skills  = student_profile.get("soft_skills", [])
    skill_highs  = skill_analysis.get("skill_highlights", [])
    skill_inventory = skill_analysis.get("technical_skill_inventory", [])

    current_tech_str = ", ".join([
        f"{s['skill']} ({s['level']})" if isinstance(s, dict) else str(s)
        for s in skill_inventory
    ]) or ", ".join(tech_skills) or "Not provided"

    prompt = f"""<|system|>
You are the Skill Gap Analysis Agent in a career counseling system.
Compare the student's current skills to the requirements of their matched careers.
Identify gaps and prioritise them. Base analysis ONLY on provided data.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
## Career Requirements (from Knowledge Base)
{requirements_context}

## Student's Current Skills
- Technical Skills: {current_tech_str}
- Soft Skills: {", ".join(soft_skills) if soft_skills else "Not provided"}
- Skill Highlights: {", ".join(skill_highs) if skill_highs else "None"}

## Task
Perform a skill gap analysis. Return a JSON object with these exact keys:
- "skill_gap_summary": 2-3 sentence summary of the gaps found
- "gaps_by_career": list of objects, one per top career, each with:
    - "career_title": string
    - "missing_technical_skills": list of missing technical skills
    - "missing_soft_skills": list of missing soft skills
    - "gap_severity": one of ["High", "Medium", "Low"]
- "prioritised_gaps": list of objects ranked by priority, each with:
    - "skill": string
    - "priority": one of ["High", "Medium", "Low"]
    - "reason": why this skill is critical to acquire
- "student_strengths": list of 4-6 genuine strengths the student already has

Return ONLY the JSON object.
<|assistant|>
"""

    model = _build_model()
    response = model.generate_text(prompt=prompt)

    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        result = {
            "skill_gap_summary": response.strip(),
            "gaps_by_career": [],
            "prioritised_gaps": [],
            "student_strengths": skill_highs,
        }

    return result
