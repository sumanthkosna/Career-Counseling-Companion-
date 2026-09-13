"""
skill_agent.py — Skill Assessment Agent
────────────────────────────────────────
Evaluates the student's technical and soft skills, projects,
internships, and certifications to produce a skills inventory.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config


def _build_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: 800,
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


def assess_skills(student_profile: dict) -> dict:
    """
    Agent: Skill Assessment Agent
    ──────────────────────────────
    Input : student_profile dict
    Output: dict with keys:
        - skill_summary (str)
        - technical_skill_inventory (list[dict])  — [{skill, level}]
        - soft_skill_inventory (list[str])
        - experience_summary (str)
        - overall_skill_score (float 0-10)
        - skill_highlights (list[str])
        - areas_for_development (list[str])
    """

    tech_skills    = student_profile.get("technical_skills", [])
    soft_skills    = student_profile.get("soft_skills", [])
    projects       = student_profile.get("projects", [])
    internships    = student_profile.get("internships", [])
    certifications = student_profile.get("certifications", [])

    tech_str  = ", ".join(tech_skills)    if tech_skills    else "None listed"
    soft_str  = ", ".join(soft_skills)    if soft_skills    else "None listed"
    proj_str  = "; ".join(projects)       if projects       else "None listed"
    intern_str = "; ".join(internships)   if internships    else "None listed"
    cert_str  = "; ".join(certifications) if certifications else "None listed"

    prompt = f"""<|system|>
You are the Skill Assessment Agent in a career counseling system.
Evaluate the student's skill set, practical experience, and certifications.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
Assess the student's skills and experience. Return a JSON object with these exact keys:
- "skill_summary": 2-3 sentence summary of the student's skill profile
- "technical_skill_inventory": list of objects, each with "skill" (string) and "level" (one of: Beginner/Intermediate/Advanced)
- "soft_skill_inventory": list of soft skills the student possesses
- "experience_summary": 1-2 sentence summary of projects/internship experience
- "overall_skill_score": float 0.0-10.0 reflecting overall skill readiness
- "skill_highlights": list of 3-5 strongest skill highlights
- "areas_for_development": list of 3-5 skill gaps or areas to improve

Student Skills & Experience:
- Technical Skills: {tech_str}
- Soft Skills: {soft_str}
- Projects: {proj_str}
- Internships / Experience: {intern_str}
- Certifications: {cert_str}

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
            "skill_summary": response.strip(),
            "technical_skill_inventory": [{"skill": s, "level": "Intermediate"} for s in tech_skills],
            "soft_skill_inventory": soft_skills,
            "experience_summary": "Experience details processed.",
            "overall_skill_score": 5.5,
            "skill_highlights": tech_skills[:3] if tech_skills else [],
            "areas_for_development": [],
        }

    return result
