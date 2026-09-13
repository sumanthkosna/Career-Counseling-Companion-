"""
career_matching_agent.py — Career Matching Agent
─────────────────────────────────────────────────
Uses the academic analysis, interest analysis, and skill assessment
plus the RAG knowledge base to recommend the top 3 most suitable career paths.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config
from knowledge_base_loader import load_career_paths, load_career_requirements


def _build_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: 1200,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.TEMPERATURE: config.TEMPERATURE,
        GenParams.REPETITION_PENALTY: config.REPETITION_PENALTY,
    }
    return ModelInference(
        model_id=config.MODEL_ID,
        credentials=credentials,
        project_id=config.WATSONX_PROJECT_ID,
        params=params,
    )


def match_careers(
    student_profile: dict,
    academic_analysis: dict,
    interest_analysis: dict,
    skill_analysis: dict,
) -> dict:
    """
    Agent: Career Matching Agent (uses RAG knowledge base)
    ────────────────────────────────────────────────────────
    Input : student_profile + prior agent outputs
    Output: dict with keys:
        - top_careers (list[dict])  — top 3, each with:
            { career_title, career_id, match_score (0-100), match_level, reasons (list[str]) }
        - matching_summary (str)
        - alignment_notes (str)
    """

    # ── RAG: Retrieve knowledge-base context ──────────────────────────────
    career_paths_kb    = load_career_paths()
    career_requirements_kb = load_career_requirements()

    # Serialise a concise excerpt of the KB to include in the prompt
    # (only titles + key skills to stay within token limits)
    kb_summary_lines = []
    for cp in career_paths_kb.get("career_paths", []):
        req = career_requirements_kb.get("career_requirements", {}).get(cp["id"], {})
        tech_req = req.get("technical_skills", [])[:5]
        kb_summary_lines.append(
            f"- {cp['title']} (id: {cp['id']}): domain={cp['domain']}, "
            f"demand={cp.get('job_demand','?')}, "
            f"key_skills=[{', '.join(tech_req)}]"
        )
    kb_context = "\n".join(kb_summary_lines)

    # ── Prepare student context ────────────────────────────────────────────
    degree     = student_profile.get("degree", "")
    spec       = student_profile.get("specialization", "")
    cgpa       = student_profile.get("cgpa", "")
    tech_skills = student_profile.get("technical_skills", [])
    career_goal = student_profile.get("career_goal", "")

    ac_score   = academic_analysis.get("academic_strength_score", 6)
    strong_areas = academic_analysis.get("strong_areas", [])
    interest_domains = interest_analysis.get("dominant_interest_domains", [])
    interest_careers = interest_analysis.get("interest_career_alignment", [])
    skill_score = skill_analysis.get("overall_skill_score", 5)
    skill_highs = skill_analysis.get("skill_highlights", [])

    prompt = f"""<|system|>
You are the Career Matching Agent in a career counseling system.
You have access to a knowledge base of career paths and their requirements.
Match the student's profile to the TOP 3 most suitable careers.
Base your reasoning ONLY on the student data and knowledge base provided.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
## Knowledge Base — Available Career Paths
{kb_context}

## Student Profile Summary
- Degree: {degree} — {spec}
- CGPA: {cgpa}
- Technical Skills: {", ".join(tech_skills)}
- Career Goal: {career_goal}
- Academic Strength Score: {ac_score}/10
- Academic Strong Areas: {", ".join(strong_areas)}
- Interest Domains: {", ".join(interest_domains)}
- Careers matching interests: {", ".join(interest_careers)}
- Overall Skill Score: {skill_score}/10
- Skill Highlights: {", ".join(skill_highs)}

## Task
Return a JSON object with these exact keys:
- "top_careers": list of exactly 3 objects, each with:
    - "career_title": string
    - "career_id": string (must match an id from the knowledge base)
    - "match_score": integer 0-100
    - "match_level": one of ["Excellent Match", "Strong Match", "Good Match", "Moderate Match"]
    - "reasons": list of 3-5 specific reasons this career suits this student
- "matching_summary": 2-3 sentence paragraph summarising the matching outcome
- "alignment_notes": 1-2 sentences on how the student's goals align with market demand

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
        # Fallback: provide basic structure
        result = {
            "top_careers": [
                {
                    "career_title": "Software Engineer",
                    "career_id": "software_engineer",
                    "match_score": 75,
                    "match_level": "Strong Match",
                    "reasons": ["Technical skills aligned", "Good academic background"],
                }
            ],
            "matching_summary": response.strip(),
            "alignment_notes": "Further analysis recommended.",
        }

    return result
