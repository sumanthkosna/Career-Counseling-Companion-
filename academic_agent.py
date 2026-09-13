"""
academic_agent.py — Academic Performance Agent
──────────────────────────────────────────────
Analyses the student's academic background (degree, CGPA, subjects,
strong/weak areas) and returns a structured academic profile summary.

This agent is called first in the orchestration pipeline.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config


def _build_model() -> ModelInference:
    """Initialise IBM Granite model via watsonx.ai SDK."""
    credentials = Credentials(
        url=config.WATSONX_URL,
        api_key=config.WATSONX_API_KEY,
    )
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


def analyze_academic_performance(student_profile: dict) -> dict:
    """
    Agent: Academic Performance Agent
    ──────────────────────────────────
    Input : student_profile dict (degree, cgpa, subjects, strong/weak subjects)
    Output: dict with keys:
        - academic_summary (str)
        - academic_strength_score (float 0-10)
        - strong_areas (list[str])
        - weak_areas (list[str])
        - academic_insights (list[str])
        - cgpa_percentile_band (str)
    """

    # Build context string from student data
    degree       = student_profile.get("degree", "Not specified")
    specialization = student_profile.get("specialization", "Not specified")
    cgpa         = student_profile.get("cgpa", "Not specified")
    subjects     = student_profile.get("academic_subjects", [])
    strong_subs  = student_profile.get("strong_subjects", [])
    weak_subs    = student_profile.get("weak_subjects", [])

    subjects_str  = ", ".join(subjects)  if subjects  else "Not provided"
    strong_str    = ", ".join(strong_subs) if strong_subs else "Not provided"
    weak_str      = ", ".join(weak_subs)   if weak_subs   else "Not provided"

    prompt = f"""<|system|>
You are the Academic Performance Agent in a career counseling system.
Your task is to analyse a student's academic background and produce a structured JSON analysis.
Base your analysis strictly on the information provided. Do not invent data.
Return ONLY valid JSON, no markdown fences, no extra text.
<|user|>
Analyse the following student's academic profile and return a JSON object with these exact keys:
- "academic_summary": a 2-3 sentence paragraph summarising the academic profile
- "academic_strength_score": a float from 0.0 to 10.0 reflecting overall academic strength
- "strong_areas": list of academic strength areas inferred from the subjects and strong subjects
- "weak_areas": list of areas that need improvement inferred from weak subjects
- "academic_insights": list of 3-5 actionable bullet-point insights for a career counselor
- "cgpa_percentile_band": one of ["Exceptional (9+)", "Strong (7.5-9)", "Good (6-7.5)", "Average (5-6)", "Below Average (<5)"]

Student Academic Profile:
- Degree: {degree}
- Specialisation: {specialization}
- CGPA/Percentage: {cgpa}
- Academic Subjects Studied: {subjects_str}
- Strong Subjects: {strong_str}
- Weak Subjects: {weak_str}

Return ONLY the JSON object.
<|assistant|>
"""

    model = _build_model()
    response = model.generate_text(prompt=prompt)

    # Robustly parse the JSON response
    try:
        # Strip any accidental markdown fences
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        # Graceful fallback — return structured defaults so pipeline continues
        result = {
            "academic_summary": response.strip(),
            "academic_strength_score": 6.0,
            "strong_areas": strong_subs,
            "weak_areas": weak_subs,
            "academic_insights": ["Analysis completed — see summary above."],
            "cgpa_percentile_band": "Good (6-7.5)",
        }

    return result
