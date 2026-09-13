"""
interest_agent.py — Interest & Preference Agent
────────────────────────────────────────────────
Analyses the student's interests, hobbies, preferred work style, and
career goals to build an interest profile that guides career matching.
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
        GenParams.MAX_NEW_TOKENS: 700,
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


def analyze_interests(student_profile: dict) -> dict:
    """
    Agent: Interest & Preference Agent
    ────────────────────────────────────
    Input : student_profile dict
    Output: dict with keys:
        - interest_summary (str)
        - dominant_interest_domains (list[str])
        - preferred_work_style (str)
        - personality_traits_inferred (list[str])
        - interest_career_alignment (list[str])  — career paths that align
        - interest_insights (list[str])
    """

    interests    = student_profile.get("interests", [])
    hobbies      = student_profile.get("hobbies", [])
    work_type    = student_profile.get("preferred_work_type", "Not specified")
    career_goal  = student_profile.get("career_goal", "Not specified")

    prompt = f"""<|system|>
You are the Interest & Preference Agent in a career counseling system.
Analyse the student's interests, hobbies, work preferences, and career goals.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
Analyse the student's interest profile and return a JSON object with these exact keys:
- "interest_summary": 2-3 sentence paragraph summarising the interest profile
- "dominant_interest_domains": list of 3-5 interest domains (e.g. Technology, Research, Creativity)
- "preferred_work_style": single string describing preferred work style
- "personality_traits_inferred": list of 3-5 personality traits inferred from interests/hobbies
- "interest_career_alignment": list of 3-5 career paths that align with these interests
- "interest_insights": list of 3-5 actionable insights for the counselor

Student Interest Profile:
- Interests: {", ".join(interests) if interests else "Not provided"}
- Hobbies: {", ".join(hobbies) if hobbies else "Not provided"}
- Preferred Work Type: {work_type}
- Career Goal: {career_goal}

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
            "interest_summary": response.strip(),
            "dominant_interest_domains": interests[:3] if interests else [],
            "preferred_work_style": work_type,
            "personality_traits_inferred": [],
            "interest_career_alignment": [],
            "interest_insights": ["Analysis completed — see summary above."],
        }

    return result
