"""
market_agent.py — Labor Market Trend Agent
────────────────────────────────────────────
Retrieves and interprets current labor-market trends for the student's
top-matched careers using the market-trends knowledge base.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config
from knowledge_base_loader import load_market_trends, load_career_paths


def _build_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: 900,
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


def analyze_market_trends(
    career_matches: dict,
    student_profile: dict,
) -> dict:
    """
    Agent: Labor Market Trend Agent (uses RAG knowledge base)
    ──────────────────────────────────────────────────────────
    Input : career_matches + student_profile
    Output: dict with keys:
        - market_summary (str)
        - career_market_data (list[dict])  — per top career:
            { career_title, demand_trend, yoy_growth, hiring_sectors,
              emerging_skills, geographic_hotspots, remote_opportunity }
        - global_outlook (str)
        - strategic_recommendation (str)
        - emerging_skills_to_watch (list[str])
    """

    # ── RAG: Load market trends and career path info ─────────────────────
    market_kb  = load_market_trends()
    careers_kb = load_career_paths()

    top_careers = career_matches.get("top_careers", [])
    market_context_lines = []

    career_market_data_raw = []

    for career in top_careers:
        cid   = career.get("career_id", "")
        title = career.get("career_title", "")
        role_data = market_kb.get("market_trends", {}).get("roles", {}).get(cid, {})

        if role_data:
            career_market_data_raw.append({
                "career_title": title,
                "demand_trend": role_data.get("demand_trend", "N/A"),
                "yoy_growth": role_data.get("yoy_job_growth_percent", "N/A"),
                "hiring_sectors": role_data.get("key_hiring_sectors", []),
                "emerging_skills": role_data.get("emerging_skills", []),
                "geographic_hotspots": role_data.get("geographic_hotspots", []),
                "remote_opportunity": role_data.get("remote_opportunity", "N/A"),
                "threat_from_automation": role_data.get("threat_from_automation", "N/A"),
            })
            market_context_lines.append(
                f"Career: {title}\n"
                f"  Demand: {role_data.get('demand_trend','N/A')} | "
                f"Growth: {role_data.get('yoy_job_growth_percent','?')}% YoY\n"
                f"  Hiring Sectors: {', '.join(role_data.get('key_hiring_sectors',[]))}\n"
                f"  Emerging Skills: {', '.join(role_data.get('emerging_skills',[]))}\n"
                f"  Remote: {role_data.get('remote_opportunity','N/A')}"
            )

    global_outlook = market_kb.get("market_trends", {}).get("global_outlook_2025", {})
    global_summary = global_outlook.get("summary", "")
    key_drivers    = global_outlook.get("key_drivers", [])

    market_context_str = "\n\n".join(market_context_lines) or "No market data available."
    global_ctx = f"Global Outlook: {global_summary}\nKey Drivers: {', '.join(key_drivers)}"

    prompt = f"""<|system|>
You are the Labor Market Trend Agent in a career counseling system.
Interpret the labor-market data for the student's matched careers and provide insights.
Do NOT speculate beyond the provided data. Return ONLY valid JSON — no markdown fences.
<|user|>
## Market Trend Data (from Knowledge Base)
{market_context_str}

## Global Technology Market Context
{global_ctx}

## Student Context
- Career Goal: {student_profile.get("career_goal", "Not specified")}
- Degree: {student_profile.get("degree", "")}

## Task
Return a JSON object with these exact keys:
- "market_summary": 2-3 sentence paragraph summarising market conditions for the student
- "career_market_data": the list of career market data objects from the knowledge base
  (include all data fields: career_title, demand_trend, yoy_growth, hiring_sectors,
   emerging_skills, geographic_hotspots, remote_opportunity, threat_from_automation)
- "global_outlook": 1-2 sentence summary of the global tech market
- "strategic_recommendation": 1-2 sentences advising the student on market timing and positioning
- "emerging_skills_to_watch": list of 5-7 emerging skills the student should monitor

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
        # Structured fallback using KB data directly
        result = {
            "market_summary": response.strip(),
            "career_market_data": career_market_data_raw,
            "global_outlook": global_summary,
            "strategic_recommendation": "Review market trends and align skills with emerging demands.",
            "emerging_skills_to_watch": [
                s for c in career_market_data_raw for s in c.get("emerging_skills", [])
            ][:7],
        }

    return result
