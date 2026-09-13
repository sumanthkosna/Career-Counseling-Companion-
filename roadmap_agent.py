"""
roadmap_agent.py — Personalized Learning Roadmap Agent
────────────────────────────────────────────────────────
Generates a structured, time-boxed learning roadmap using the skill-gap
analysis and the learning resources / certifications knowledge base.
"""

import json
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config
from knowledge_base_loader import (
    load_learning_resources,
    load_certifications,
    load_career_requirements,
)


def _build_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: 1500,
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


def generate_roadmap(
    student_profile: dict,
    skill_gap_analysis: dict,
    career_matches: dict,
) -> dict:
    """
    Agent: Personalized Learning Roadmap Agent (uses RAG knowledge base)
    ──────────────────────────────────────────────────────────────────────
    Input : skill gap analysis + career matches
    Output: dict with keys:
        - roadmap_summary (str)
        - phases (list[dict])  — 3 phases:
            { phase_name, duration, focus_areas, resources (list), milestones (list) }
        - recommended_certifications (list[dict])  — [{name, provider, level}]
        - recommended_projects (list[dict])  — [{title, description, skills_practiced}]
        - immediate_actions (list[str])  — things to do this week
        - total_estimated_duration (str)
    """

    # ── RAG: Pull learning resources & certifications ───────────────────
    learning_kb = load_learning_resources()
    cert_kb     = load_certifications()
    req_kb      = load_career_requirements()

    # Collect gaps and primary career
    prioritised_gaps = skill_gap_analysis.get("prioritised_gaps", [])
    top_career = career_matches.get("top_careers", [{}])[0]
    top_career_id    = top_career.get("career_id", "")
    top_career_title = top_career.get("career_title", "")

    # Retrieve recommended resources for high-priority gaps
    resources_context_lines = []
    for gap in prioritised_gaps[:6]:
        skill_name = gap.get("skill", "")
        # Fuzzy-match skill to resource keys
        for key, resources in learning_kb.get("learning_resources", {}).items():
            if any(word.lower() in key.lower() or key.lower() in skill_name.lower()
                   for word in skill_name.split()):
                for r in resources[:2]:
                    resources_context_lines.append(
                        f"  [{key}] {r['title']} by {r['provider']} ({r['level']}, {r['duration']}) — {r['url']}"
                    )
                break

    # Certifications for primary career
    certs_for_career = cert_kb.get("certifications", {}).get(top_career_id, [])[:4]
    certs_context = "\n".join([
        f"  - {c['name']} ({c['provider']}, {c['level']})" for c in certs_for_career
    ]) or "  (No specific certifications found in KB)"

    # Recommended projects from career requirements KB
    req_projects = req_kb.get("career_requirements", {}).get(top_career_id, {}).get("projects_recommended", [])

    gaps_str = "\n".join([
        f"  - {g.get('skill','')} [Priority: {g.get('priority','')}]: {g.get('reason','')}"
        for g in prioritised_gaps[:8]
    ]) or "  No significant gaps identified."

    resources_str = "\n".join(resources_context_lines) or "  (Check online platforms for resources)"

    prompt = f"""<|system|>
You are the Personalized Learning Roadmap Agent in a career counseling system.
Create a practical, time-boxed learning roadmap for the student based on their skill gaps
and the available learning resources. Be specific and actionable.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
## Primary Career Target
{top_career_title} (id: {top_career_id})

## Identified Skill Gaps (Prioritised)
{gaps_str}

## Available Learning Resources (from Knowledge Base)
{resources_str}

## Recommended Certifications (from Knowledge Base)
{certs_context}

## Suggested Projects (from Knowledge Base)
{", ".join(req_projects) if req_projects else "Not specified"}

## Student Background
- Degree: {student_profile.get("degree","")}, CGPA: {student_profile.get("cgpa","")}
- Current Technical Skills: {", ".join(student_profile.get("technical_skills",[]))}

## Task
Create a personalised learning roadmap. Return a JSON object with these exact keys:
- "roadmap_summary": 2-3 sentence description of the overall roadmap
- "phases": list of exactly 3 phase objects, each with:
    - "phase_name": string (e.g. "Phase 1: Foundation Building")
    - "duration": string (e.g. "Months 1-2")
    - "focus_areas": list of 3-5 skills/topics to focus on
    - "resources": list of 2-4 specific resource names to use
    - "milestones": list of 2-3 measurable milestones
- "recommended_certifications": list of 2-4 certification objects, each with:
    - "name": certification name
    - "provider": provider name
    - "level": difficulty level
    - "when_to_pursue": e.g. "After Phase 2"
- "recommended_projects": list of 2-4 project objects, each with:
    - "title": project title
    - "description": 1-2 sentence description
    - "skills_practiced": list of skills this project will develop
- "immediate_actions": list of 5-7 things the student should start doing THIS WEEK
- "total_estimated_duration": string (e.g. "6-8 months")

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
            "roadmap_summary": response.strip(),
            "phases": [],
            "recommended_certifications": certs_for_career,
            "recommended_projects": [{"title": p, "description": "", "skills_practiced": []} for p in req_projects],
            "immediate_actions": ["Review identified skill gaps", "Sign up for an online learning platform"],
            "total_estimated_duration": "6-12 months",
        }

    return result
