"""
career_agent.py — Career Counselor / Orchestrator Agent
─────────────────────────────────────────────────────────
This is the central orchestrator that:
  1. Validates credentials and student input
  2. Runs each specialist agent in the defined workflow order
  3. Passes intermediate results between agents (pipeline context)
  4. Calls the Final Career Advisor Agent to synthesise everything
  5. Returns the complete career counseling report

Workflow:
  Student Profile
      → Academic Performance Agent
      → Interest & Preference Agent
      → Skill Assessment Agent
      → Career Matching Agent
      → Skill Gap Analysis Agent
      → Learning Roadmap Agent
      → Labor Market Trend Agent
      → Final Career Advisor Agent
      → Complete Report

IBM watsonx Orchestrate Integration Note:
─────────────────────────────────────────
Each agent function below represents a watsonx Orchestrate "skill" or "tool".
In a full IBM watsonx Orchestrate deployment, these would be registered as
individual skills in an Orchestrate instance, and the orchestrator flow would
be defined using the Orchestrate Builder UI or YAML DSL.

For this implementation, the orchestration is handled programmatically via
this Python module, keeping the integration clearly separated so it can be
moved to watsonx Orchestrate with minimal refactoring.
"""

import json
import time
from typing import Callable, Any

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import config
from academic_agent        import analyze_academic_performance
from interest_agent        import analyze_interests
from skill_agent           import assess_skills
from career_matching_agent import match_careers
from skill_gap_agent       import analyze_skill_gaps
from roadmap_agent         import generate_roadmap
from market_agent          import analyze_market_trends
from knowledge_base_loader import load_interview_topics


# ── Final Career Advisor Agent ─────────────────────────────────────────────

def _build_final_advisor_model() -> ModelInference:
    credentials = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    params = {
        GenParams.DECODING_METHOD: config.DECODING_METHOD,
        GenParams.MAX_NEW_TOKENS: config.MAX_NEW_TOKENS,
        GenParams.MIN_NEW_TOKENS: config.MIN_NEW_TOKENS,
        GenParams.TEMPERATURE: config.TEMPERATURE,
        GenParams.REPETITION_PENALTY: config.REPETITION_PENALTY,
    }
    return ModelInference(
        model_id=config.MODEL_ID,
        credentials=credentials,
        project_id=config.WATSONX_PROJECT_ID,
        params=params,
    )


def _run_final_advisor(
    student_profile: dict,
    academic_analysis: dict,
    interest_analysis: dict,
    skill_analysis: dict,
    career_matches: dict,
    skill_gap_analysis: dict,
    roadmap: dict,
    market_analysis: dict,
) -> dict:
    """
    Agent: Final Career Advisor Agent
    ────────────────────────────────────
    Synthesises all prior agent outputs into a final personalised career
    counseling report with recommendations, action plan, and interview prep.

    Returns dict with keys:
        - final_recommendation_summary (str)
        - top_recommendation (str) — single best career
        - key_strengths (list[str])
        - interview_preparation (dict)  — {topics, resources, tips}
        - action_plan_30_days (list[str])
        - action_plan_90_days (list[str])
        - encouragement_note (str)
        - disclaimer (str)
    """

    # Pull interview topics from KB for top career
    interview_kb = load_interview_topics()
    top_career_id = (career_matches.get("top_careers") or [{}])[0].get("career_id", "")
    interview_data = interview_kb.get("interview_topics", {}).get(top_career_id, {})
    interview_topics_str = json.dumps(interview_data, indent=2) if interview_data else "{}"

    # Build a concise synthesis context
    top_careers_str = json.dumps(career_matches.get("top_careers", []), indent=2)
    gaps_str        = json.dumps(skill_gap_analysis.get("prioritised_gaps", [])[:6], indent=2)
    phases_str      = json.dumps(roadmap.get("phases", []), indent=2)
    strengths_str   = json.dumps(skill_gap_analysis.get("student_strengths", []), indent=2)
    market_str      = career_matches.get("top_careers", [{}])[0].get("career_title", "")
    mkt_data        = market_analysis.get("career_market_data", [{}])[0] if market_analysis.get("career_market_data") else {}

    prompt = f"""<|system|>
You are the Final Career Advisor Agent — the most senior agent in the career counseling system.
You synthesise all previous analysis into a final, empathetic, and actionable career counseling report.
Base your recommendations ONLY on the data provided. Do not guarantee employment, salaries, or outcomes.
Return ONLY valid JSON — no markdown fences, no extra text.
<|user|>
## Student Profile
- Name: {student_profile.get("name", "Student")}
- Degree: {student_profile.get("degree","")} — {student_profile.get("specialization","")}
- CGPA: {student_profile.get("cgpa","")}
- Career Goal: {student_profile.get("career_goal","")}

## Top Career Matches
{top_careers_str}

## Student Strengths
{strengths_str}

## Top Priority Skill Gaps
{gaps_str}

## Learning Roadmap Phases
{phases_str}

## Market Context for Primary Career ({market_str})
Demand: {mkt_data.get("demand_trend","N/A")} | Growth: {mkt_data.get("yoy_growth","N/A")}% YoY
Remote: {mkt_data.get("remote_opportunity","N/A")}

## Interview Preparation Data (from Knowledge Base)
{interview_topics_str}

## Task
Synthesise everything and return a JSON object with these exact keys:
- "final_recommendation_summary": 3-4 paragraph personalised counseling summary
- "top_recommendation": single best career title for this student
- "key_strengths": list of 5-7 genuine student strengths
- "interview_preparation": object with:
    - "core_topics": list of top 5 interview topics for the primary career
    - "preparation_resources": list of 3-5 recommended preparation resources
    - "tips": list of 3-5 specific interview preparation tips for this student
- "action_plan_30_days": list of 5-7 concrete actions for the next 30 days
- "action_plan_90_days": list of 5-7 concrete actions for the next 90 days
- "encouragement_note": 1-2 uplifting, realistic sentences for the student
- "disclaimer": standard disclaimer — do not guarantee employment or salary outcomes

Return ONLY the JSON object.
<|assistant|>
"""

    model = _build_final_advisor_model()
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
            "final_recommendation_summary": response.strip(),
            "top_recommendation": (career_matches.get("top_careers") or [{"career_title": ""}])[0].get("career_title", ""),
            "key_strengths": skill_gap_analysis.get("student_strengths", []),
            "interview_preparation": {
                "core_topics": interview_data.get("core_topics", []),
                "preparation_resources": interview_data.get("preparation_resources", []),
                "tips": [],
            },
            "action_plan_30_days": roadmap.get("immediate_actions", []),
            "action_plan_90_days": [],
            "encouragement_note": "Keep learning and stay consistent — your goals are achievable.",
            "disclaimer": "This report is for guidance only. Career outcomes depend on many factors and cannot be guaranteed.",
        }

    return result


# ── Orchestration Pipeline ─────────────────────────────────────────────────

class AgentPipeline:
    """
    Orchestrates all career counseling agents in the defined workflow order.

    This class represents the IBM watsonx Orchestrate workflow.
    In a watsonx Orchestrate deployment:
      - Each `_run_*` method maps to an Orchestrate skill
      - The `run()` method maps to an Orchestrate flow/automation
      - Progress callbacks map to Orchestrate step events
    """

    def __init__(self, on_progress: Callable[[str, str], None] | None = None):
        """
        Args:
            on_progress: Optional callback fn(agent_name, status_message)
                         called after each agent completes.
        """
        self.on_progress = on_progress or (lambda name, msg: None)

    def _step(self, name: str, fn: Callable, *args, **kwargs) -> Any:
        """Run a single agent step with progress reporting and basic error handling."""
        self.on_progress(name, "running")
        try:
            result = fn(*args, **kwargs)
            self.on_progress(name, "complete")
            return result
        except Exception as exc:
            self.on_progress(name, f"error: {exc}")
            raise

    def run(self, student_profile: dict) -> dict:
        """
        Execute the full career counseling pipeline.

        Returns the complete report dict:
            {
                student_profile,
                academic_analysis,
                interest_analysis,
                skill_analysis,
                career_matches,
                skill_gap_analysis,
                roadmap,
                market_analysis,
                final_advice,
                pipeline_metadata,
            }
        """
        start_time = time.time()

        # ── Step 1: Academic Performance Agent ───────────────────────────
        academic_analysis = self._step(
            "Academic Performance Agent",
            analyze_academic_performance,
            student_profile,
        )

        # ── Step 2: Interest & Preference Agent ──────────────────────────
        interest_analysis = self._step(
            "Interest & Preference Agent",
            analyze_interests,
            student_profile,
        )

        # ── Step 3: Skill Assessment Agent ───────────────────────────────
        skill_analysis = self._step(
            "Skill Assessment Agent",
            assess_skills,
            student_profile,
        )

        # ── Step 4: Career Matching Agent ────────────────────────────────
        career_matches = self._step(
            "Career Matching Agent",
            match_careers,
            student_profile,
            academic_analysis,
            interest_analysis,
            skill_analysis,
        )

        # ── Step 5: Skill Gap Analysis Agent ─────────────────────────────
        skill_gap_analysis = self._step(
            "Skill Gap Analysis Agent",
            analyze_skill_gaps,
            student_profile,
            skill_analysis,
            career_matches,
        )

        # ── Step 6: Learning Roadmap Agent ────────────────────────────────
        roadmap = self._step(
            "Learning Roadmap Agent",
            generate_roadmap,
            student_profile,
            skill_gap_analysis,
            career_matches,
        )

        # ── Step 7: Labor Market Trend Agent ──────────────────────────────
        market_analysis = self._step(
            "Labor Market Trend Agent",
            analyze_market_trends,
            career_matches,
            student_profile,
        )

        # ── Step 8: Final Career Advisor Agent ────────────────────────────
        final_advice = self._step(
            "Final Career Advisor Agent",
            _run_final_advisor,
            student_profile,
            academic_analysis,
            interest_analysis,
            skill_analysis,
            career_matches,
            skill_gap_analysis,
            roadmap,
            market_analysis,
        )

        elapsed = round(time.time() - start_time, 1)

        return {
            "student_profile":    student_profile,
            "academic_analysis":  academic_analysis,
            "interest_analysis":  interest_analysis,
            "skill_analysis":     skill_analysis,
            "career_matches":     career_matches,
            "skill_gap_analysis": skill_gap_analysis,
            "roadmap":            roadmap,
            "market_analysis":    market_analysis,
            "final_advice":       final_advice,
            "pipeline_metadata": {
                "agents_run": 8,
                "model_used": config.MODEL_ID,
                "elapsed_seconds": elapsed,
                "kb_files_used": [
                    "career_paths.json",
                    "career_requirements.json",
                    "skills_database.json",
                    "learning_resources.json",
                    "certifications.json",
                    "interview_topics.json",
                    "market_trends.json",
                ],
            },
        }


# ── Public entry point ─────────────────────────────────────────────────────

def run_career_counseling(
    student_profile: dict,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """
    Public API for the career counseling system.

    Args:
        student_profile: dict of student data (see app.py for schema)
        on_progress: optional callback fn(agent_name, status)

    Returns:
        Complete career counseling report dict
    """
    if not config.validate_credentials():
        raise ValueError(
            "IBM watsonx credentials are not configured. "
            "Please set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL in your .env file."
        )

    pipeline = AgentPipeline(on_progress=on_progress)
    return pipeline.run(student_profile)
