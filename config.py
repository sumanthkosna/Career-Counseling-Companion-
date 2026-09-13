"""
config.py — Central configuration for the Career Counseling Companion.
All IBM watsonx credentials and model settings are loaded from environment variables.
Never hard-code credentials here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai credentials ─────────────────────────────────────────────
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# ── IBM Granite model ──────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")

# ── Generation parameters ──────────────────────────────────────────────────
DECODING_METHOD    = "greedy"
MAX_NEW_TOKENS     = 2048
MIN_NEW_TOKENS     = 50
TEMPERATURE        = 0.7
TOP_P              = 0.95
REPETITION_PENALTY = 1.1

# ── RAG / Knowledge-base paths ─────────────────────────────────────────────
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

CAREER_PATHS_FILE        = os.path.join(KB_DIR, "career_paths.json")
CAREER_REQUIREMENTS_FILE = os.path.join(KB_DIR, "career_requirements.json")
SKILLS_DATABASE_FILE     = os.path.join(KB_DIR, "skills_database.json")
LEARNING_RESOURCES_FILE  = os.path.join(KB_DIR, "learning_resources.json")
CERTIFICATIONS_FILE      = os.path.join(KB_DIR, "certifications.json")
INTERVIEW_TOPICS_FILE    = os.path.join(KB_DIR, "interview_topics.json")
MARKET_TRENDS_FILE       = os.path.join(KB_DIR, "market_trends.json")

# ── Validation helper ──────────────────────────────────────────────────────
def validate_credentials() -> bool:
    """Return True only when all mandatory IBM watsonx credentials are present."""
    return all([WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL])
