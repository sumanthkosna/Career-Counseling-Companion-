"""
knowledge_base_loader.py — RAG Knowledge Base Loader
──────────────────────────────────────────────────────
Central module for loading JSON knowledge-base files.
All agents import from here so KB file paths are managed in one place.
"""

import json
import os
from typing import Any

import config


def _load_json(filepath: str) -> Any:
    """Load and return a JSON file. Returns an empty dict on error."""
    if not os.path.exists(filepath):
        print(f"[KB Warning] Knowledge base file not found: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_career_paths() -> dict:
    return _load_json(config.CAREER_PATHS_FILE)


def load_career_requirements() -> dict:
    return _load_json(config.CAREER_REQUIREMENTS_FILE)


def load_skills_database() -> dict:
    return _load_json(config.SKILLS_DATABASE_FILE)


def load_learning_resources() -> dict:
    return _load_json(config.LEARNING_RESOURCES_FILE)


def load_certifications() -> dict:
    return _load_json(config.CERTIFICATIONS_FILE)


def load_interview_topics() -> dict:
    return _load_json(config.INTERVIEW_TOPICS_FILE)


def load_market_trends() -> dict:
    return _load_json(config.MARKET_TRENDS_FILE)
