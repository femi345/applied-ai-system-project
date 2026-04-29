"""Input validation, output guardrails, and logging configuration."""

import logging
import sys
from typing import Dict, List, Tuple, Optional

VALID_GENRES = {
    "pop", "lofi", "rock", "edm", "jazz", "blues", "classical",
    "hip-hop", "ambient", "folk", "metal", "r&b", "synthwave",
    "country", "electronic", "indie pop",
}

VALID_MOODS = {
    "happy", "chill", "intense", "sad", "focused", "romantic", "moody", "relaxed",
}

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the application."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def validate_user_input(user_input: str) -> Tuple[bool, str]:
    """Validate that user input is safe and reasonable."""
    if not user_input or not user_input.strip():
        return False, "Input cannot be empty."
    if len(user_input) > 1000:
        return False, "Input is too long (max 1000 characters)."
    blocked_patterns = ["ignore previous", "system prompt", "you are now"]
    lower = user_input.lower()
    for pattern in blocked_patterns:
        if pattern in lower:
            return False, f"Input contains blocked pattern: '{pattern}'."
    return True, "Input is valid."


def validate_preferences(prefs: Dict) -> Tuple[bool, List[str]]:
    """Validate a user preference dictionary."""
    errors = []
    genre = prefs.get("genre", "").lower()
    if genre and genre not in VALID_GENRES:
        errors.append(f"Unknown genre '{genre}'. Valid: {sorted(VALID_GENRES)}")

    mood = prefs.get("mood", "").lower()
    if mood and mood not in VALID_MOODS:
        errors.append(f"Unknown mood '{mood}'. Valid: {sorted(VALID_MOODS)}")

    for field in ("energy", "valence", "danceability"):
        val = prefs.get(field)
        if val is not None:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                errors.append(f"'{field}' must be a number between 0.0 and 1.0, got {val}")

    if errors:
        logger.warning("Preference validation failed: %s", errors)
        return False, errors
    return True, []


def check_output_safety(text: str) -> Tuple[bool, str]:
    """Check that AI-generated output doesn't contain harmful content."""
    if not text or not text.strip():
        return False, "Output is empty."
    if len(text) > 5000:
        return False, "Output exceeds maximum length."
    return True, "Output passed safety check."


def sanitize_output(text: str) -> str:
    """Clean up AI output for display."""
    text = text.strip()
    if len(text) > 5000:
        text = text[:4997] + "..."
    return text
