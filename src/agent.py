"""Agentic workflow orchestrator for the AI music recommender.

The agent follows a multi-step reasoning pipeline:
  1. PARSE   — extract structured preferences from natural language input
  2. RETRIEVE — pull relevant music knowledge from the RAG knowledge base
  3. SCORE   — run the scoring algorithm on the song catalog
  4. ENHANCE — use the LLM + retrieved context to generate rich explanations
  5. EVALUATE — self-check the recommendations for quality and diversity
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from src.recommender import load_songs, score_song, recommend_songs
from src.knowledge_base import load_knowledge_base, build_kb_index, retrieve
from src.guardrails import (
    validate_user_input,
    validate_preferences,
    check_output_safety,
    sanitize_output,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """A single observable step in the agent's reasoning chain."""
    name: str
    status: str = "pending"
    input_data: Any = None
    output_data: Any = None
    duration_ms: float = 0
    details: str = ""


@dataclass
class AgentResult:
    """Complete result from an agent run."""
    success: bool
    recommendations: List[Dict] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    confidence: float = 0.0
    error: str = ""


class MusicRecommenderAgent:
    """Multi-step reasoning agent for AI-powered music recommendations."""

    def __init__(self, client, songs_path: str = "data/songs.csv"):
        self.client = client
        self.songs = load_songs(songs_path)
        self.kb_docs = load_knowledge_base()
        self.kb_index = build_kb_index(self.kb_docs, client)
        logger.info("Agent initialized with %d songs and %d KB docs",
                     len(self.songs), len(self.kb_docs))

    def run(self, user_input: str, context: Optional[str] = None) -> AgentResult:
        """Execute the full agent pipeline on a user query."""
        steps = []
        logger.info("Agent run started for input: '%s'", user_input[:80])

        # Step 1: Validate input
        step = AgentStep(name="Input Validation", input_data=user_input)
        t0 = time.time()
        valid, msg = validate_user_input(user_input)
        if not valid:
            step.status = "failed"
            step.details = msg
            step.duration_ms = (time.time() - t0) * 1000
            steps.append(step)
            return AgentResult(success=False, steps=steps, error=msg)
        step.status = "passed"
        step.details = msg
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        # Step 2: Parse preferences from natural language
        step = AgentStep(name="Preference Extraction", input_data=user_input)
        t0 = time.time()
        prefs = self._parse_preferences(user_input, context)
        step.output_data = prefs
        step.status = "completed"
        step.details = f"Extracted: {json.dumps(prefs, indent=2)}"
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        # Step 3: Retrieve relevant knowledge
        step = AgentStep(name="Knowledge Retrieval (RAG)")
        t0 = time.time()
        query = self._build_retrieval_query(prefs)
        step.input_data = query
        retrieved = retrieve(query, self.kb_index, self.client, top_k=5)
        step.output_data = [
            {"title": r["title"], "similarity": round(r["similarity"], 3)}
            for r in retrieved
        ]
        step.status = "completed"
        step.details = f"Retrieved {len(retrieved)} relevant documents"
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        # Step 4: Score songs using the algorithmic recommender
        step = AgentStep(name="Algorithmic Scoring", input_data=prefs)
        t0 = time.time()
        scored = recommend_songs(prefs, self.songs, k=10)
        step.output_data = [
            {"title": s[0]["title"], "score": round(s[1], 2)} for s in scored[:5]
        ]
        step.status = "completed"
        step.details = f"Scored {len(self.songs)} songs, top score: {scored[0][1]:.2f}"
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        # Step 5: Enhance recommendations with LLM + RAG context
        step = AgentStep(name="AI Enhancement (LLM + RAG)")
        t0 = time.time()
        enhanced = self._enhance_recommendations(prefs, scored[:5], retrieved, user_input, context)
        step.output_data = "Generated enhanced explanations"
        step.status = "completed"
        step.details = f"Enhanced {len(enhanced)} recommendations with context"
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        # Step 6: Self-evaluate quality
        step = AgentStep(name="Quality Self-Evaluation")
        t0 = time.time()
        evaluation = self._evaluate_recommendations(enhanced, prefs, user_input)
        step.output_data = evaluation
        step.status = "completed"
        step.details = f"Confidence: {evaluation['confidence']:.0%}, Issues: {len(evaluation['issues'])}"
        step.duration_ms = (time.time() - t0) * 1000
        steps.append(step)

        total_ms = sum(s.duration_ms for s in steps)
        logger.info("Agent run completed in %.0fms, confidence: %.0f%%",
                     total_ms, evaluation["confidence"] * 100)

        return AgentResult(
            success=True,
            recommendations=enhanced,
            steps=steps,
            confidence=evaluation["confidence"],
        )

    def _parse_preferences(self, user_input: str, context: Optional[str] = None) -> Dict:
        """Use the LLM to extract structured preferences from natural language."""
        system_msg = (
            "You are a music preference parser. Extract structured preferences from the user's "
            "message. Return ONLY valid JSON with these fields (omit any that aren't mentioned):\n"
            '{"genre": "string", "mood": "string", "energy": 0.0-1.0, '
            '"valence": 0.0-1.0, "danceability": 0.0-1.0, "context": "string"}\n\n'
            "Valid genres: pop, lofi, rock, edm, jazz, blues, classical, hip-hop, ambient, "
            "folk, metal, r&b, synthwave, country, electronic, indie pop\n"
            "Valid moods: happy, chill, intense, sad, focused, romantic, moody, relaxed\n"
            "For energy/valence/danceability, infer from context clues (e.g., 'workout' = high energy).\n"
            "Return ONLY the JSON object, no markdown or explanation."
        )
        user_msg = user_input
        if context:
            user_msg += f"\nAdditional context: {context}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            prefs = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM preference output: %s", raw)
            prefs = {"genre": "pop", "mood": "happy", "energy": 0.5}
        return prefs

    def _build_retrieval_query(self, prefs: Dict) -> str:
        """Build a search query from extracted preferences."""
        parts = []
        if "genre" in prefs:
            parts.append(f"{prefs['genre']} music")
        if "mood" in prefs:
            parts.append(f"{prefs['mood']} mood")
        if "context" in prefs:
            parts.append(prefs["context"])
        if "energy" in prefs:
            level = "high" if prefs["energy"] > 0.6 else "low"
            parts.append(f"{level} energy")
        return " ".join(parts) if parts else "general music recommendation"

    def _enhance_recommendations(
        self,
        prefs: Dict,
        top_songs: list,
        retrieved_docs: List[Dict],
        user_input: str,
        context: Optional[str],
    ) -> List[Dict]:
        """Use the LLM + retrieved context to generate rich recommendation explanations."""
        kb_context = "\n\n".join(
            f"[{doc['title']}]: {doc['content']}" for doc in retrieved_docs[:3]
        )

        songs_text = "\n".join(
            f"  {i+1}. \"{s[0]['title']}\" by {s[0]['artist']} "
            f"(genre={s[0]['genre']}, mood={s[0]['mood']}, energy={s[0]['energy']}, "
            f"score={s[1]:.2f}) — scoring reasons: {s[2]}"
            for i, s in enumerate(top_songs)
        )

        system_msg = (
            "You are a knowledgeable music curator. Given a user's request, music knowledge context, "
            "and algorithmically scored songs, write a personalized explanation for each recommendation.\n\n"
            "For each song, write 2-3 sentences explaining WHY this song fits the user's taste, "
            "drawing on the music knowledge provided. Be specific about musical qualities.\n\n"
            "Also assign a confidence score (0.0-1.0) for how well each song matches the request.\n\n"
            "Return ONLY valid JSON as a list of objects:\n"
            '[{"title": "...", "artist": "...", "explanation": "...", "confidence": 0.0-1.0}]\n'
            "No markdown, no extra text — just the JSON array."
        )

        user_msg = (
            f"User request: {user_input}\n\n"
            f"Music knowledge context:\n{kb_context}\n\n"
            f"Top scored songs:\n{songs_text}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            enhanced = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse enhanced recommendations: %s", raw[:200])
            enhanced = [
                {
                    "title": s[0]["title"],
                    "artist": s[0]["artist"],
                    "explanation": s[2],
                    "confidence": round(s[1] / 5.0, 2),
                }
                for s in top_songs
            ]

        for i, item in enumerate(enhanced):
            if i < len(top_songs):
                song_data = top_songs[i][0]
                item["score"] = round(top_songs[i][1], 2)
                item["genre"] = song_data["genre"]
                item["mood"] = song_data["mood"]
                item["energy"] = song_data["energy"]
                safe, msg = check_output_safety(item.get("explanation", ""))
                if not safe:
                    item["explanation"] = f"[Filtered: {msg}]"
                else:
                    item["explanation"] = sanitize_output(item["explanation"])

        return enhanced

    def _evaluate_recommendations(
        self, recommendations: List[Dict], prefs: Dict, user_input: str
    ) -> Dict:
        """Self-evaluate the quality and diversity of recommendations."""
        issues = []
        total_confidence = 0.0

        if len(recommendations) == 0:
            return {"confidence": 0.0, "issues": ["No recommendations generated"], "summary": "Failed"}

        genres_seen = set()
        artists_seen = set()
        for rec in recommendations:
            total_confidence += rec.get("confidence", 0.5)
            genres_seen.add(rec.get("genre", "unknown"))
            artists_seen.add(rec.get("artist", "unknown"))

        avg_confidence = total_confidence / len(recommendations)

        if len(genres_seen) == 1:
            issues.append("Low genre diversity — all recommendations are the same genre")
        if len(artists_seen) < len(recommendations):
            issues.append("Artist repetition detected")
        if avg_confidence < 0.5:
            issues.append(f"Low average confidence ({avg_confidence:.2f})")

        target_genre = prefs.get("genre", "")
        genre_matches = sum(1 for r in recommendations if r.get("genre") == target_genre)
        if target_genre and genre_matches == 0:
            issues.append(f"No songs match requested genre '{target_genre}'")

        summary = "Good" if avg_confidence >= 0.7 and len(issues) == 0 else "Acceptable" if avg_confidence >= 0.5 else "Needs improvement"

        return {
            "confidence": round(avg_confidence, 3),
            "issues": issues,
            "genre_diversity": len(genres_seen),
            "artist_diversity": len(artists_seen),
            "summary": summary,
        }
