"""Core recommendation engine for the Music Recommender Simulation."""

import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by relevance to the user profile."""
        scored = []
        for song in self.songs:
            score = self._score(user, song)
            scored.append((score, song))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [song for _, song in scored[:k]]

    def _score(self, user: UserProfile, song: Song) -> float:
        """Compute a numeric relevance score for a single song."""
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 2.0
        if song.mood == user.favorite_mood:
            score += 1.0
        energy_similarity = 1.0 - abs(song.energy - user.target_energy)
        score += energy_similarity
        if user.likes_acoustic:
            score += song.acousticness
        else:
            score += 1.0 - song.acousticness
        return score

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre match: {song.genre} (+2.0)")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood match: {song.mood} (+1.0)")
        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity: {energy_sim:.2f} (song={song.energy}, target={user.target_energy})")
        if user.likes_acoustic:
            reasons.append(f"acousticness bonus: +{song.acousticness:.2f}")
        else:
            reasons.append(f"low-acousticness bonus: +{1.0 - song.acousticness:.2f}")
        return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Functional API (used by src/main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a single song against user preferences.

    Scoring algorithm:
        +2.0  exact genre match
        +1.0  exact mood match
        +[0-1] energy similarity (1 - |song_energy - target_energy|)
        +[0-1] valence similarity (1 - |song_valence - target_valence|), weighted x0.5
        +[0-1] danceability similarity, weighted x0.5
    """
    score = 0.0
    reasons = []

    # Genre match
    if song["genre"].lower() == user_prefs.get("genre", "").lower():
        score += 2.0
        reasons.append("genre match (+2.0)")

    # Mood match
    if song["mood"].lower() == user_prefs.get("mood", "").lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Energy similarity
    target_energy = user_prefs.get("energy", 0.5)
    energy_sim = 1.0 - abs(song["energy"] - target_energy)
    score += energy_sim
    reasons.append(f"energy similarity (+{energy_sim:.2f})")

    # Valence similarity (half weight)
    if "valence" in user_prefs:
        valence_sim = 1.0 - abs(song["valence"] - user_prefs["valence"])
        bonus = valence_sim * 0.5
        score += bonus
        reasons.append(f"valence similarity (+{bonus:.2f})")

    # Danceability similarity (half weight)
    if "danceability" in user_prefs:
        dance_sim = 1.0 - abs(song["danceability"] - user_prefs["danceability"])
        bonus = dance_sim * 0.5
        score += bonus
        reasons.append(f"danceability similarity (+{bonus:.2f})")

    return (score, reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the top-k songs ranked by score, with explanations."""
    scored = []
    for song in songs:
        total, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, total, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
