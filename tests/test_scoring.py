"""Tests for the scoring and recommendation functions."""

from src.recommender import load_songs, score_song, recommend_songs


def test_load_songs_returns_list():
    songs = load_songs("data/songs.csv")
    assert isinstance(songs, list)
    assert len(songs) == 20


def test_load_songs_has_required_fields():
    songs = load_songs("data/songs.csv")
    required = {"id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
                 "valence", "danceability", "acousticness"}
    for song in songs:
        assert required.issubset(song.keys())


def test_score_song_genre_match():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.8}
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    score, reasons = score_song(prefs, song)
    assert score >= 3.0
    assert any("genre" in r for r in reasons)


def test_score_song_no_match():
    song = {"genre": "metal", "mood": "intense", "energy": 0.9, "valence": 0.3, "danceability": 0.5}
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.3}
    score, reasons = score_song(prefs, song)
    assert score < 2.0


def test_score_returns_reasons():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.8}
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    score, reasons = score_song(prefs, song)
    assert isinstance(reasons, list)
    assert len(reasons) >= 1


def test_recommend_songs_returns_top_k():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    recs = recommend_songs(prefs, songs, k=3)
    assert len(recs) == 3


def test_recommend_songs_sorted_descending():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "rock", "mood": "intense", "energy": 0.9}
    recs = recommend_songs(prefs, songs, k=5)
    scores = [r[1] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_top_result_matches_genre():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "edm", "mood": "intense", "energy": 0.95}
    recs = recommend_songs(prefs, songs, k=3)
    assert recs[0][0]["genre"] == "edm"


def test_score_with_valence_and_danceability():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.9}
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "danceability": 0.9}
    score, reasons = score_song(prefs, song)
    assert score >= 4.5
    assert any("valence" in r for r in reasons)
    assert any("danceability" in r for r in reasons)
