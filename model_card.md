# Model Card: AI Music Recommender

## 1. Model Overview

VibeFinder 2.0

Built on top of the Music Recommender Simulation from CodePath AI110 Module 3. The original was a deterministic scorer that matched songs to taste profiles. This version adds RAG retrieval, an agentic workflow, and LLM-generated explanations.

## 2. Intended Use

Recommends songs from a 20-track catalog based on natural language descriptions. This is a classroom project showing how RAG, agents, and reliability testing work together. Not meant for real music streaming.

## 3. How It Works

The system runs a 6-step pipeline:

1. **Input Validation** — blocks empty input, excessive length, and prompt injection patterns.
2. **Preference Extraction** — GPT-4o-mini parses natural language into structured preferences (genre, mood, energy, valence, danceability, context).
3. **Knowledge Retrieval (RAG)** — the query gets embedded with text-embedding-3-small and compared against 30 pre-embedded knowledge base documents. Top 5 by cosine similarity are pulled.
4. **Algorithmic Scoring** — the original scoring algorithm ranks all 20 songs. Genre match = +2.0, mood match = +1.0, energy/valence/danceability similarity = up to +2.0 combined.
5. **AI Enhancement** — GPT-4o-mini writes personalized explanations for the top 5 songs using the retrieved knowledge as context. Each song gets a confidence score.
6. **Self-Evaluation** — checks genre diversity, artist diversity, average confidence, and genre match coverage. Flags problems.

## 4. Data

Song catalog: 20 tracks in `data/songs.csv` with genre, mood, energy (0-1), tempo_bpm, valence (0-1), danceability (0-1), acousticness (0-1). Hand-curated, skewed toward English-language Western music.

Knowledge base: 30 JSON documents in three files:
- `genres.json` — 16 genre descriptions
- `moods.json` — 8 mood descriptions
- `listening_contexts.json` — 6 context descriptions (study, workout, commute, sleep, social, creative)

## 5. What Works Well

Natural language input means users can say "something chill for studying" instead of figuring out what energy level to type. The scoring algorithm gives consistent, explainable rankings while the LLM writes explanations that reference actual music knowledge from the RAG retrieval. Every agent step is logged and visible in the UI, so you can see exactly how your input was interpreted. Guardrails block malicious input before it hits the LLM. Self-evaluation catches low-confidence or low-diversity results.

## 6. Limitations and Bias

20 songs is tiny. Some genres (metal, blues) have one track each, so you always get the same song. Genre match is worth 40% of the max score, which means the system locks you into your stated genre even when a cross-genre song might fit better. The catalog and knowledge base only cover Western music — no K-pop, Afrobeat, Bollywood, etc. The system can't learn from other users since there's no collaborative filtering. If the OpenAI API is down, you lose the natural language parsing and explanations (falls back to raw algorithm output). Mood matching is still exact strings — "chill" and "relaxed" get zero overlap.

## 7. Misuse and Mitigation

This could create filter bubbles like any recommender — the self-evaluation step partially addresses that by flagging when all results are the same genre. Prompt injection is possible but the guardrails block common patterns ("ignore previous instructions", etc.). If someone tampered with the knowledge base files, the recommendations would be wrong. A production system would need access controls on those files.

What's in place: input validation, prompt injection detection, output length limits, confidence scoring, diversity checks.

## 8. Testing

28 unit tests pass — input validation, scoring logic, recommendation ordering.

8 evaluation harness test cases:
- Genre-specific (pop, lofi, rock, blues, EDM) — checks expected genre appears in results
- Context-driven (night drive + synthwave) — checks mood and genre alignment
- Vague input ("play me something good") — checks system still returns reasonable results
- Prompt injection — checks it gets blocked

7/8 pass consistently. The vague request test sometimes falls below the confidence threshold because no genre was specified. Average confidence: 0.78 across genre-specific tests. Prompt injection blocked every time.

## 9. AI Collaboration

I used AI throughout — generating knowledge base content, designing the agent pipeline, and writing test cases.

**Where AI helped:** It suggested using cosine similarity on OpenAI embeddings for the RAG retrieval instead of keyword matching. That was the right approach. A query like "something for a late night drive" pulls up synthwave and moody music docs even though those words aren't in the query. Keyword matching wouldn't have caught that.

**Where AI was wrong:** It initially said to use GPT-4o for preference extraction and wrote an overly complex prompt that tried to parse 12 different fields. That was slow, expensive, and less accurate. I cut it to GPT-4o-mini and 6 fields — faster responses, lower cost, and the model actually performed better with fewer fields to fill.

## 10. Future Work

- Expand to 100+ songs with better genre coverage
- Use embedding similarity instead of exact string matching for mood
- Add a diversity penalty so the same genre doesn't dominate all 5 slots
- Track user preferences across sessions
- Cache embeddings and use a local model so it works without an API key
