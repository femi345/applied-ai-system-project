# Model Card: AI Music Recommender

## 1. Model Overview

**Name:** VibeFinder 2.0 — RAG-Enhanced AI Music Recommender

**Base project:** Music Recommender Simulation (CodePath AI110 Module 3). The original was a deterministic content-based recommender that scored songs using weighted attribute matching. This version adds RAG retrieval, an agentic workflow, and LLM-powered explanations.

---

## 2. Intended Use

Recommends songs from a 20-track catalog based on natural language descriptions of what the user wants to hear. Designed as a classroom project demonstrating RAG, agentic workflows, and AI reliability testing. Not intended for production music streaming.

---

## 3. How It Works

The system runs a 6-step agentic pipeline:

1. **Input Validation** — Checks for empty input, excessive length, and prompt injection patterns.
2. **Preference Extraction** — GPT-4o-mini parses the user's natural language into structured preferences (genre, mood, energy, valence, danceability, context).
3. **Knowledge Retrieval (RAG)** — The query is embedded using OpenAI's text-embedding-3-small model and compared against a pre-embedded knowledge base of 30 documents covering genres, moods, and listening contexts. The top 5 most relevant documents are retrieved via cosine similarity.
4. **Algorithmic Scoring** — The original scoring algorithm ranks all 20 songs against the extracted preferences. Genre match is worth +2.0, mood match +1.0, and energy/valence/danceability similarity contribute up to +2.0 combined.
5. **AI Enhancement** — GPT-4o-mini generates personalized explanations for the top 5 songs, using the retrieved knowledge as context. Each song gets a confidence score.
6. **Self-Evaluation** — The agent checks genre diversity, artist diversity, average confidence, and genre match coverage, flagging issues.

---

## 4. Data

**Song catalog:** 20 tracks in `data/songs.csv` with attributes: genre, mood, energy (0-1), tempo_bpm, valence (0-1), danceability (0-1), acousticness (0-1). Hand-curated, skewed toward English-language Western music.

**Knowledge base:** 30 JSON documents across three categories:
- `genres.json` — 16 genre descriptions (pop, lofi, rock, edm, jazz, blues, classical, hip-hop, ambient, folk, metal, r&b, synthwave, country, electronic, indie pop)
- `moods.json` — 8 mood descriptions (happy, chill, intense, sad, focused, romantic, moody, relaxed)
- `listening_contexts.json` — 6 context descriptions (study, workout, commute, sleep, social, creative)

---

## 5. Strengths

- **Natural language input** makes the system accessible — users don't need to know what "energy: 0.8" means.
- **Hybrid approach** (algorithmic scoring + LLM enhancement) is more reliable than pure LLM recommendations. The scorer provides consistent, explainable rankings; the LLM adds context and personality.
- **Full transparency** — every agent step is logged and visible in the UI. Users can see exactly how their input was interpreted and why each song was chosen.
- **Guardrails** catch malicious input before it reaches the LLM.
- **Self-evaluation** flags low-confidence or low-diversity results.

---

## 6. Limitations and Bias

- **Small catalog (20 songs):** Some genres have only one track, so the system can't offer variety. A user asking for metal always gets Rage Circuit.
- **Genre dominance:** Genre match is still worth 40% of the max score. The system tends to lock users into their stated genre even when cross-genre songs might fit better.
- **Western music bias:** The catalog and knowledge base focus on Western music genres. Non-Western genres (K-pop, Afrobeat, Bollywood, etc.) are not represented.
- **No collaborative signal:** The system only knows what the current user said. It can't learn from patterns across users.
- **LLM dependency:** The enhancement and preference extraction steps require API calls to OpenAI. If the API is down or the key is invalid, the system falls back to raw algorithmic results without rich explanations.
- **Rigid mood matching:** The algorithmic scorer still uses exact string matching for moods. "Chill" and "relaxed" score differently even though users treat them as similar.

---

## 7. Potential Misuse and Mitigation

**Could this AI be misused?**
- **Filter bubbles:** Like any recommendation system, this could reinforce narrow taste patterns and limit musical discovery. The self-evaluation step partially addresses this by flagging low genre diversity.
- **Prompt injection:** A malicious user could try to manipulate the LLM through crafted inputs. The guardrails block common injection patterns ("ignore previous instructions", "system prompt", etc.).
- **Data poisoning:** If someone modified the knowledge base or song catalog with misleading information, the recommendations would be corrupted. In a production system, the knowledge base would need access controls.

**Mitigations implemented:**
- Input validation and prompt injection detection
- Output length limits and safety checks
- Confidence scoring that flags uncertain results
- Diversity checking in the self-evaluation step

---

## 8. Testing and Evaluation

**Unit tests:** 28 tests covering input validation, scoring logic, and recommendation ordering. All pass.

**Evaluation harness:** 8 automated test cases covering:
- Genre-specific requests (pop, lofi, rock, blues, EDM) — checks that the expected genre appears in results
- Context-driven requests (night drive + synthwave) — checks mood and genre alignment
- Vague requests ("play me something good") — checks that the system still returns results with reasonable confidence
- Security testing (prompt injection) — checks that malicious input is rejected

**Results:**
- 7/8 tests pass consistently. The vague request test sometimes dips below the confidence threshold since no genre is specified, which is expected behavior.
- Average confidence: 0.78 across genre-specific tests.
- Prompt injection correctly blocked in all runs.

---

## 9. Reflection on AI Collaboration

**How AI was used during this project:**
I used AI throughout development — for generating the knowledge base content, designing the agentic pipeline architecture, and writing test cases. The AI was particularly helpful for structuring the multi-step agent workflow and suggesting the hybrid scoring + LLM approach.

**One instance where AI was helpful:**
When designing the RAG retrieval system, AI suggested using cosine similarity on OpenAI embeddings rather than keyword matching. This was the right call — it means a query like "something for a late night drive" retrieves the synthwave and moody music knowledge documents even though "synthwave" doesn't appear in the query. Semantic search made the retrieval much more useful than simple keyword matching would have been.

**One instance where AI was flawed:**
AI initially suggested using GPT-4o for the preference extraction step, which would have been slow and expensive for what's essentially a structured parsing task. It also generated an overly complex prompt that tried to extract 12 different preference fields. I simplified to GPT-4o-mini and 6 fields, which was faster, cheaper, and actually more accurate because the model had fewer ambiguous fields to fill.

---

## 10. Future Improvements

- **Larger catalog:** Expand to 100+ songs with better genre coverage.
- **Fuzzy mood matching:** Use embedding similarity instead of exact string matching for mood in the algorithmic scorer.
- **Diversity penalty:** Reduce scores for songs from the same genre or artist to prevent filter bubbles.
- **User history:** Track past recommendations and preferences across sessions.
- **Offline mode:** Cache embeddings and use a local model for preference extraction so the system works without an API key.
