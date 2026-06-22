# Changelog

All notable changes to PromptStrike are documented here.

---

## [0.1.0] — 2026-06-22

### Added

#### Core algorithms
- **PAIR** (Chao et al. 2023, arXiv:2310.08419) — adaptive single-chain jailbreak loop; attacker LLM refines prompts based on judge score feedback; converges in < 20 queries
- **TAP** (Mehrotra et al. 2023, arXiv:2312.02119) — tree search extension of PAIR; `branching_factor × depth` nodes with score-based pruning; finds jailbreaks that single-chain refinement misses
- **JailbreakBench judge** (Chao et al. 2024, arXiv:2404.01318) — 1–10 scoring rubric; `[[N]]` parse pattern; `score ≥ threshold` declares jailbreak

#### Adapters
- `GroqAdapter` — async Groq API client with exponential backoff on rate limits (5s → 10s → 20s)
- `OpenAIAdapter` — async OpenAI-compatible client; supports custom `base_url` for proxies
- `OllamaAdapter` — local model server via httpx; no API key required

#### CLI (`promptstrike`)
- `scan` — run PAIR or TAP against any supported target; live Rich progress bar
- `ci` — CI/CD safety gate; `exit 1` if ASR > threshold; `--json` for log parsing; `--report` for HTML artifact
- `sweep` — run the same behavior sample across N models; shared behaviors ensure comparability; Rich comparison table
- `history` — list all past campaigns from SQLite
- `report` — export OWASP LLM Top 10 mapped HTML report for any campaign
- `serve` — start FastAPI + WebSocket live dashboard

#### Dashboard
- FastAPI backend with `POST /api/scan`, `GET /api/campaigns`, `GET /api/campaigns/{id}/report`, `WS /ws/{scan_id}`
- Glassmorphism HTML dashboard — real-time attack feed over WebSocket; goal cards with score bars; campaign history sidebar

#### Reports
- Single-campaign glassmorphism HTML report — 4 KPI cards, OWASP breakdown bars, category chips, jailbreak spotlight grid, full results table
- Multi-model sweep comparison report — model rankings with ASR bars, category heatmap (rows = categories, cols = models, colors = red/amber/green)

#### Data
- 50 behavior goals across 10 attack categories: cybercrime, malware, fraud, privacy, disinformation, weapons, violence, hate, financial crime, AI attacks
- Each goal tagged to OWASP LLM01–LLM09

#### Storage
- SQLite persistence: `campaigns`, `attack_runs`, `iterations` tables
- Full attack tree stored: every iteration, attacker prompt, target response, judge score

#### Tests (30 total)
- `test_pair.py` (9) — JSON parsing, loop logic, call tracking, ASR property
- `test_tap.py` (9) — budget guard, success path, 3-calls-per-node invariant, max-score, depth tracking, FAILED status
- `test_api.py` (6) — dashboard HTML, campaigns list, 404, missing key rejection, scan_id, WebSocket error
- `test_ci.py` (6) — exit codes 0/1/2, threshold=0 strict mode, JSON output structure

#### CI/CD
- GitHub Actions workflow: unit tests on every push/PR + LLM safety gate on merges to main
- Step summary with formatted gate result table
- HTML report uploaded as 30-day build artifact
