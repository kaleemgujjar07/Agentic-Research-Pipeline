# 🔬 AutoResearch: 7-Agent Research Pipeline with an Interactive Human-AI Gap Explorer

A Python / FastAPI pipeline of seven agents that discovers papers, reads them, detects research gaps,
drafts hypotheses and writes a report, plus an **interactive Gap Explorer** where the researcher and the AI
organise the detected gaps together.

> Final Year Project, BS Computer Science.
> **Live demo:** https://agentic-research-pipeline-vt8dv8hpisclw4axfknqhm.streamlit.app/

## The 7 agents

| # | Agent | What it does | How |
|---|---|---|---|
| 1 | Literature Discovery | Finds papers on a topic | arXiv + Semantic Scholar APIs (PubMed titles optional), de-duplicated |
| 2 | Deep Reading | Extracts methods, domains, claims, limitations, metrics | Keyword heuristics (an optional `llm_client` hook exists; no LLM is bundled) |
| 3 | Citation Network | Similarity graph, clusters, bridge papers | Domain/method/author overlap (not real citation links) |
| 4 | Gap Detection | Methodological, evaluation, temporal, cross-domain and **stated-limitation** gaps | Rules over the reading output; stated limitations keep a link to their source paper |
| 5 | Hypothesis Generation | One hypothesis per gap | Templates |
| 6 | Critic | Screens hypotheses | Keyword / overlap checks (heuristic, **not** expert verification) |
| 7 | Report Generator | Structured Markdown report | Templates over all agent outputs |

An orchestrator runs them in order. Extra modules: input sanitisation (security), citation-network and
trend chart data (vision), simple statistics (data science), and a *simulated* agent-health/IoT layer.

## 🧩 Gap Explorer (human-AI co-organisation of the detected gaps)

Every gap the Gap Detection agent finds becomes a dot on a map. The AI proposes a grouping and you can
reshape it:

- **AI proposes**: TF-IDF + KMeans clusters, named from their most distinctive terms
- **You override**: draw a loop (lasso) around dots and group them under your own name
- **AI never moves what you grouped**: re-clustering only touches gaps you haven't grouped
- **Preview before applying**: moving the cluster slider shows how many gaps would move, and which
- **Everything is editable**: rename AI clusters, star important gaps, remove irrelevant ones, hand gaps back to the AI
- **Provenance**: stated-limitation gaps show the paper they came from; aggregate gaps are marked "Cross-paper analysis"
- **The report follows you**: the downloadable report's gap section reflects *your* organisation

Design is informed by *MindTrellis: Co-Creating Knowledge Structures with AI through Interactive Visual
Exploration* (Li, Li, Kuang, Liu, Zhao; ACM DIS 2026, doi:10.1145/3800645.3813045): AI structure is
provisional and editable, changes are previewed before they happen, and manual organisation is respected.
This is a much smaller prototype: it has no chat interface, no LLM, no knowledge graph, and no user study.

## Run it

```bash
git clone https://github.com/kaleemgujjar07/Agentic-Research-Pipeline.git
cd Agentic-Research-Pipeline
pip install -r requirements.txt

streamlit run app.py                          # Streamlit front-end
uvicorn autoresearch.api.main:app --reload    # FastAPI backend + dashboard at http://localhost:8000
```

Both front-ends run the same 7-agent orchestrator and the same Gap Explorer service.

### FastAPI endpoints
| Endpoint | Purpose |
|---|---|
| `POST /api/research` | Run the 7-agent pipeline; the response includes a `gap_explorer` session |
| `GET /api/explorer/{id}` | Gaps, groups, map coordinates, history |
| `POST /api/explorer/{id}/preview` `{k}` | What a re-clustering *would* move (changes nothing) |
| `POST /api/explorer/{id}/recluster` `{k}` | AI re-clusters the gaps you haven't grouped |
| `POST /api/explorer/{id}/group` `{ids, name}` | Group gaps under your own name |
| `POST /api/explorer/{id}/unpin` · `star` · `remove` · `restore` · `rename` | Edit the organisation |
| `GET /api/explorer/{id}/export` · `/report` | Gap map / full report as Markdown |

Interactive docs: `/docs`.

## Tests
```bash
python -m pytest tests -q
```
54 tests: agents, explorer logic, the HTTP API (including validation errors), and headless end-to-end runs
of the Streamlit app. Paper APIs are mocked, so tests run offline.

## Limitations (honest)
- Agents are rule-based; no LLM. Gap quality depends on what authors write in **abstracts** (no full text).
- Clustering is lexical (TF-IDF), not embedding-based, so paraphrases may not group together.
- "Verification" and the scores are heuristics, not validation of scientific merit.
- Explorer sessions live in server memory (last 25); they are lost on restart and are not saved per user.
- The citation network is built from topical similarity, not real citation data.
- No user study has been run.

## Project structure
```
├── app.py                      # Streamlit front-end (7-agent pipeline + explorer)
├── gap_explorer_ui.py          # Streamlit panel for the Gap Explorer
├── autoresearch/
│   ├── core/orchestrator.py    # runs the 7 agents, opens explorer sessions
│   ├── agents/                 # the 7 agents
│   ├── modules/                # gap_explorer (shared core), security, vision, datascience, iot
│   └── api/                    # FastAPI app + interactive dashboard
├── tests/
└── requirements.txt
```
