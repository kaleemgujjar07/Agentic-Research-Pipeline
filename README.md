# AutoResearch v3.0: Unified Multi-Agent Research System

**A production-grade, modular research assistant with specialized capabilities for security, computer vision, data science, and IoT domains.**

## 🎯 Why This Project Is Different

Unlike typical FYP projects (dashboards, chatbots, management systems), AutoResearch is a **research infrastructure system** that demonstrates:
- Multi-agent architecture design
- Real academic API integration (arXiv, Semantic Scholar, PubMed)
- Security-hardened input processing
- Computer vision for document analysis
- Statistical analysis and forecasting
- Distributed systems messaging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTORESEARCH v3.0                          │
├─────────────────────────────────────────────────────────────────┤
│  SECURITY MODULE          │  VISION MODULE                     │
│  • Input sanitization     │  • PDF figure extraction           │
│  • Rate limiting          │  • Chart type detection            │
│  • Adversarial testing    │  • Network graph generation        │
│  • Sandboxed PDF parsing  │  • Trend visualization data        │
├─────────────────────────────────────────────────────────────────┤
│  DATA SCIENCE MODULE      │  IOT MODULE                        │
│  • Citation impact metrics│  • Distributed agent messaging     │
│  • Trend forecasting      │  • Heartbeat monitoring            │
│  • Statistical gap tests  │  • Edge deployment mode            │
│  • Research health scores │  • Lightweight operation           │
├─────────────────────────────────────────────────────────────────┤
│                     CORE PIPELINE (7 Agents)                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Literature Discovery → arXiv + Semantic Scholar + PubMed   │
│  2. Deep Reading → Methodology, claims, limitations extraction │
│  3. Citation Network → Graph building, clustering, bridges     │
│  4. Gap Detection → 5 gap types with statistical scoring       │
│  5. Hypothesis Generation → Testable hypotheses from gaps      │
│  6. Critic/Verification → Filters against existing literature  │
│  7. Report Generator → Structured, citable output              │
├─────────────────────────────────────────────────────────────────┤
│                     INTERFACES                                  │
├─────────────────────────────────────────────────────────────────┤
│  • FastAPI REST API        • Interactive HTML Dashboard         │
│  • Real-time status updates • JSON/Markdown export              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API with dashboard
python -m autoresearch.api.main

# Or run from Python
from autoresearch.core.orchestrator import AutoResearchOrchestratorV3
o = AutoResearchOrchestratorV3(mode="full")
result = o.run("transformer architectures for medical imaging", max_papers=10)
print(result["markdown_report"])
```

## 📊 Dashboard

Visit `http://localhost:8000/` to see the interactive dashboard showing:
- Real-time pipeline status
- Security module results
- Data science metrics (citation impact, health score)
- Vision module outputs (network graphs)
- IoT agent health monitoring

## 🔒 Security Features

- **Input Sanitization**: Detects SQL injection, XSS, path traversal, entropy-based garbage detection
- **Rate Limiting**: Per-client request throttling
- **Adversarial Testing**: Automated test suite against 7 attack vectors
- **Sandboxed PDF Parsing**: File size limits, extension validation, hash verification

## 👁️ Computer Vision Features

- **Figure Extraction**: Extracts images and captions from academic PDFs
- **Chart Detection**: Classifies charts as line, bar, pie, or scatter using OpenCV
- **Network Visualization**: Generates D3.js-compatible graph data for citation networks
- **Trend Charts**: Time-series visualization data for publication trends

## 📈 Data Science Features

- **Citation Impact Metrics**: H-index, mean/median citations, velocity trends
- **Trend Forecasting**: Linear regression-based publication forecasting
- **Statistical Gap Testing**: Significance testing for detected research gaps
- **Research Health Score**: Composite score based on diversity, limitations, and recency

## 📡 IoT Features

- **Distributed Messaging**: MQTT-style pub/sub between agents
- **Heartbeat Monitoring**: Agent health tracking with automatic failure detection
- **Edge Mode**: Lightweight deployment for resource-constrained devices
- **Message Logging**: Complete audit trail of inter-agent communication

## 🛠️ Tech Stack

- **Python 3.8+**
- **FastAPI** for REST API
- **SQLAlchemy** for database
- **OpenCV + PIL** for vision
- **PyMuPDF + pdfplumber** for PDF parsing
- **Requests** for academic APIs

## 📁 Project Structure

```
autoresearch/
├── agents/              # 7 core research agents
├── modules/             # 4 specialized modules
│   ├── security.py      # Cybersecurity features
│   ├── vision.py        # Computer vision features
│   ├── datascience.py   # Data science features
│   └── iot.py           # IoT/distributed features
├── core/                # Orchestrator
├── api/                 # FastAPI + Dashboard
└── tests/               # Unit and integration tests
```

## 📄 License

MIT License — Built for research and educational purposes.
