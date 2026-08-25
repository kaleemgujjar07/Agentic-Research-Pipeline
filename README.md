# 🔬 AutoResearch: Multi-Agent Research Pipeline

## Final Year Project - BS Computer Science


## 📖 Project Overview

**AutoResearch** is a multi-agent research assistant that autonomously performs:

1. **Paper Retrieval** – Fetches real papers from ArXiv and Semantic Scholar
2. **Relevance Filtering** – Ranks papers using TF-IDF (Classical ML)
3. **Gap Detection** – Identifies research gaps using rules or LLM
4. **Code Generation** – Writes executable PyTorch code

---

## 🧠 Agentic Architecture

The system has **4 autonomous agents** managed by an orchestrator:

| Agent | Function | Decision-Making |
|-------|----------|-----------------|
| **1. Retrieval** | Fetches papers from ArXiv/Semantic Scholar | Autonomously selects API (Semantic Scholar → ArXiv fallback) |
| **2. Filter** | Ranks papers by relevance | Uses TF-IDF + Cosine Similarity |
| **3. Gap Detector** | Analyzes abstracts, finds gaps | Rule-based OR LLM (Ollama) |
| **4. Code Generator** | Writes PyTorch code | Template-based OR LLM (Ollama) |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Language | Python 3.11+ |
| Classical ML | Scikit-learn (TF-IDF) |
| Data Sources | ArXiv API, Semantic Scholar API |
| PDF Export | FPDF |

---

## 📦 Installation

### Option 1: Cloud Version

```bash
git clone https://github.com/kaelemgujjar07/agentic-research-pipeline.git
cd agentic-research-pipeline
pip install -r requirements.txt
streamlit run app.py
```


## 📊 Output Example

### Papers Retrieved
- Transformers in Medical Imaging: A Survey (2022) - 2847 citations
- UNETR: Transformers for 3D Medical Image Segmentation (2022) - 1892 citations

### Research Gaps Detected
1. Papers mention limitations: limitation, challenge. Addressing these could improve methods.
2. Existing methods evaluated on limited datasets. Cross-dataset generalization remains underexplored.
3. Computational efficiency is mentioned but not thoroughly benchmarked.

### Generated Code
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # ... complete PyTorch code generated
```

---

## 📁 Project Structure

```
agentic-research-pipeline/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # This file
```


## 🔗 Link

- **Live Demo**: [Streamlit App](https://agentic-research-pipeline-vt8dv8hpisclw4axfknqhm.streamlit.app/)
