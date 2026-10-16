"""Synthetic test abstracts (NOT real papers) used only by unit tests."""
PAPERS = [
    {"title": "Paper A", "year": 2022, "url": "http://x/a", "abstract":
     "We propose a transformer for 3D medical image segmentation. Results are strong on public benchmarks. However, annotated medical data is scarce and labeling is expensive, which limits supervised training. Future work should address data efficiency."},
    {"title": "Paper B", "year": 2023, "url": "http://x/b", "abstract":
     "A hybrid vision transformer is presented for CT analysis. The model is computationally expensive and memory usage remains a challenge for deployment on edge devices. Latency is a bottleneck in clinical settings."},
    {"title": "Paper C", "year": 2021, "url": "http://x/c", "abstract":
     "We study self-supervised pretraining for radiology. Generalization across hospitals remains underexplored, and models suffer under domain shift. Cross-dataset evaluation is rarely reported."},
    {"title": "Paper D", "year": 2024, "url": "http://x/d", "abstract":
     "This survey reviews attention models in imaging. Existing benchmarks lack standardized evaluation protocols, making comparison between methods difficult. Metrics differ across studies."},
    {"title": "Paper E", "year": 2022, "url": "http://x/e", "abstract":
     "Interpretability is a major concern: black box predictions are hard to trust in clinical practice. Explanations produced by attention maps often fail to match radiologist reasoning."},
    {"title": "Paper F", "year": 2023, "url": "http://x/f", "abstract":
     "We train on a small labeled dataset with limited annotation. Training large models requires large-scale data that is not available in many hospitals."},
    {"title": "No cues", "year": 2020, "url": "http://x/g", "abstract":
     "We present a new network. It achieves good accuracy on three datasets with fast inference."},
    {"title": "Empty", "year": 2020, "url": "", "abstract": ""},
]
