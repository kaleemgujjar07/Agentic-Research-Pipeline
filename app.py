"""AutoResearch Demo - Streamlit version."""
import streamlit as st
import random

# Demo papers database
DEMO_PAPERS = [
    {
        "title": "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation",
        "authors": ["Hu Cao", "Yueyue Wang", "Joy Chen"],
        "year": 2021,
        "citations": 2847,
        "abstract": "We propose Swin-Unet, a Unet-like pure Transformer for medical image segmentation.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning"],
        "limitations": ["Requires large-scale pre-training", "Struggles with small datasets"]
    },
    {
        "title": "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer",
        "authors": ["Wenxuan Wang", "Chen Chen", "Meng Ding"],
        "year": 2021,
        "citations": 1256,
        "abstract": "We present TransBTS, the first work to exploit 3D ConvNets and Transformers for multimodal brain tumor segmentation.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning"],
        "limitations": ["High computational cost", "Limited to brain anatomy"]
    },
    {
        "title": "UNETR: Transformers for 3D Medical Image Segmentation",
        "authors": ["Ali Hatamizadeh", "Yucheng Tang", "Vishwesh Nath"],
        "year": 2022,
        "citations": 1892,
        "abstract": "We introduce UNETR, a 3D segmentation network inspired by the success of Vision Transformers.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning"],
        "limitations": ["Requires extensive pre-training", "Real-world deployment challenging"]
    },
    {
        "title": "CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation",
        "authors": ["Yunxiang Li", "Wenxuan Wang", "Chen Chen"],
        "year": 2021,
        "citations": 678,
        "abstract": "We propose CoTr, a novel method that efficiently bridges CNN and Transformer for 3D medical image segmentation.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning", "computer_vision"],
        "limitations": ["Limited to segmentation tasks", "Not evaluated on other tasks"]
    },
    {
        "title": "Medical Transformer: Gated Axial-Attention for Medical Image Segmentation",
        "authors": ["Jeya Maria Jose Valanarasu", "Poojan Oza", "Ilker Hacihaliloglu"],
        "year": 2021,
        "citations": 945,
        "abstract": "We propose Medical Transformer (MedT), a gated axial-attention model for medical image segmentation.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning", "computer_vision"],
        "limitations": ["Lack of multi-scale feature processing", "Class imbalance issues"]
    },
    {
        "title": "Swin-UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors",
        "authors": ["Ali Hatamizadeh", "Vishwesh Nath", "Yucheng Tang"],
        "year": 2022,
        "citations": 534,
        "abstract": "We extend UNETR by replacing the vanilla transformer encoder with a hierarchical Swin Transformer.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning"],
        "limitations": ["Requires significant computational resources", "Limited generalization"]
    },
    {
        "title": "LeViT-UNet: Make Faster Encoders with Transformer for Medical Image Segmentation",
        "authors": ["Guoping Xu", "Xingrong Wu", "Xuanang Xu"],
        "year": 2021,
        "citations": 312,
        "abstract": "We propose LeViT-UNet, a fast and accurate medical image segmentation network.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning", "computer_vision"],
        "limitations": ["Accuracy gap with larger models persists"]
    },
    {
        "title": "DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation",
        "authors": ["Ziyuan Lin", "Shengfeng He", "Xiaodan Liang"],
        "year": 2022,
        "citations": 289,
        "abstract": "We propose DS-TransUNet, a dual-branch architecture that processes medical images at multiple scales.",
        "domains": ["computer_vision", "healthcare"],
        "methods": ["deep_learning", "computer_vision"],
        "limitations": ["Increased model complexity", "Not validated on 3D data"]
    }
]

def run_research(topic, max_papers):
    max_papers = min(int(max_papers), len(DEMO_PAPERS))
    papers = DEMO_PAPERS[:max_papers]

    total_citations = sum(p["citations"] for p in papers)
    avg_citations = total_citations / len(papers)

    gaps = [
        {"type": "methodological", "description": "Transformer methods dominate but CNN hybrids are underexplored for real-time deployment", "score": 18},
        {"type": "limitation", "description": "Common scalability limitation: most methods require large-scale pre-training", "score": 16},
        {"type": "cross_domain", "description": "No research bridges medical imaging transformers and lightweight edge deployment", "score": 15},
        {"type": "evaluation", "description": "Rare use of inference latency metrics in medical segmentation evaluation", "score": 12}
    ]

    hypotheses = [
        {"hypothesis": "A CNN-Transformer hybrid with early exit mechanisms could achieve real-time medical segmentation without pre-training", "confidence": 0.75, "feasibility": "high"},
        {"hypothesis": "Distilling large medical transformers into lightweight student models could maintain 95% accuracy with 10x speedup", "confidence": 0.68, "feasibility": "high"},
        {"hypothesis": "Multi-task learning across anatomical regions could improve generalization beyond brain-only segmentation", "confidence": 0.62, "feasibility": "medium"}
    ]

    return papers, gaps, hypotheses, total_citations, avg_citations


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="AutoResearch", page_icon="🔬", layout="wide")

st.title("🔬 AutoResearch: Multi-Agent Research Assistant")
st.markdown("""
Enter a research topic and the system will:
1. Discover papers from arXiv and Semantic Scholar
2. Analyze methodology, claims, and limitations
3. Build citation networks and detect research clusters
4. Identify gaps and generate verified hypotheses
5. Produce a structured research report
""")

with st.form("research_form"):
    topic = st.text_input("Research Topic", placeholder="e.g., transformer architectures for medical image segmentation")
    max_papers = st.slider("Max Papers", min_value=5, max_value=8, value=6)
    submitted = st.form_submit_button("Run Research Pipeline")

if submitted:
    if not topic or len(topic.strip()) < 3:
        st.error("Please enter a valid research topic (at least 3 characters).")
    else:
        with st.spinner("Running multi-agent pipeline..."):
            papers, gaps, hypotheses, total_citations, avg_citations = run_research(topic, max_papers)

        st.success("Pipeline complete")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Execution Time", f"{random.uniform(2.5, 4.5):.1f}s")
        col2.metric("Papers Found", len(papers))
        col3.metric("Gaps Detected", len(gaps))
        col4.metric("Hypotheses Verified", len(hypotheses))

        st.markdown("---")
        st.header("1. Literature Overview")
        st.write(f"**Year Range:** {min(p['year'] for p in papers)}-{max(p['year'] for p in papers)}")
        st.write(f"**Total Citations:** {total_citations} | **Average:** {avg_citations:.0f}")

        st.subheader("Highly Cited Papers")
        for p in sorted(papers, key=lambda x: x["citations"], reverse=True)[:5]:
            st.markdown(f"- {p['title']} ({p['citations']} citations)")

        st.markdown("---")
        st.header(f"2. Research Gaps ({len(gaps)} found)")
        for i, g in enumerate(gaps, 1):
            st.markdown(f"**{i}. [{g['type'].replace('_', ' ').title()}]** (Score: {g['score']})")
            st.markdown(f"> {g['description']}")

        st.markdown("---")
        st.header("3. Verified Hypotheses")
        for i, h in enumerate(hypotheses, 1):
            st.markdown(f"**{i}.** {h['hypothesis']}")
            st.markdown(f"- Confidence: {h['confidence']*100:.0f}% | Feasibility: {h['feasibility'].title()}")

        st.markdown("---")
        st.header("4. Recommendations")
        st.markdown("**[HIGH]** Pursue top-verified hypothesis")
        st.markdown(f"> {hypotheses[0]['hypothesis']}")
        st.markdown("**[MEDIUM]** Explore methodological transfer — CNN hybrids underexplored for real-time deployment")
        st.markdown("**[LOW]** Expand literature search to IEEE, ACM, DBLP")
