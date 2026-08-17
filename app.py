"""AutoResearch Demo - Gradio interface for Hugging Face Spaces."""
import gradio as gr
import sys
import os

# Add autoresearch to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoresearch.core.orchestrator import AutoResearchOrchestratorV3

orchestrator = AutoResearchOrchestratorV3(mode="full")

def run_research(topic, max_papers):
    if not topic or len(topic.strip()) < 3:
        return "Please enter a valid research topic (at least 3 characters)."
    
    try:
        result = orchestrator.run(
            topic=topic.strip(),
            max_papers=int(max_papers),
            sources=["arxiv", "semantic_scholar"]
        )
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        markdown = result.get("markdown_report", "Report generation failed.")
        stats = result.get("pipeline_stats", {})
        
        stats_text = f"""## Pipeline Statistics
- **Execution Time:** {stats.get('total_time_seconds', 0):.1f} seconds
- **Papers Found:** {stats.get('papers_found', 0)}
- **Gaps Detected:** {stats.get('gaps_detected', 0)}
- **Hypotheses Generated:** {stats.get('hypotheses_generated', 0)}
- **Hypotheses Verified:** {stats.get('hypotheses_verified', 0)}
"""
        return stats_text + "\n\n---\n\n" + markdown
        
    except Exception as e:
        return f"Error during execution: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=run_research,
    inputs=[
        gr.Textbox(label="Research Topic", placeholder="e.g., transformer architectures for medical image segmentation", lines=2),
        gr.Slider(minimum=5, maximum=15, value=8, step=1, label="Max Papers")
    ],
    outputs=gr.Markdown(label="Research Report"),
    title="🔬 AutoResearch: Multi-Agent Research Assistant",
    description="""Enter a research topic and the system will:
1. Discover papers from arXiv and Semantic Scholar
2. Analyze methodology, claims, and limitations
3. Build citation networks and detect research clusters
4. Identify gaps and generate verified hypotheses
5. Produce a structured research report""",
    examples=[
        ["transformer architectures for medical image segmentation", 8],
        ["reinforcement learning for robotics", 8],
        ["large language model safety", 8]
    ]
)

if __name__ == "__main__":
    demo.launch()
