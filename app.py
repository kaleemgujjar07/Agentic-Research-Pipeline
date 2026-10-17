"""AutoResearch: 7-agent research pipeline with an interactive human-AI Gap Explorer (Streamlit front-end).

The same pipeline and the same Gap Explorer service also back the FastAPI app in autoresearch/api.
"""
import streamlit as st

from autoresearch.core.orchestrator import AutoResearchOrchestratorV3
from gap_explorer_ui import render_gap_explorer

st.set_page_config(page_title="AutoResearch", page_icon="🔬", layout="wide")
st.title("🔬 AutoResearch: 7-Agent Research Pipeline")
st.markdown(
    "*Literature discovery → deep reading → citation network → gap detection → hypotheses → "
    "screening → report, plus an interactive Gap Explorer where you and the AI organise the gaps together.*"
)

with st.sidebar:
    st.header("🤖 The 7 agents")
    st.markdown(
        """
        1️⃣ **Literature Discovery** – arXiv + Semantic Scholar  
        2️⃣ **Deep Reading** – methods, domains, limitations (Claude if a key is set, else heuristic)  
        3️⃣ **Citation Network** – similarity graph + clusters  
        4️⃣ **Gap Detection** – methodological, evaluation, stated limitations…  
        5️⃣ **Hypothesis Generation** – template-based  
        6️⃣ **Critic** – automated screening checks  
        7️⃣ **Report Generator** – Markdown report  

        👤 **You + AI** – Gap Explorer: group, re-cluster, star, prune
        """
    )
    st.markdown("---")
    st.caption("Rule-based by default. Add an Anthropic API key to let Claude read the papers and write a brief. "
               "The Gap Explorer is where human judgement enters.")
    user_key = st.text_input("Anthropic API key (optional)", type="password",
                             help="Used only for this session; never stored or logged.")
    st.caption("💡 Try: vision transformers, graph neural networks, federated learning")

with st.form("form"):
    topic = st.text_input("Research topic", placeholder="e.g., vision transformers for medical imaging")
    max_papers = st.slider("Max papers", 3, 15, 8)
    submitted = st.form_submit_button("🚀 Run the 7-agent pipeline")

if submitted:
    if not topic or len(topic.strip()) < 3:
        st.error("❌ Enter a valid research topic (≥3 characters).")
    else:
        # One orchestrator per browser session: it owns this session's Gap Explorer state.
        from autoresearch.llm import get_llm_client
        try:
            secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            secret_key = ""
        llm = get_llm_client(user_key or secret_key) or get_llm_client()
        # Rebuild the orchestrator only when the LLM on/off state changes, so Gap Explorer state is kept otherwise.
        if "orch" not in st.session_state or st.session_state.get("orch_llm_key") != (user_key or secret_key):
            st.session_state["orch"] = AutoResearchOrchestratorV3(mode="full", llm_client=llm)
            st.session_state["orch_llm_key"] = user_key or secret_key
        orch = st.session_state["orch"]
        with st.spinner("Running 7 agents (retrieval from public APIs can take 30–60 s)…"):
            result = orch.run(topic.strip(), max_papers=max_papers)
        st.session_state["result"] = result

result = st.session_state.get("result")
if result:
    if "error" in result:
        st.error(f"{result['error']}. Try a broader topic.")
        for f in result.get("findings", []):
            st.caption(str(f))
        status = (result.get("discovery") or {}).get("source_status") or {}
        if status:
            st.markdown("**What each source returned:**")
            for src, s in status.items():
                outcome = f"{s.get('count', 0)} papers" if s.get("ok") else f"failed ({s.get('error', 'unknown error')})"
                st.markdown(f"- `{src}`: {outcome}")
            if not any(s.get("ok") for s in status.values()):
                st.info("Every source failed. This is usually a temporary rate limit on the free paper APIs. "
                        "Wait a minute and try again; successful searches are cached for an hour.")
    else:
        orch = st.session_state["orch"]
        stats = result["pipeline_stats"]
        papers = result["stages"]["discovery"]["papers"]
        disc = result["stages"]["discovery"]
        note = f" (served from cache)" if disc.get("from_cache") else ""
        note += f" (no matches for the full query, so searched '{disc['broadened_query']}')" if disc.get("broadened_query") else ""
        st.success(f"✅ Pipeline complete in {stats['total_time_seconds']}s. {len(papers)} real papers analysed{note}.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers", stats["papers_found"])
        c2.metric("Gaps detected", stats["gaps_detected"])
        c3.metric("Hypotheses", stats["hypotheses_generated"])
        c4.metric("Passed screening", stats["hypotheses_verified"])

        st.markdown("---")
        st.header("📄 Papers retrieved")
        for p in papers:
            year = p.get("published_year") or "n/a"
            cites = p.get("citation_count") or 0
            title = p.get("title", "Untitled")
            url = p.get("url", "")
            label = f"[{title}]({url})" if url.startswith(("http://", "https://")) else title
            st.markdown(f"- **{label}** ({year}) · {p.get('source', '')} · "
                        f"{'⭐ ' + str(cites) + ' citations' if cites else 'no citation data'}")

        st.markdown("---")
        gx = result.get("gap_explorer", {})
        sid = gx.get("session_id")
        if sid:
            render_gap_explorer(orch.explorers.get_state(sid), orch.explorers.get_coords(sid), sid)
        else:
            st.info(gx.get("reason", "No gaps to organise."))

        st.markdown("---")
        llm_info = result.get("llm", {})
        if llm_info.get("brief"):
            st.header("🧠 AI research brief")
            st.caption(f"Written by {llm_info['model']} from the detected gaps; "
                       f"{llm_info['papers_read_by_llm']} paper(s) read by the LLM. Verify before relying on it.")
            st.write(llm_info["brief"])
        st.header("💡 Hypotheses (template-based, automated screening)")
        for h in result["stages"]["verification"].get("verified_hypotheses", [])[:6]:
            st.markdown(f"- {h['hypothesis']}  \n  _feasibility: {h.get('feasibility', 'n/a')}_")

        st.markdown("---")
        st.header("📝 Report")
        md = orch.curated_report_markdown(sid) if sid else result["markdown_report"]
        st.caption("The gap section of this report reflects how you organised the gaps above."
                   if sid else "")
        st.download_button("📥 Download report (.md)", data=md, file_name="research_report.md",
                           mime="text/markdown", use_container_width=True)
        with st.expander("Preview report"):
            st.markdown(md)
