"""AutoResearch Demo - Streamlit version with LIVE arXiv + Semantic Scholar data."""
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import time

# ---------------- Live data fetchers ----------------

def fetch_arxiv(topic, max_results=6):
    """Query arXiv's public API and parse the Atom XML response."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    papers = []
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            published = entry.find("atom:published", ns).text[:4]
            link = entry.find("atom:id", ns).text
            papers.append({
                "title": title,
                "authors": authors,
                "year": int(published),
                "abstract": summary,
                "citations": None,  # arXiv doesn't provide citation counts
                "url": link,
                "source": "arXiv"
            })
    except Exception as e:
        st.warning(f"arXiv fetch failed: {e}")
    return papers


def fetch_semantic_scholar(topic, max_results=6):
    """Query Semantic Scholar's public API (no key required for light use)."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,abstract,year,authors,citationCount,url"
    }
    papers = []
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for p in data.get("data", []):
            papers.append({
                "title": p.get("title", "Untitled"),
                "authors": [a.get("name", "") for a in p.get("authors", [])],
                "year": p.get("year"),
                "abstract": p.get("abstract") or "No abstract available.",
                "citations": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "source": "Semantic Scholar"
            })
    except Exception as e:
        st.warning(f"Semantic Scholar fetch failed: {e}")
    return papers


def gather_papers(topic, max_papers):
    papers = fetch_semantic_scholar(topic, max_papers)
    time.sleep(1)  # be polite to the free API
    papers += fetch_arxiv(topic, max_papers)
    # dedupe by title (case-insensitive)
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:max_papers]


# ---------------- Simple analysis (still yours to expand) ----------------

def detect_gaps(papers):
    gaps = []
    years = [p["year"] for p in papers if p.get("year")]
    if years and max(years) - min(years) > 3:
        gaps.append({"type": "temporal", "description": "Papers span multiple years — check if older methods are still state of the art.", "score": 14})
    if any(p.get("citations") == 0 for p in papers if p.get("citations") is not None):
        gaps.append({"type": "evaluation", "description": "Some retrieved papers have very low citation counts — may indicate niche or emerging subtopics worth investigating.", "score": 11})
    gaps.append({"type": "cross_domain", "description": "Check whether methods here have been applied to adjacent domains — often a fruitful gap.", "score": 10})
    return gaps


def generate_hypotheses(papers, gaps):
    hyps = []
    for g in gaps:
        hyps.append({
            "hypothesis": f"Based on the '{g['type']}' gap, a follow-up study could directly address: {g['description']}",
            "confidence": 0.6,
            "feasibility": "medium"
        })
    return hyps


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="AutoResearch", page_icon="🔬", layout="wide")
st.title("🔬 AutoResearch: Multi-Agent Research Assistant")
st.markdown("Live literature search across **arXiv** and **Semantic Scholar**.")

with st.form("research_form"):
    topic = st.text_input("Research Topic", placeholder="e.g., transformer architectures for medical image segmentation")
    max_papers = st.slider("Max Papers", min_value=3, max_value=10, value=6)
    submitted = st.form_submit_button("Run Research Pipeline")

if submitted:
    if not topic or len(topic.strip()) < 3:
        st.error("Please enter a valid research topic (at least 3 characters).")
    else:
        with st.spinner("Querying arXiv and Semantic Scholar..."):
            papers = gather_papers(topic.strip(), max_papers)

        if not papers:
            st.error("No papers found. Try a broader topic or try again in a minute (API rate limits).")
        else:
            gaps = detect_gaps(papers)
            hypotheses = generate_hypotheses(papers, gaps)

            st.success(f"Found {len(papers)} papers")

            col1, col2, col3 = st.columns(3)
            col1.metric("Papers Found", len(papers))
            col2.metric("Gaps Detected", len(gaps))
            col3.metric("Hypotheses Generated", len(hypotheses))

            st.markdown("---")
            st.header("1. Literature Found")
            for p in papers:
                with st.expander(f"{p['title']} ({p.get('year', 'n/a')}) — {p['source']}"):
                    st.write(f"**Authors:** {', '.join(p['authors']) if p['authors'] else 'Unknown'}")
                    if p.get("citations") is not None:
                        st.write(f"**Citations:** {p['citations']}")
                    st.write(p["abstract"])
                    if p.get("url"):
                        st.markdown(f"[View paper]({p['url']})")

            st.markdown("---")
            st.header("2. Research Gaps")
            for i, g in enumerate(gaps, 1):
                st.markdown(f"**{i}. [{g['type'].title()}]** (Score: {g['score']})")
                st.markdown(f"> {g['description']}")

            st.markdown("---")
            st.header("3. Generated Hypotheses")
            for i, h in enumerate(hypotheses, 1):
                st.markdown(f"**{i}.** {h['hypothesis']}")
                st.markdown(f"- Confidence: {h['confidence']*100:.0f}% | Feasibility: {h['feasibility'].title()}")
