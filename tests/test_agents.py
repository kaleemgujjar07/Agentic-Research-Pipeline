import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from autoresearch.agents.citation_network import CitationNetworkBuilder
from autoresearch.agents.deep_reading import DeepReadingAgent
from autoresearch.agents.gap_detection import GapDetectionAgent
from autoresearch.agents.report_generator import ReportGenerator
from fixtures import PAPERS


def _papers():
    return [{"external_id": str(i), "title": p["title"], "abstract": p["abstract"], "published_year": p["year"],
             "url": p["url"], "authors": [f"A{i}"], "citation_count": i, "venue": "V"} for i, p in enumerate(PAPERS)]


def test_reading_keeps_year_and_url_and_year_distribution():
    res = DeepReadingAgent().execute(_papers())
    first = res["analyses"][0]
    assert first["published_year"] == 2022 and first["url"] == "http://x/a"
    assert res["summary"]["year_distribution"]            # was always empty before the fix


def test_reading_detects_limitations_and_metrics():
    a = DeepReadingAgent().analyze_paper({"title": "T", "abstract":
        "We report accuracy and F1 on two datasets. However the model is computationally expensive and data is scarce."})
    assert a["limitations"] and "accuracy" in a["evaluation_metrics"] and "f1" in a["evaluation_metrics"]


def test_gap_agent_stated_limitations_have_provenance():
    analyses = DeepReadingAgent().execute(_papers())["analyses"]
    out = GapDetectionAgent().execute(analyses, {"clusters": [], "edges": []})
    stated = [g for g in out["all_gaps"] if g["type"] == "stated_limitation"]
    assert len(stated) >= 5
    for g in stated:
        assert g["paper_title"] and g["description"] and g["theme"]
    assert len({g["description"].lower() for g in stated}) == len(stated)     # de-duplicated
    assert out["total_gaps"] == len(out["all_gaps"]) and len(out["gaps"]) <= 15


def test_gap_agent_never_emits_placeholder_gaps():
    analyses = DeepReadingAgent().execute(_papers())["analyses"]
    out = GapDetectionAgent().execute(analyses, {"clusters": [], "edges": []})
    for g in out["all_gaps"]:
        assert "not_specified" not in g["description"] and "general_ai" not in g["description"]
        assert "No explicit limitations" not in g["description"]


def test_gap_agent_handles_empty_input():
    out = GapDetectionAgent().execute([], {"clusters": [], "edges": []})
    assert out["total_gaps"] == 0 and out["all_gaps"] == []


def test_bridge_papers_with_cluster_id_zero():
    """Regression: cluster id 0 used to be treated as 'no cluster' (falsy)."""
    nodes = [{"title": f"n{i}", "citations": 0} for i in range(4)]
    clusters = [{"id": 0, "node_ids": [0, 1]}, {"id": 2, "node_ids": [2, 3]}]
    edges = [{"source": 1, "target": 2, "weight": 0.5, "type": "cross_domain"}]
    bridges = CitationNetworkBuilder()._find_bridge_papers(nodes, edges, clusters)
    assert {b["node_id"] for b in bridges} == {1, 2}


def test_report_is_honest_and_embeds_curated_map():
    gen = ReportGenerator()
    ver = {"total_verified": 1, "verified_hypotheses": [{"hypothesis": "H", "confidence": 0.7, "feasibility": "high"}], "verification_stats": {}}
    rep = gen.generate_report({"total_found": 1, "papers": []}, {"summary": {}}, {"network_stats": {"num_nodes": 0, "num_edges": 0, "num_clusters": 0}}, {"total_gaps": 1, "gaps": [{"type": "limitation", "description": "d", "score": 5}], "gap_categories": {}},
                              {"total_hypotheses": 1}, ver, "topic")
    md = gen.generate_markdown(rep)
    assert "verified against existing literature" not in md and "Confidence:" not in md and "Heuristic score" in md
    cur = gen.generate_markdown(rep, curated_gap_map="# Research gap map\n\n## My group (your group)  ·  1 gap(s)\n- thing\n")
    assert "My group" in cur and "Organised interactively" in cur
    assert "Research gap map" not in cur.split("## 3.")[1]
