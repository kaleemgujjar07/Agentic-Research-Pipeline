"""Pipeline + FastAPI integration, fully offline (paper APIs mocked)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from fastapi.testclient import TestClient
from mocks import offline_apis

TOPIC = "vision transformers medical imaging"


@pytest.fixture(scope="module")
def client():
    from autoresearch.api.main import app
    with offline_apis():
        yield TestClient(app)


@pytest.fixture(scope="module")
def run(client):
    r = client.post("/api/research", json={"topic": TOPIC, "max_papers": 8})
    assert r.status_code == 200, r.text
    return r.json()


def test_all_seven_agents_ran(run):
    assert run["pipeline_stats"]["papers_found"] >= 5
    for stage in ("discovery", "reading", "network", "gap", "hypothesis", "verification"):
        assert stage in run["stages"]
    assert "ReportGenerator" in run["report"]["metadata"]["agents"] and run["markdown_report"]
    assert run["stages"]["reading"]["summary"]["year_distribution"]


def test_explorer_built_from_agent_gaps(run):
    ge = run["gap_explorer"]
    assert ge["session_id"] and ge["counts"]["gaps"] == run["stages"]["gap"]["total_gaps"]
    assert any(i["kind"] == "stated_limitation" and i["paper_title"] for i in ge["items"])


def test_full_interaction_over_http(client, run):
    sid = run["gap_explorer"]["session_id"]; B = f"/api/explorer/{sid}"
    ids = [i["id"] for i in client.get(B).json()["items"]][:3]
    pv = client.post(f"{B}/preview", json={"k": 3}).json()
    assert {"moves", "moved", "ai_grouped", "human_grouped"} <= set(pv)
    g = client.post(f"{B}/group", json={"ids": ids, "name": "Mine"}).json()
    assert g["counts"]["pinned"] == 3
    rc = client.post(f"{B}/recluster", json={"k": 2}).json()
    assert {i["id"]: i["group"] for i in rc["items"] if i["id"] in ids} == {i: "Mine" for i in ids}
    assert client.post(f"{B}/star", json={"ids": ids[:1]}).json()["counts"]["starred"] == 1
    assert client.post(f"{B}/remove", json={"ids": ids[1:2]}).json()["counts"]["removed"] == 1
    assert client.post(f"{B}/restore").json()["counts"]["removed"] == 0
    assert client.post(f"{B}/unpin", json={"ids": ids}).json()["counts"]["pinned"] == 0
    cid = client.get(B).json()["ai_clusters"][0]["id"]
    ren = client.post(f"{B}/rename", json={"cluster_id": cid, "name": "Renamed theme"}).json()
    assert any(g["name"] == "Renamed theme" for g in ren["groups"])


def test_exports_reflect_user_organisation(client, run):
    sid = run["gap_explorer"]["session_id"]; B = f"/api/explorer/{sid}"
    ids = [i["id"] for i in client.get(B).json()["items"]][:2]
    client.post(f"{B}/group", json={"ids": ids, "name": "Curated by me"})
    gm = client.get(f"{B}/export"); rep = client.get(f"{B}/report")
    assert "Curated by me" in gm.text and "attachment" in gm.headers["content-disposition"]
    assert "Curated by me" in rep.text and "Organised interactively" in rep.text


@pytest.mark.parametrize("path,body,code", [
    ("preview", {"k": 0}, 422), ("preview", {"k": 99}, 422), ("preview", {}, 422),
    ("group", {"ids": [], "name": "x"}, 422), ("group", {"ids": ["g1"], "name": ""}, 422),
    ("group", {"ids": ["g1"], "name": "x" * 81}, 422), ("group", {"ids": ["zzz"], "name": "x"}, 400),
    ("rename", {"cluster_id": "C99", "name": "x"}, 400),
])
def test_validation_errors(client, run, path, body, code):
    sid = run["gap_explorer"]["session_id"]
    assert client.post(f"/api/explorer/{sid}/{path}", json=body).status_code == code


def test_unknown_session_is_404(client):
    assert client.get("/api/explorer/nope").status_code == 404
    assert client.post("/api/explorer/nope/restore").status_code == 404
    assert client.get("/api/explorer/nope/report").status_code == 404


def test_research_input_validation(client):
    assert client.post("/api/research", json={"topic": "ab"}).status_code == 422
    assert client.post("/api/research", json={"topic": TOPIC, "max_papers": 500}).status_code == 422


def test_dashboard_ships_the_explorer(client):
    html = client.get("/").text
    assert "Gap Explorer" in html
    assert "const esc =" in html and "const safeUrl =" in html     # output escaping + http(s)-only links
    assert "esc((data.markdown_report" in html                     # report preview was unescaped in the old dashboard


def test_result_is_json_serialisable(run):
    json.dumps(run)
