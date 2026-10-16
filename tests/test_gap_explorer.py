import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from fixtures import PAPERS
from autoresearch.modules.gap_explorer import (ExplorerState, GapExplorerService, align_to_previous, cluster_gaps,
                                               diff_assignments, extract_gap_statements, gaps_from_agent_output, layout_2d)


@pytest.fixture
def gaps():
    return extract_gap_statements(PAPERS)


def test_extraction_has_provenance_and_skips_cueless(gaps):
    assert len(gaps) >= 8
    titles = {g["paper_title"] for g in gaps}
    assert "No cues" not in titles and "Empty" not in titles
    for g in gaps:
        assert g["paper_title"] and g["text"] and g["cues"]
    assert len({g["id"] for g in gaps}) == len(gaps)


def test_extraction_empty_input():
    assert extract_gap_statements([]) == []
    assert extract_gap_statements([{"title": "t", "abstract": None}]) == []


def test_cluster_counts_and_ids(gaps):
    c = cluster_gaps(gaps, 3)
    assert set(c.assign) == {g["id"] for g in gaps}
    assert set(c.assign.values()) == set(c.labels)
    assert len(set(c.assign.values())) == 3
    # C1 is the largest cluster
    sizes = {k: list(c.assign.values()).count(k) for k in c.labels}
    assert sizes["C1"] == max(sizes.values())


def test_k_is_clamped(gaps):
    assert len(set(cluster_gaps(gaps, 999).assign.values())) == len(gaps)
    assert len(set(cluster_gaps(gaps, 0).assign.values())) == 1


def test_clustering_is_deterministic(gaps):
    assert cluster_gaps(gaps, 3).assign == cluster_gaps(gaps, 3).assign


def test_clusters_are_semantically_coherent(gaps):
    # the two 'computationally expensive / latency / memory' sentences should share a cluster
    c = cluster_gaps(gaps, 4)
    comp = [g["id"] for g in gaps if g["paper_title"] == "Paper B"]
    assert len({c.assign[i] for i in comp}) == 1


def test_pinned_gaps_survive_recluster(gaps):
    st = ExplorerState.create(gaps, k=3)
    ids = [g["id"] for g in gaps[:2]]
    st.pin_to_group(ids, "My data problem")
    for k in (2, 4, 5, 3):
        st.recluster(k)
        assert all(st.group_of(i) == "My data problem" for i in ids)


def test_pinned_gaps_excluded_from_ai_clusters(gaps):
    st = ExplorerState.create(gaps, k=3)
    ids = [g["id"] for g in gaps[:2]]
    st.pin_to_group(ids, "Mine")
    prop = st.propose_recluster(3)
    assert not set(ids) & set(prop.assign)


def test_preview_does_not_mutate_state(gaps):
    st = ExplorerState.create(gaps, k=3)
    before = dict(st.ai_assign), dict(st.ai_labels), st.k, list(st.history)
    st.preview_recluster(5)
    assert (dict(st.ai_assign), dict(st.ai_labels), st.k, list(st.history)) == before


def test_preview_matches_apply(gaps):
    st = ExplorerState.create(gaps, k=3)
    predicted = st.preview_recluster(5)
    moved = st.recluster(5)
    assert moved == len(predicted)


def test_same_k_moves_nothing(gaps):
    st = ExplorerState.create(gaps, k=3)
    assert st.preview_recluster(3) == []


def test_alignment_keeps_ids_stable():
    old = {"a": "C1", "b": "C1", "c": "C2", "d": "C2"}
    from autoresearch.modules.gap_explorer import Clustering
    new = Clustering(assign={"a": "C2", "b": "C2", "c": "C1", "d": "C1"}, labels={"C1": "x", "C2": "y"})
    aligned = align_to_previous(new, old)
    assert diff_assignments(old, aligned.assign) == []


def test_remove_and_restore(gaps):
    st = ExplorerState.create(gaps, k=3)
    gid = gaps[0]["id"]
    st.pin_to_group([gid], "G")
    st.remove([gid])
    assert gid not in {g["id"] for g in st.active}
    assert gid not in st.pinned
    st.recluster(2)  # must not crash or reassign the removed gap
    assert gid not in st.propose_recluster(2).assign
    st.restore_all()
    assert gid in {g["id"] for g in st.active}


def test_rename_cluster_and_cleanup(gaps):
    st = ExplorerState.create(gaps, k=3)
    st.rename_cluster("C1", "Big theme")
    assert any(name == "Big theme" for name in st.groups())
    st.recluster(1)  # C2/C3 vanish
    assert set(st.renamed) <= set(st.ai_assign.values())


def test_pin_validation(gaps):
    st = ExplorerState.create(gaps, k=3)
    with pytest.raises(ValueError):
        st.pin_to_group([gaps[0]["id"]], "   ")
    assert st.pin_to_group(["nope"], "X") == 0


def test_unpin_returns_to_ai(gaps):
    st = ExplorerState.create(gaps, k=3)
    gid = gaps[0]["id"]
    st.pin_to_group([gid], "Mine")
    assert st.unpin([gid]) == 1
    assert st.group_of(gid) != "Mine"


def test_every_active_gap_in_exactly_one_group(gaps):
    st = ExplorerState.create(gaps, k=3)
    st.pin_to_group([gaps[1]["id"]], "Mine")
    st.remove([gaps[2]["id"]])
    st.recluster(4)
    flat = [g["id"] for items in st.groups().values() for g in items]
    assert sorted(flat) == sorted(g["id"] for g in st.active)
    assert len(flat) == len(set(flat))


def test_tiny_inputs():
    one = extract_gap_statements([{"title": "t", "year": 1, "abstract":
        "However this approach remains computationally expensive for large datasets."}])
    st = ExplorerState.create(one, k=3)
    assert st.k == 1 and len(st.groups()) == 1
    assert layout_2d([g["text"] for g in one]).shape == (1, 2)
    empty = ExplorerState.create([], k=3)
    assert empty.groups() == {}


def test_layout_shape(gaps):
    xy = layout_2d([g["text"] for g in gaps])
    assert xy.shape == (len(gaps), 2) and np.isfinite(xy).all()


def test_markdown_export(gaps):
    st = ExplorerState.create(gaps, k=3)
    st.toggle_star(gaps[0]["id"])
    md = st.to_markdown()
    assert "★" in md and "Source:" in md


# ----------------------------------------------------------------- service / adapter
def _agent_gaps(n=12):
    base = [("limitation", "Common data limitation mentioned in 3 papers"),
            ("stated_limitation", "Latency is a bottleneck in clinical settings."),
            ("stated_limitation", "Annotated medical data is scarce and labeling is expensive."),
            ("evaluation", "Segmentation research rarely uses dice metrics"),
            ("stated_limitation", "Generalization across hospitals remains underexplored.")]
    out = []
    for i in range(n):
        t, d = base[i % len(base)]
        out.append({"type": t, "description": f"{d} (variant {i})", "paper_title": f"P{i}" if t == "stated_limitation" else None,
                    "year": 2020 + i % 5 if t == "stated_limitation" else None, "theme": "Evaluation"})
    return out


def test_adapter_keeps_provenance_and_skips_empty():
    items = gaps_from_agent_output(_agent_gaps(5) + [{"type": "x", "description": "  "}])
    assert len(items) == 5 and all(g["id"].startswith("g") for g in items)
    assert items[0]["paper_title"] == "Cross-paper analysis" and items[1]["paper_title"] == "P1"


def test_service_requires_two_gaps():
    assert GapExplorerService().create(_agent_gaps(1)) is None


def test_service_view_and_actions():
    svc = GapExplorerService(); sid = svc.create(_agent_gaps(12), topic="t")
    v = svc.view(sid)
    assert v["counts"]["gaps"] == 12 and all(0 <= i["x"] <= 1 and 0 <= i["y"] <= 1 for i in v["items"])
    v = svc.group(sid, ["g1", "g2", "bogus"], "  Mine ")
    assert v["counts"]["pinned"] == 2 and any(g["name"] == "Mine" and g["human"] for g in v["groups"])
    before = svc.view(sid)
    pv = svc.preview(sid, 6)
    assert svc.view(sid)["history"] == before["history"] and pv["human_grouped"] == 2
    out = svc.recluster(sid, 6)
    assert {i["id"]: i["group"] for i in out["items"] if i["id"] in ("g1", "g2")} == {"g1": "Mine", "g2": "Mine"}
    assert svc.toggle_star(sid, ["g3"])["counts"]["starred"] == 1
    assert svc.remove(sid, ["g4"])["counts"]["removed"] == 1
    assert svc.restore_all(sid)["counts"]["removed"] == 0
    assert svc.unpin(sid, ["g1"])["counts"]["pinned"] == 1


def test_service_validation_and_clamping():
    svc = GapExplorerService(); sid = svc.create(_agent_gaps(6))
    for bad in ("", "   ", "x" * 81):
        try:
            svc.group(sid, ["g1"], bad); assert False
        except ValueError:
            pass
    try:
        svc.group(sid, ["nope"], "X"); assert False
    except ValueError:
        pass
    assert svc.preview(sid, 999)["k"] == svc.view(sid)["max_k"]
    assert svc.recluster(sid, 0)["k"] == 1
    try:
        svc.rename_cluster(sid, "C99", "x"); assert False
    except ValueError:
        pass
    try:
        svc.view("missing"); assert False
    except KeyError:
        pass


def test_service_evicts_oldest_sessions():
    svc = GapExplorerService(); first = svc.create(_agent_gaps(4))
    for _ in range(GapExplorerService.MAX_SESSIONS):
        svc.create(_agent_gaps(4))
    try:
        svc.view(first); assert False
    except KeyError:
        pass
