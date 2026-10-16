"""End-to-end UI test: runs the real app.py headlessly (7-agent pipeline, mocked paper APIs)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from mocks import offline_apis
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def _run():
    with offline_apis():
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        at.text_input[0].set_value("vision transformers medical imaging")
        at.slider[0].set_value(8)
        [b for b in at.button if "Run the 7-agent" in b.label][0].click()
        at.run()
    return at


def _state(at):
    orch = at.session_state["orch"]
    sid = at.session_state["result"]["gap_explorer"]["session_id"]
    return orch.explorers.get_state(sid)


def _btn(at, text):
    return [b for b in at.button if text in b.label][0]


def test_runs_seven_agents_and_shows_explorer_without_errors():
    at = _run()
    assert not at.exception, at.exception
    assert any("Gap Explorer" in h.value for h in at.header)
    assert any("Papers retrieved" in h.value for h in at.header)
    assert _state(at).active


def test_results_survive_widget_interaction():
    """Regression: clicking inside the explorer must not wipe the pipeline results."""
    at = _run(); n = len(_state(at).active)
    at.multiselect[0].set_value([_state(at).active[0]["id"]]).run()
    assert not at.exception, at.exception
    assert any("Papers retrieved" in h.value for h in at.header) and len(_state(at).active) == n


def test_group_selected_pins():
    at = _run()
    ids = [g["id"] for g in _state(at).active[:2]]
    at.multiselect[0].set_value(ids).run()
    [t for t in at.text_input if "Group name" in t.label][0].set_value("My group").run()
    _btn(at, "Group selected").click().run()
    assert not at.exception, at.exception
    assert all(_state(at).group_of(i) == "My group" for i in ids)


def test_recluster_preview_then_apply_keeps_pins():
    at = _run()
    ids = [g["id"] for g in _state(at).active[:2]]
    _state(at).pin_to_group(ids, "Mine"); at.run()
    cur = _state(at).k
    at.slider[1].set_value(cur + 1 if cur < 6 else cur - 1).run()
    assert not at.exception, at.exception
    assert any("Preview" in i.value for i in at.info)
    _btn(at, "Apply AI").click().run()
    assert not at.exception, at.exception
    assert _state(at).k != cur and all(_state(at).group_of(i) == "Mine" for i in ids)


def test_report_download_reflects_grouping():
    at = _run()
    ids = [g["id"] for g in _state(at).active[:2]]
    _state(at).pin_to_group(ids, "ReportCheck"); at.run()
    assert not at.exception, at.exception
    orch = at.session_state["orch"]; sid = at.session_state["result"]["gap_explorer"]["session_id"]
    assert "ReportCheck" in orch.curated_report_markdown(sid)


def test_short_topic_is_rejected():
    with offline_apis():
        at = AppTest.from_file(APP, default_timeout=60); at.run()
        at.text_input[0].set_value("ab")
        [b for b in at.button if "Run the 7-agent" in b.label][0].click(); at.run()
    assert at.error and "valid research topic" in at.error[0].value
