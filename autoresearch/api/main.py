"""FastAPI application: 7-agent research pipeline + interactive Gap Explorer.

Run:  uvicorn autoresearch.api.main:app --reload
"""
from pathlib import Path
from typing import Callable, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from autoresearch.core.orchestrator import AutoResearchOrchestratorV3

app = FastAPI(
    title="AutoResearch v3.0",
    description="7-agent research pipeline (discovery, reading, citation network, gap detection, "
                "hypotheses, verification, report) with an interactive human-AI Gap Explorer.",
    version="3.1.0",
)

orchestrator = AutoResearchOrchestratorV3(mode="full")
_DASHBOARD = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- models
class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    max_papers: int = Field(default=10, ge=1, le=20)
    sources: List[str] = ["arxiv", "semantic_scholar"]


class KRequest(BaseModel):
    k: int = Field(ge=1, le=20)


class GroupRequest(BaseModel):
    ids: List[str] = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=80)


class IdsRequest(BaseModel):
    ids: List[str] = Field(min_length=1, max_length=200)


class RenameRequest(BaseModel):
    cluster_id: str = Field(min_length=1, max_length=40)
    name: str = Field(default="", max_length=80)  # empty string resets to the AI's name


def _explorer(fn: Callable, *args):
    """Run an explorer action; map domain errors to HTTP errors."""
    try:
        return fn(*args)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown explorer session (the server may have restarted).")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------------------------------------------------------- pipeline
@app.post("/api/research")
def start_research(request: ResearchRequest):
    """Run the full 7-agent pipeline. Synchronous on purpose: FastAPI runs it in a worker thread."""
    result = orchestrator.run(topic=request.topic, max_papers=request.max_papers, sources=request.sources)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result)


@app.get("/api/status")
def get_status():
    return orchestrator.get_status()


@app.get("/api/adversarial-test")
def adversarial_test():
    return orchestrator.run_adversarial_tests("test")


# ----------------------------------------------------------------------------- gap explorer
@app.get("/api/explorer/{sid}")
def explorer_state(sid: str):
    """Current state of a Gap Explorer session (gaps, groups, map coordinates, history)."""
    return _explorer(orchestrator.explorers.view, sid)


@app.post("/api/explorer/{sid}/preview")
def explorer_preview(sid: str, body: KRequest):
    """What *would* move if the AI re-clustered with k clusters. Changes nothing."""
    return _explorer(orchestrator.explorers.preview, sid, body.k)


@app.post("/api/explorer/{sid}/recluster")
def explorer_recluster(sid: str, body: KRequest):
    """AI re-clusters the gaps the user has not grouped themselves."""
    return _explorer(orchestrator.explorers.recluster, sid, body.k)


@app.post("/api/explorer/{sid}/group")
def explorer_group(sid: str, body: GroupRequest):
    """User groups gaps under their own name (the AI will never move these)."""
    return _explorer(orchestrator.explorers.group, sid, body.ids, body.name)


@app.post("/api/explorer/{sid}/unpin")
def explorer_unpin(sid: str, body: IdsRequest):
    return _explorer(orchestrator.explorers.unpin, sid, body.ids)


@app.post("/api/explorer/{sid}/rename")
def explorer_rename(sid: str, body: RenameRequest):
    return _explorer(orchestrator.explorers.rename_cluster, sid, body.cluster_id, body.name)


@app.post("/api/explorer/{sid}/star")
def explorer_star(sid: str, body: IdsRequest):
    return _explorer(orchestrator.explorers.toggle_star, sid, body.ids)


@app.post("/api/explorer/{sid}/remove")
def explorer_remove(sid: str, body: IdsRequest):
    return _explorer(orchestrator.explorers.remove, sid, body.ids)


@app.post("/api/explorer/{sid}/restore")
def explorer_restore(sid: str):
    return _explorer(orchestrator.explorers.restore_all, sid)


@app.get("/api/explorer/{sid}/export", response_class=PlainTextResponse)
def explorer_export(sid: str):
    """The gap map as Markdown."""
    return PlainTextResponse(_explorer(orchestrator.explorers.export_markdown, sid),
                             headers={"Content-Disposition": 'attachment; filename="research_gap_map.md"'})


@app.get("/api/explorer/{sid}/report", response_class=PlainTextResponse)
def explorer_report(sid: str):
    """The full research report, with the gap section reflecting the user's own organisation."""
    return PlainTextResponse(_explorer(orchestrator.curated_report_markdown, sid),
                             headers={"Content-Disposition": 'attachment; filename="research_report.md"'})


# ----------------------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_DASHBOARD)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
