"""FastAPI Application with Interactive Dashboard."""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import json

from autoresearch.core.orchestrator import AutoResearchOrchestratorV3

app = FastAPI(
    title="AutoResearch v3.0",
    description="Unified Multi-Agent Research System with Security, Vision, Data Science & IoT",
    version="3.0.0"
)

orchestrator = AutoResearchOrchestratorV3(mode="full")

class ResearchRequest(BaseModel):
    topic: str
    max_papers: int = 10
    sources: List[str] = ["arxiv", "semantic_scholar"]

@app.post("/api/research")
async def start_research(request: ResearchRequest):
    """Start a new research pipeline."""
    result = orchestrator.run(
        topic=request.topic,
        max_papers=request.max_papers,
        sources=request.sources
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return JSONResponse(content=result)

@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return orchestrator.get_status()

@app.get("/api/adversarial-test")
async def adversarial_test():
    """Run security adversarial tests."""
    return orchestrator.run_adversarial_tests("test")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoResearch v3.0 - Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
            .header { background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 2rem; text-align: center; }
            .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
            .header p { color: #93c5fd; }
            .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
            .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
            .card h3 { color: #60a5fa; margin-bottom: 1rem; font-size: 1.2rem; }
            .metric { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #334155; }
            .metric:last-child { border-bottom: none; }
            .metric-label { color: #94a3b8; }
            .metric-value { color: #34d399; font-weight: bold; }
            .input-section { background: #1e293b; padding: 2rem; border-radius: 12px; margin-top: 2rem; }
            input[type="text"] { width: 100%; padding: 1rem; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: white; font-size: 1rem; margin-bottom: 1rem; }
            button { background: #3b82f6; color: white; padding: 1rem 2rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; }
            button:hover { background: #2563eb; }
            .status-bar { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
            .status-pill { padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; }
            .status-pill.running { background: #f59e0b; color: #1e293b; }
            .status-pill.complete { background: #10b981; color: white; }
            .status-pill.pending { background: #475569; color: #94a3b8; }
            #results { margin-top: 2rem; }
            .module-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; margin: 0.25rem; }
            .badge-security { background: #ef4444; }
            .badge-vision { background: #8b5cf6; }
            .badge-ds { background: #06b6d4; }
            .badge-iot { background: #f97316; }
            pre { background: #0f172a; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 AutoResearch v3.0</h1>
            <p>Unified Multi-Agent Research System | Security • Vision • Data Science • IoT</p>
        </div>

        <div class="container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <span class="module-badge badge-security">🔒 Security</span>
                <span class="module-badge badge-vision">👁️ Vision</span>
                <span class="module-badge badge-ds">📊 Data Science</span>
                <span class="module-badge badge-iot">📡 IoT</span>
            </div>

            <div class="input-section">
                <h3>Start Research</h3>
                <input type="text" id="topic" placeholder="e.g., transformer architectures for medical image segmentation" />
                <button onclick="runResearch()">🚀 Run Pipeline</button>
                <div id="loading" style="display:none; margin-top: 1rem; color: #60a5fa;">Processing... This may take 30-60 seconds</div>
            </div>

            <div id="results"></div>

            <div class="grid" id="modules" style="display:none;">
                <div class="card">
                    <h3>🔒 Security Module</h3>
                    <div id="security-content"></div>
                </div>
                <div class="card">
                    <h3>📊 Data Science</h3>
                    <div id="ds-content"></div>
                </div>
                <div class="card">
                    <h3>👁️ Vision Module</h3>
                    <div id="vision-content"></div>
                </div>
                <div class="card">
                    <h3>📡 IoT Module</h3>
                    <div id="iot-content"></div>
                </div>
            </div>
        </div>

        <script>
            async function runResearch() {
                const topic = document.getElementById('topic').value;
                if (!topic) return alert('Please enter a topic');

                document.getElementById('loading').style.display = 'block';
                document.getElementById('modules').style.display = 'none';

                try {
                    const response = await fetch('/api/research', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({topic: topic, max_papers: 8})
                    });

                    const data = await response.json();
                    document.getElementById('loading').style.display = 'none';

                    if (data.error) {
                        document.getElementById('results').innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
                        return;
                    }

                    displayResults(data);
                } catch (e) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').innerHTML = `<div style="color: #ef4444;">Error: ${e.message}</div>`;
                }
            }

            function displayResults(data) {
                document.getElementById('modules').style.display = 'grid';

                // Security
                const sec = data.security || {};
                document.getElementById('security-content').innerHTML = `
                    <div class="metric"><span class="metric-label">Input Safe</span><span class="metric-value">${!sec.input_sanitized}</span></div>
                    <div class="metric"><span class="metric-label">Entropy</span><span class="metric-value">${sec.entropy || 'N/A'}</span></div>
                    <div class="metric"><span class="metric-label">Findings</span><span class="metric-value">${(sec.findings || []).length}</span></div>
                `;

                // Data Science
                const ds = data.datascience || {};
                const health = ds.research_health || {};
                const metrics = ds.citation_metrics || {};
                document.getElementById('ds-content').innerHTML = `
                    <div class="metric"><span class="metric-label">Health Score</span><span class="metric-value">${health.health_score || 0}/100</span></div>
                    <div class="metric"><span class="metric-label">Mean Citations</span><span class="metric-value">${metrics.mean_citations || 0}</span></div>
                    <div class="metric"><span class="metric-label">H-Index</span><span class="metric-value">${metrics.h_index || 0}</span></div>
                    <div class="metric"><span class="metric-label">Trend</span><span class="metric-value">${(ds.trend_forecast || {}).trend || 'N/A'}</span></div>
                `;

                // Vision
                const viz = data.vision || {};
                const netViz = viz.network_visualization || {};
                document.getElementById('vision-content').innerHTML = `
                    <div class="metric"><span class="metric-label">Network Nodes</span><span class="metric-value">${netViz.num_nodes || 0}</span></div>
                    <div class="metric"><span class="metric-label">Network Links</span><span class="metric-value">${netViz.num_links || 0}</span></div>
                    <div class="metric"><span class="metric-label">Chart Type</span><span class="metric-value">${(viz.trend_chart || {}).chart_type || 'N/A'}</span></div>
                `;

                // IoT
                const iot = data.iot || {};
                const health_iot = iot.agent_health || {};
                document.getElementById('iot-content').innerHTML = `
                    <div class="metric"><span class="metric-label">Healthy Agents</span><span class="metric-value">${(health_iot.healthy_agents || []).length}/7</span></div>
                    <div class="metric"><span class="metric-label">Mode</span><span class="metric-value">${(iot.deployment_mode || {}).mode || 'N/A'}</span></div>
                    <div class="metric"><span class="metric-label">Messages</span><span class="metric-value">${(iot.message_log || []).length}</span></div>
                `;

                // Pipeline stats
                const stats = data.pipeline_stats || {};
                document.getElementById('results').innerHTML = `
                    <div class="card" style="margin-bottom: 2rem;">
                        <h3>📈 Pipeline Results</h3>
                        <div class="metric"><span class="metric-label">Time</span><span class="metric-value">${stats.total_time_seconds}s</span></div>
                        <div class="metric"><span class="metric-label">Papers</span><span class="metric-value">${stats.papers_found}</span></div>
                        <div class="metric"><span class="metric-label">Gaps</span><span class="metric-value">${stats.gaps_detected}</span></div>
                        <div class="metric"><span class="metric-label">Verified Hypotheses</span><span class="metric-value">${stats.hypotheses_verified}</span></div>
                    </div>
                    <div class="card">
                        <h3>📝 Report Preview</h3>
                        <pre>${(data.markdown_report || '').substring(0, 1500)}...</pre>
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
