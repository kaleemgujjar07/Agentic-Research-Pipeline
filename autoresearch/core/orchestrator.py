"""AutoResearch Orchestrator v3 - Integrated with all specialized modules."""
import sys
import os
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# Core agents
from autoresearch.agents.literature_discovery import LiteratureDiscoveryAgent
from autoresearch.agents.deep_reading import DeepReadingAgent
from autoresearch.agents.citation_network import CitationNetworkBuilder
from autoresearch.agents.gap_detection import GapDetectionAgent
from autoresearch.agents.hypothesis_generation import HypothesisGenerationAgent
from autoresearch.agents.critic_verification import CriticVerificationAgent
from autoresearch.agents.report_generator import ReportGenerator

# Specialized modules
from autoresearch.modules.security import SecurityModule
from autoresearch.modules.vision import VisionModule
from autoresearch.modules.datascience import DataScienceModule
from autoresearch.modules.iot import IoTModule

class AutoResearchOrchestratorV3:
    """
    Enhanced orchestrator with security, vision, data science, and IoT capabilities.
    """

    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.security = SecurityModule()
        self.vision = VisionModule()
        self.datascience = DataScienceModule()
        self.iot = IoTModule(mode=mode)

        # Core agents
        self.agents = {
            "discovery": LiteratureDiscoveryAgent(),
            "reading": DeepReadingAgent(),
            "network": CitationNetworkBuilder(),
            "gap": GapDetectionAgent(),
            "hypothesis": HypothesisGenerationAgent(),
            "critic": CriticVerificationAgent(),
            "report": ReportGenerator()
        }

        # Register agents in IoT module
        for name, agent in self.agents.items():
            self.iot.register_agent(name, ["research", "analysis"])

        self.pipeline_status = {}
        self.results_cache = {}

    def run(self, topic: str, max_papers: int = 10, sources: List[str] = None,
            client_id: str = "default") -> Dict[str, Any]:
        """
        Execute full pipeline with all modules.
        """
        start_time = time.time()

        # Security: Input sanitization
        security_report = self.security.sanitize_query(topic)
        if not security_report["is_safe"]:
            return {
                "error": "Input blocked by security module",
                "findings": security_report["security_findings"],
                "sanitized": security_report["sanitized"]
            }

        clean_topic = security_report["sanitized"]

        # IoT: Start distributed pipeline simulation
        self.iot.simulate_distributed_pipeline(clean_topic)

        print(f"\n{'='*60}")
        print(f"  AUTORESEARCH v3.0 - UNIFIED RESEARCH SYSTEM")
        print(f"  Topic: {clean_topic}")
        print(f"  Mode: {self.mode}")
        print(f"{'='*60}\n")

        # Stage 1: Literature Discovery
        self._update_status("discovery", "running")
        discovery = self.agents["discovery"].execute(clean_topic, max_papers, sources)
        self._update_status("discovery", "complete", discovery)
        self.iot.send_heartbeat("LiteratureDiscovery")

        if discovery.get("total_found", 0) == 0:
            return {"error": "No papers found", "discovery": discovery}

        papers = discovery.get("papers", [])

        # Stage 2: Deep Reading
        self._update_status("reading", "running")
        reading = self.agents["reading"].execute(papers)
        self._update_status("reading", "complete", reading)
        self.iot.send_heartbeat("DeepReading")

        analyses = reading.get("analyses", [])

        # Stage 3: Citation Network
        self._update_status("network", "running")
        network = self.agents["network"].execute(papers, analyses)
        self._update_status("network", "complete", network)
        self.iot.send_heartbeat("CitationNetwork")

        # Vision: Generate network visualization data
        network_viz = self.vision.generate_citation_network_viz(network)
        trend_chart = self.vision.generate_trend_chart(
            reading.get("summary", {}).get("year_distribution", {})
        )

        # Stage 4: Gap Detection
        self._update_status("gap", "running")
        gaps = self.agents["gap"].execute(analyses, network)
        self._update_status("gap", "complete", gaps)
        self.iot.send_heartbeat("GapDetection")

        # Data Science: Statistical analysis
        gap_stats = self.datascience.statistical_gap_significance(gaps.get("gaps", []))
        citation_metrics = self.datascience.calculate_citation_impact_metrics(papers)
        trend_forecast = self.datascience.forecast_research_trend(
            reading.get("summary", {}).get("year_distribution", {})
        )
        health_score = self.datascience.generate_research_health_score(analyses)

        # Stage 5: Hypothesis Generation
        self._update_status("hypothesis", "running")
        hypotheses = self.agents["hypothesis"].execute(gaps.get("gaps", []), analyses)
        self._update_status("hypothesis", "complete", hypotheses)
        self.iot.send_heartbeat("HypothesisGeneration")

        # Stage 6: Verification
        self._update_status("critic", "running")
        verification = self.agents["critic"].execute(hypotheses.get("hypotheses", []), analyses)
        self._update_status("critic", "complete", verification)
        self.iot.send_heartbeat("CriticVerification")

        # Stage 7: Report
        self._update_status("report", "running")
        report = self.agents["report"].execute(
            discovery, reading, network, gaps, hypotheses, verification, clean_topic
        )
        self._update_status("report", "complete", report)
        self.iot.send_heartbeat("ReportGenerator")

        elapsed = time.time() - start_time

        # Compile full result with all modules
        full_result = {
            "topic": clean_topic,
            "mode": self.mode,
            "pipeline_stats": {
                "total_time_seconds": round(elapsed, 2),
                "papers_found": len(papers),
                "gaps_detected": gaps.get("total_gaps", 0),
                "hypotheses_generated": hypotheses.get("total_hypotheses", 0),
                "hypotheses_verified": verification.get("total_verified", 0)
            },
            "security": {
                "input_sanitized": not security_report["is_safe"],
                "entropy": security_report["entropy"],
                "findings": security_report["security_findings"]
            },
            "datascience": {
                "citation_metrics": citation_metrics,
                "gap_statistics": gap_stats,
                "trend_forecast": trend_forecast,
                "research_health": health_score
            },
            "vision": {
                "network_visualization": network_viz,
                "trend_chart": trend_chart
            },
            "iot": {
                "agent_health": self.iot.check_agent_health(),
                "message_log": self.iot.get_message_log(limit=10),
                "deployment_mode": self.iot.edge_mode_summary()
            },
            "stages": {
                "discovery": discovery,
                "reading": reading,
                "network": network,
                "gap": gaps,
                "hypothesis": hypotheses,
                "verification": verification
            },
            "report": report,
            "markdown_report": self.agents["report"].generate_markdown(report),
            "pipeline_status": self.pipeline_status
        }

        self.results_cache[clean_topic] = full_result

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
        print(f"  Papers: {len(papers)} | Gaps: {gaps.get('total_gaps', 0)} | Verified: {verification.get('total_verified', 0)}")
        print(f"  Health Score: {health_score.get('health_score', 0)}")
        print(f"{'='*60}\n")

        return full_result

    def _update_status(self, stage: str, status: str, data: Dict = None):
        """Update pipeline status for real-time monitoring."""
        self.pipeline_status[stage] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }

    def run_adversarial_tests(self, topic: str) -> Dict[str, Any]:
        """Run security adversarial tests on the pipeline."""
        def dummy_pipeline(q):
            return {"query": q, "status": "processed"}

        return self.security.adversarial_test_suite(dummy_pipeline)

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "pipeline_status": self.pipeline_status,
            "agent_health": self.iot.check_agent_health(),
            "mode": self.mode,
            "cached_results": list(self.results_cache.keys())
        }
