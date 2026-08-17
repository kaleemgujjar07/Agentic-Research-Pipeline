"""Hypothesis Generation Agent v2 - LLM-powered hypothesis creation."""
from typing import Dict, List, Any
from autoresearch.agents.base_agent import BaseAgent

class HypothesisGenerationAgent(BaseAgent):
    def __init__(self, db_session=None, llm_client=None):
        super().__init__("HypothesisGeneration", db_session, llm_client)

    def generate_hypotheses(self, gaps: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        self.start_timer()
        self.log(f"Generating hypotheses from {len(gaps)} gaps")

        hypotheses = []
        for gap in gaps[:12]:
            h = self._template_hypothesis(gap)
            hypotheses.append(h)

        scored = self._score_hypotheses(hypotheses)
        duration = self.stop_timer()

        return {
            "total_hypotheses": len(scored),
            "hypotheses": scored[:12],
            "by_type": self._group_by_type(scored),
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

    def _template_hypothesis(self, gap: Dict) -> Dict:
        templates = {
            "methodological": f"Apply {gap.get('method', 'novel methods')} to {gap.get('target_domain', 'new domain')} to address existing limitations.",
            "cross_domain": f"Bridge {gap.get('cluster_1_domains', ['domain A'])} and {gap.get('cluster_2_domains', ['domain B'])} through unified framework.",
            "limitation": f"Address {gap.get('theme', 'key limitation')} in {gap.get('domain', 'the field')} using improved methodology.",
            "temporal": f"Revitalize {gap.get('domain', 'the area')} research with modern techniques and comprehensive evaluation.",
            "evaluation": f"Introduce rigorous multi-metric evaluation for {gap.get('domain', 'the domain')} to drive methodological improvements."
        }
        return {
            "hypothesis": templates.get(gap["type"], f"Investigate: {gap['description']}"),
            "gap_type": gap["type"],
            "testability": "medium",
            "novelty": "medium",
            "impact": "medium",
            "suggested_methodology": [],
            "expected_outcome": "",
            "overall_score": 0,
            "source_gap_score": gap.get("score", 0)
        }

    def _score_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        for h in hypotheses:
            score = h.get("source_gap_score", 0) * 0.5
            score += {"high": 5, "medium": 3, "low": 1}.get(h.get("testability"), 3)
            score += {"high": 5, "medium": 3, "low": 1}.get(h.get("novelty"), 3)
            score += {"high": 5, "medium": 3, "low": 1}.get(h.get("impact"), 3)
            h["overall_score"] = round(score, 2)
        return sorted(hypotheses, key=lambda x: x["overall_score"], reverse=True)

    def _group_by_type(self, hypotheses: List[Dict]) -> Dict[str, int]:
        groups = {}
        for h in hypotheses:
            gt = h.get("gap_type", "unknown")
            groups[gt] = groups.get(gt, 0) + 1
        return groups

    def execute(self, gaps: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        return self.generate_hypotheses(gaps, analyses)
