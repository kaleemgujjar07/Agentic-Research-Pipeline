"""Deep Reading Agent v2 - LLM-powered paper analysis with DB persistence."""
from typing import Dict, List, Any
from autoresearch.agents.base_agent import BaseAgent

class DeepReadingAgent(BaseAgent):
    """Performs deep reading using LLM + heuristic extraction."""

    def __init__(self, db_session=None, llm_client=None):
        super().__init__("DeepReading", db_session, llm_client)

    def analyze_paper(self, paper: Dict) -> Dict[str, Any]:
        """Analyze a single paper using LLM if available, fallback to heuristics."""
        self.log(f"Analyzing: {paper.get('title', 'Unknown')[:60]}...")

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", "")

        # Try LLM analysis first
        if self.llm_client:
            try:
                llm_result = self.llm_client.analyze_paper(title, abstract, full_text)
                analysis = {
                    "external_id": paper.get("external_id", ""),
                    "title": title,
                    "methodology": llm_result.get("methodology", []),
                    "domains": llm_result.get("domains", []),
                    "key_claims": llm_result.get("key_claims", []),
                    "limitations": llm_result.get("limitations", []),
                    "results_summary": llm_result.get("results_summary", ""),
                    "research_type": llm_result.get("research_type", "empirical"),
                    "data_sources": llm_result.get("data_sources", []),
                    "evaluation_metrics": llm_result.get("evaluation_metrics", []),
                    "novelty_score": llm_result.get("novelty_score", 5.0),
                    "impact_score": llm_result.get("impact_score", 5.0),
                    "analysis_method": "llm"
                }
                self.metrics["calls"] += 1
                return analysis
            except Exception as e:
                self.log(f"LLM analysis failed, using heuristics: {str(e)}", "WARNING")

        # Fallback to heuristic analysis
        analysis = self._heuristic_analysis(paper)
        analysis["analysis_method"] = "heuristic"
        return analysis

    def _heuristic_analysis(self, paper: Dict) -> Dict[str, Any]:
        """Heuristic-based analysis when LLM unavailable."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = (title + " " + abstract).lower()

        # Methodology detection
        methods = []
        method_keywords = {
            "deep_learning": ["neural network", "deep learning", "cnn", "rnn", "lstm", "transformer", "bert", "gpt"],
            "reinforcement_learning": ["reinforcement learning", "q-learning", "policy gradient", "rl"],
            "nlp": ["natural language processing", "nlp", "text mining", "sentiment"],
            "computer_vision": ["computer vision", "image", "object detection", "segmentation"],
            "graph_methods": ["graph neural network", "gnn", "graph convolution"],
            "time_series": ["time series", "forecasting", "sarimax", "arima"],
            "statistical": ["statistical analysis", "regression", "hypothesis testing"],
            "survey": ["survey", "literature review", "systematic review"],
            "experimental": ["experiment", "empirical study", "user study"]
        }
        for category, keywords in method_keywords.items():
            if any(kw in text for kw in keywords):
                methods.append(category)

        # Domain classification
        domains = []
        domain_keywords = {
            "computer_vision": ["image", "video", "visual", "object detection", "segmentation"],
            "nlp": ["text", "language", "sentiment", "translation", "summarization"],
            "rl": ["reinforcement", "agent", "game", "policy", "reward"],
            "multimodal": ["multimodal", "vision-language", "cross-modal"],
            "healthcare": ["medical", "health", "clinical", "diagnosis", "disease"],
            "robotics": ["robot", "autonomous", "navigation", "manipulation"],
            "security": ["security", "adversarial", "attack", "defense", "privacy"],
            "systems": ["distributed", "cloud", "edge", "system"]
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domains.append(domain)

        # Claims extraction
        claims = []
        claim_indicators = ["we propose", "we introduce", "we present", "we develop", "we demonstrate", "we show", "we achieve", "our approach", "our method", "this paper", "in this work"]
        sentences = abstract.split(".")
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(indicator in sentence_lower for indicator in claim_indicators):
                if len(sentence.strip()) > 20:
                    claims.append(sentence.strip())

        # Limitations
        limitations = []
        limitation_keywords = ["limitation", "limited", "future work", "future research", "further investigation", "not addressed", "remains challenging", "open problem", "difficult to", "struggle to", "fails to", "does not", "cannot", "unable to", "requires further"]
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in limitation_keywords):
                limitations.append(sentence.strip())

        return {
            "external_id": paper.get("external_id", ""),
            "title": title,
            "methodology": methods if methods else ["not_specified"],
            "domains": domains if domains else ["general_ai"],
            "key_claims": claims[:3] if claims else ["No explicit claims found"],
            "limitations": limitations if limitations else ["No explicit limitations stated"],
            "results_summary": "",
            "research_type": "empirical",
            "data_sources": [],
            "evaluation_metrics": [],
            "novelty_score": 5.0,
            "impact_score": 5.0,
            "analysis_method": "heuristic"
        }

    def execute(self, papers: List[Dict]) -> Dict[str, Any]:
        self.start_timer()
        self.log(f"Starting deep reading on {len(papers)} papers")

        analyses = []
        for paper in papers:
            analysis = self.analyze_paper(paper)
            analyses.append(analysis)

        # Build summary statistics
        domains = {}
        methods = {}
        years = {}

        for analysis in analyses:
            for domain in analysis.get("domains", []):
                domains[domain] = domains.get(domain, 0) + 1
            for method in analysis.get("methodology", []):
                methods[method] = methods.get(method, 0) + 1
            year = analysis.get("published_year")
            if year:
                years[str(year)] = years.get(str(year), 0) + 1

        duration = self.stop_timer()

        result = {
            "total_analyzed": len(analyses),
            "analyses": analyses,
            "summary": {
                "domains": domains,
                "methodologies": methods,
                "year_distribution": years
            },
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

        self.log(f"Deep reading complete: {len(analyses)} papers in {duration:.1f}s")
        return result
