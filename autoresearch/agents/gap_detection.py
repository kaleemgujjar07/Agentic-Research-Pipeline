"""Gap Detection Agent v2 - Multi-type gap detection with scoring."""
from typing import Dict, List, Any
from collections import defaultdict
from autoresearch.agents.base_agent import BaseAgent

class GapDetectionAgent(BaseAgent):
    """Detects research gaps using multiple strategies."""

    def __init__(self, db_session=None, llm_client=None):
        super().__init__("GapDetection", db_session, llm_client)

    def detect_gaps(self, analyses: List[Dict], network: Dict) -> Dict[str, Any]:
        self.start_timer()
        self.log("Starting gap detection")

        gaps = []
        gaps.extend(self._methodological_gaps(analyses, network))
        gaps.extend(self._temporal_gaps(analyses))
        gaps.extend(self._limitation_gaps(analyses))
        gaps.extend(self._cross_domain_gaps(network))
        gaps.extend(self._evaluation_gaps(analyses))

        scored = self._score_gaps(gaps, analyses)

        duration = self.stop_timer()

        return {
            "total_gaps": len(scored),
            "gaps": scored[:15],
            "gap_categories": self._categorize(scored),
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

    def _methodological_gaps(self, analyses: List[Dict], network: Dict) -> List[Dict]:
        gaps = []
        method_domains = defaultdict(set)
        for a in analyses:
            for m in a.get("methodology", []):
                for d in a.get("domains", []):
                    method_domains[m].add(d)

        all_domains = set()
        for a in analyses:
            all_domains.update(a.get("domains", []))

        for method, domains in method_domains.items():
            if 0 < len(domains) < len(all_domains):
                for missing in list(all_domains - domains)[:2]:
                    gaps.append({
                        "type": "methodological",
                        "description": f"Method '{method}' used in {', '.join(domains)} but not explored in {missing}",
                        "method": method,
                        "source_domains": list(domains),
                        "target_domain": missing,
                        "evidence": f"Found in {len(domains)} domains"
                    })
        return gaps

    def _temporal_gaps(self, analyses: List[Dict]) -> List[Dict]:
        gaps = []
        year_domain = defaultdict(lambda: defaultdict(int))
        for a in analyses:
            year = a.get("year") or a.get("published_year")
            if year:
                for d in a.get("domains", []):
                    try:
                        year_domain[d][int(year)] += 1
                    except:
                        pass

        for domain, years in year_domain.items():
            if len(years) >= 2:
                sy = sorted(years.keys())
                if len(sy) >= 2 and years.get(sy[-1], 0) < years.get(sy[-2], 0):
                    gaps.append({
                        "type": "temporal",
                        "description": f"Recent decline in {domain} research ({years.get(sy[-2], 0)} in {sy[-2]} vs {years.get(sy[-1], 0)} in {sy[-1]})",
                        "domain": domain,
                        "evidence": "Declining publication trend"
                    })
        return gaps

    def _limitation_gaps(self, analyses: List[Dict]) -> List[Dict]:
        gaps = []
        all_lims = []
        for a in analyses:
            all_lims.extend(a.get("limitations", []))

        themes = {
            "scalability": ["scalability", "scale", "large-scale", "computational cost", "efficiency"],
            "real_world": ["real-world", "real world", "deployment", "practical", "clinical"],
            "data": ["dataset", "data quality", "limited data", "small dataset"],
            "generalization": ["generalize", "generalization", "transfer", "domain shift"],
            "evaluation": ["evaluation", "metric", "benchmark", "comparison"]
        }

        for theme, keywords in themes.items():
            count = sum(1 for lim in all_lims if any(kw in lim.lower() for kw in keywords))
            if count >= 2:
                gaps.append({
                    "type": "limitation",
                    "description": f"Common {theme} limitation mentioned in {count} papers",
                    "theme": theme,
                    "frequency": count,
                    "evidence": f"Recurring across {count} papers"
                })
        return gaps

    def _cross_domain_gaps(self, network: Dict) -> List[Dict]:
        gaps = []
        clusters = network.get("clusters", [])
        if len(clusters) < 2:
            return gaps

        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters[i+1:], i+1):
                has_conn = False
                for edge in network.get("edges", []):
                    src_in_c1 = edge["source"] in c1["node_ids"]
                    tgt_in_c2 = edge["target"] in c2["node_ids"]
                    src_in_c2 = edge["source"] in c2["node_ids"]
                    tgt_in_c1 = edge["target"] in c1["node_ids"]
                    if (src_in_c1 and tgt_in_c2) or (src_in_c2 and tgt_in_c1):
                        has_conn = True
                        break
                if not has_conn:
                    gaps.append({
                        "type": "cross_domain",
                        "description": f"No research bridges {', '.join(c1['domains'])} and {', '.join(c2['domains'])}",
                        "cluster_1_domains": c1["domains"],
                        "cluster_2_domains": c2["domains"],
                        "evidence": "Disconnected clusters"
                    })
        return gaps

    def _evaluation_gaps(self, analyses: List[Dict]) -> List[Dict]:
        gaps = []
        domain_metrics = defaultdict(set)
        for a in analyses:
            for d in a.get("domains", []):
                for m in a.get("evaluation_metrics", []):
                    domain_metrics[d].add(m)

        all_metrics = set()
        for a in analyses:
            all_metrics.update(a.get("evaluation_metrics", []))

        for domain, metrics in domain_metrics.items():
            missing = all_metrics - metrics
            if missing and metrics:
                gaps.append({
                    "type": "evaluation",
                    "description": f"{domain} research rarely uses {', '.join(list(missing)[:2])} metrics",
                    "domain": domain,
                    "missing_metrics": list(missing)[:3],
                    "evidence": f"Current: {', '.join(metrics)}"
                })
        return gaps

    def _score_gaps(self, gaps: List[Dict], analyses: List[Dict]) -> List[Dict]:
        for gap in gaps:
            score = 5
            if "frequency" in gap:
                score += min(gap["frequency"] * 2, 10)
            if gap["type"] == "cross_domain":
                score += 8
            if gap["type"] == "methodological":
                score += 6
            if gap["type"] == "limitation":
                score += 4
            gap["score"] = min(score, 25)
        return sorted(gaps, key=lambda x: x["score"], reverse=True)

    def _categorize(self, gaps: List[Dict]) -> Dict[str, int]:
        cats = defaultdict(int)
        for g in gaps:
            cats[g["type"]] += 1
        return dict(cats)

    def execute(self, analyses: List[Dict], network: Dict) -> Dict[str, Any]:
        return self.detect_gaps(analyses, network)
