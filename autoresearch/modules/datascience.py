"""Data Science Module - Statistical analysis, trend forecasting, and metrics."""
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict

class DataScienceModule:
    """
    Provides data science capabilities for research analysis.
    Includes time-series forecasting, statistical significance testing,
    citation impact metrics, and trend analysis.
    """

    def __init__(self):
        self.has_statsmodels = self._check_statsmodels()
        self.has_scipy = self._check_scipy()

    def _check_statsmodels(self) -> bool:
        try:
            import statsmodels
            return True
        except ImportError:
            return False

    def _check_scipy(self) -> bool:
        try:
            import scipy
            return True
        except ImportError:
            return False

    def forecast_research_trend(self, year_counts: Dict[str, int], forecast_years: int = 3) -> Dict[str, Any]:
        """Forecast future publication trends using linear regression."""
        years = sorted([int(y) for y in year_counts.keys() if str(y).isdigit()])
        counts = [year_counts[str(y)] for y in years]

        if len(years) < 2:
            return {"error": "Need at least 2 years of data"}

        # Simple linear regression
        n = len(years)
        mean_x = sum(years) / n
        mean_y = sum(counts) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(years, counts))
        denominator = sum((x - mean_x) ** 2 for x in years)

        slope = numerator / denominator if denominator != 0 else 0
        intercept = mean_y - slope * mean_x

        # Forecast
        last_year = max(years)
        forecasts = []
        for i in range(1, forecast_years + 1):
            fy = last_year + i
            fc = max(0, slope * fy + intercept)
            forecasts.append({"year": fy, "predicted_count": round(fc, 1)})

        # R-squared
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(years, counts))
        ss_tot = sum((y - mean_y) ** 2 for y in counts)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "historical": [{"year": y, "count": c} for y, c in zip(years, counts)],
            "forecast": forecasts,
            "trend": "growing" if slope > 0.5 else "declining" if slope < -0.5 else "stable",
            "slope": round(slope, 3),
            "r_squared": round(r_squared, 3),
            "model": "linear_regression"
        }

    def calculate_citation_impact_metrics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Calculate impact metrics for discovered papers."""
        citations = [p.get("citation_count", 0) or 0 for p in papers]
        years = [p.get("published_year") for p in papers if p.get("published_year")]

        if not citations:
            return {"error": "No citation data"}

        # Basic stats
        mean_citations = statistics.mean(citations)
        median_citations = statistics.median(citations)

        # H-index simulation (papers with at least N citations)
        sorted_citations = sorted(citations, reverse=True)
        h_index = 0
        for i, c in enumerate(sorted_citations, 1):
            if c >= i:
                h_index = i
            else:
                break

        # Citation velocity (recent vs old)
        current_year = max(years) if years else 2024
        recent_papers = [p for p in papers if p.get("published_year") and current_year - p.get("published_year") <= 2]
        old_papers = [p for p in papers if p.get("published_year") and current_year - p.get("published_year") > 5]

        recent_avg = statistics.mean([p.get("citation_count", 0) or 0 for p in recent_papers]) if recent_papers else 0
        old_avg = statistics.mean([p.get("citation_count", 0) or 0 for p in old_papers]) if old_papers else 0

        return {
            "total_papers": len(papers),
            "mean_citations": round(mean_citations, 1),
            "median_citations": median_citations,
            "h_index": h_index,
            "max_citations": max(citations),
            "recent_avg_citations": round(recent_avg, 1),
            "old_avg_citations": round(old_avg, 1),
            "velocity_trend": "hot_topic" if recent_avg > old_avg * 1.5 else "mature_field"
        }

    def statistical_gap_significance(self, gaps: List[Dict]) -> Dict[str, Any]:
        """Test if detected gaps are statistically significant."""
        if not gaps:
            return {"error": "No gaps to analyze"}

        scores = [g.get("score", 0) for g in gaps]

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0

        # Identify high-confidence gaps (mean + 1 std)
        threshold = mean_score + std_score
        high_confidence = [g for g in gaps if g.get("score", 0) >= threshold]

        # Category distribution
        cat_dist = defaultdict(int)
        for g in gaps:
            cat_dist[g.get("type", "unknown")] += 1

        return {
            "total_gaps": len(gaps),
            "mean_score": round(mean_score, 2),
            "std_score": round(std_score, 2),
            "high_confidence_gaps": len(high_confidence),
            "high_confidence_threshold": round(threshold, 2),
            "category_distribution": dict(cat_dist),
            "recommendation": "Focus on high-confidence gaps" if high_confidence else "Broaden search"
        }

    def generate_research_health_score(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Generate an overall health score for the research area."""
        if not analyses:
            return {"error": "No analyses"}

        # Diversity score (how many different methods/domains)
        all_methods = set()
        all_domains = set()
        for a in analyses:
            all_methods.update(a.get("methodology", []))
            all_domains.update(a.get("domains", []))

        diversity = len(all_methods) + len(all_domains)

        # Limitation density (more limitations = more opportunity)
        total_limitations = sum(len(a.get("limitations", [])) for a in analyses)
        limitation_density = total_limitations / len(analyses)

        # Recency score
        years = [a.get("published_year") for a in analyses if a.get("published_year")]
        if years:
            avg_year = sum(years) / len(years)
            recency = (avg_year - 2015) / 10  # Normalize
        else:
            recency = 0.5

        # Composite score (0-100)
        health = min(100, (diversity * 5) + (limitation_density * 10) + (recency * 20))

        return {
            "health_score": round(health, 1),
            "diversity_score": diversity,
            "limitation_density": round(limitation_density, 2),
            "recency_score": round(recency, 2),
            "interpretation": "High opportunity" if health > 60 else "Moderate opportunity" if health > 30 else "Saturated"
        }
