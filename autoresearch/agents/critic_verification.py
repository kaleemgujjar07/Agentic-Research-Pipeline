"""Critic/Verification Agent v2 - Validates hypotheses against literature."""
from typing import Dict, List, Any
from autoresearch.agents.base_agent import BaseAgent

class CriticVerificationAgent(BaseAgent):
    def __init__(self, db_session=None, llm_client=None):
        super().__init__("CriticVerification", db_session, llm_client)

    def verify_hypotheses(self, hypotheses: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        self.start_timer()
        self.log(f"Verifying {len(hypotheses)} hypotheses")

        verified = []
        rejected = []

        for h in hypotheses:
            result = self._verify_single(h, analyses)
            if result["status"] == "verified":
                verified.append(result)
            else:
                rejected.append(result)

        duration = self.stop_timer()

        return {
            "total_verified": len(verified),
            "total_rejected": len(rejected),
            "verified_hypotheses": verified[:8],
            "rejected_hypotheses": rejected[:3],
            "verification_stats": {
                "verification_rate": len(verified) / len(hypotheses) if hypotheses else 0,
                "avg_confidence": sum(v.get("confidence", 0) for v in verified) / len(verified) if verified else 0
            },
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

    def _verify_single(self, hypothesis: Dict, analyses: List[Dict]) -> Dict:
        h_text = hypothesis.get("hypothesis", "")
        h_lower = h_text.lower()

        # Check 1: Already addressed?
        for analysis in analyses:
            claims = " ".join(analysis.get("key_claims", [])).lower()
            h_words = set(h_lower.split())
            c_words = set(claims.split())
            if h_words and c_words:
                overlap = len(h_words & c_words) / len(h_words)
                if overlap > 0.7:
                    return {
                        "hypothesis": h_text,
                        "status": "rejected",
                        "reason": "Already addressed in existing literature",
                        "confidence": 0.1,
                        "original_score": hypothesis.get("overall_score", 0)
                    }

        # Check 2: Vagueness
        vague = ["somehow", "maybe", "probably", "might", "could possibly"]
        if sum(1 for v in vague if v in h_lower) >= 2:
            return {
                "hypothesis": h_text,
                "status": "rejected",
                "reason": "Too vague - lacks specificity",
                "confidence": 0.2,
                "original_score": hypothesis.get("overall_score", 0)
            }

        # Check 3: Feasibility
        has_eval = any(t in h_lower for t in ["accuracy", "f1", "precision", "recall", "improve", "reduce"])
        confidence = 0.5 + (0.2 if has_eval else 0) + (0.2 if hypothesis.get("novelty") == "high" else 0.1)

        return {
            "hypothesis": h_text,
            "status": "verified",
            "reason": "Passes verification checks",
            "confidence": round(min(confidence, 1.0), 2),
            "original_score": hypothesis.get("overall_score", 0),
            "feasibility": "high" if has_eval else "medium",
            "has_evaluation": has_eval
        }

    def execute(self, hypotheses: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        return self.verify_hypotheses(hypotheses, analyses)
