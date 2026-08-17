"""Enhanced Base Agent with database and LLM integration."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import time
import logging
from datetime import datetime

class BaseAgent(ABC):
    """Abstract base class for all research agents with full infrastructure."""

    def __init__(self, name: str, db_session=None, llm_client=None):
        self.name = name
        self.db_session = db_session
        self.llm_client = llm_client
        self.memory = []
        self.metrics = {
            "calls": 0, 
            "tokens": 0, 
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        self.logger = logging.getLogger(f"autoresearch.agent.{name}")

    def log(self, message: str, level: str = "INFO"):
        """Log agent activity with structured format."""
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "time": timestamp, 
            "level": level, 
            "message": message,
            "agent": self.name
        }
        self.memory.append(entry)

        if level == "ERROR":
            self.logger.error(f"[{self.name}] {message}")
            self.metrics["errors"] += 1
        elif level == "WARNING":
            self.logger.warning(f"[{self.name}] {message}")
        else:
            self.logger.info(f"[{self.name}] {message}")

    def start_timer(self):
        """Start execution timer."""
        self.metrics["start_time"] = time.time()

    def stop_timer(self) -> float:
        """Stop execution timer and return duration."""
        self.metrics["end_time"] = time.time()
        return self.metrics["end_time"] - self.metrics["start_time"]

    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the agent's primary task."""
        pass

    def get_memory(self) -> List[Dict]:
        """Return agent's activity log."""
        return self.memory

    def get_metrics(self) -> Dict:
        """Return performance metrics."""
        metrics = self.metrics.copy()
        if metrics["start_time"] and metrics["end_time"]:
            metrics["duration_seconds"] = round(metrics["end_time"] - metrics["start_time"], 2)
        return metrics

    def save_to_db(self, data: Dict) -> None:
        """Save agent output to database if available."""
        if self.db_session:
            try:
                self.db_session.commit()
            except Exception as e:
                self.log(f"DB save failed: {str(e)}", "ERROR")
                self.db_session.rollback()
