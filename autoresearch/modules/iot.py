"""IoT Module - Distributed agent messaging, lightweight deployment, edge computing."""
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from queue import Queue

class IoTModule:
    """
    Provides IoT-style distributed architecture for the AutoResearch pipeline.
    Includes message queuing between agents, lightweight mode for edge devices,
    and heartbeat monitoring.
    """

    def __init__(self, mode: str = "full"):
        self.mode = mode  # "full" or "edge"
        self.message_queue = Queue()
        self.agent_status = {}
        self.message_history = deque(maxlen=1000)
        self.subscribers = {}
        self.running = False
        self.heartbeat_interval = 5  # seconds

    def publish(self, topic: str, message: Dict[str, Any], sender: str = "unknown") -> str:
        """Publish a message to a topic (MQTT-style)."""
        msg_id = f"{sender}_{int(time.time()*1000)}"
        envelope = {
            "id": msg_id,
            "topic": topic,
            "sender": sender,
            "timestamp": time.time(),
            "payload": message,
            "delivered": False
        }
        self.message_queue.put(envelope)
        self.message_history.append(envelope)

        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(envelope)
                    envelope["delivered"] = True
                except Exception as e:
                    envelope["delivery_error"] = str(e)

        return msg_id

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def register_agent(self, agent_name: str, capabilities: List[str]) -> Dict[str, Any]:
        """Register an agent in the distributed system."""
        self.agent_status[agent_name] = {
            "name": agent_name,
            "capabilities": capabilities,
            "status": "online",
            "last_heartbeat": time.time(),
            "messages_processed": 0,
            "errors": 0
        }
        return self.agent_status[agent_name]

    def send_heartbeat(self, agent_name: str) -> Dict[str, Any]:
        """Send heartbeat to keep agent registered."""
        if agent_name in self.agent_status:
            self.agent_status[agent_name]["last_heartbeat"] = time.time()
            self.agent_status[agent_name]["status"] = "online"
            return {"status": "ok", "agent": agent_name}
        return {"status": "error", "reason": "Agent not registered"}

    def check_agent_health(self) -> Dict[str, Any]:
        """Check which agents are alive."""
        now = time.time()
        dead_threshold = self.heartbeat_interval * 3

        healthy = []
        unhealthy = []

        for name, status in self.agent_status.items():
            last_beat = status["last_heartbeat"]
            if now - last_beat > dead_threshold:
                status["status"] = "offline"
                unhealthy.append(name)
            else:
                healthy.append(name)

        return {
            "healthy_agents": healthy,
            "unhealthy_agents": unhealthy,
            "total_agents": len(self.agent_status),
            "health_rate": len(healthy) / len(self.agent_status) if self.agent_status else 0
        }

    def get_message_log(self, agent_name: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent messages, optionally filtered by agent."""
        messages = list(self.message_history)
        if agent_name:
            messages = [m for m in messages if m["sender"] == agent_name]
        return messages[-limit:]

    def edge_mode_summary(self) -> Dict[str, Any]:
        """Generate summary for edge deployment mode."""
        if self.mode == "edge":
            return {
                "mode": "edge",
                "features_disabled": ["llm_deep_reading", "pdf_parsing", "citation_network_viz"],
                "features_enabled": ["literature_discovery", "basic_reading", "gap_detection"],
                "memory_footprint": "low",
                "recommended_deployment": "Raspberry Pi 4 or better",
                "message_queue_size": self.message_queue.qsize()
            }
        return {
            "mode": "full",
            "features_enabled": "all",
            "memory_footprint": "high",
            "recommended_deployment": "Cloud server with GPU",
            "message_queue_size": self.message_queue.qsize()
        }

    def simulate_distributed_pipeline(self, topic: str) -> List[Dict]:
        """Simulate how agents would communicate in a distributed IoT setup."""
        events = []

        # Discovery agent publishes papers
        events.append(self.publish("papers/discovered", {"topic": topic, "count": 10}, "LiteratureDiscovery"))

        # Reading agent subscribes and publishes analyses
        events.append(self.publish("papers/analyzed", {"methodologies": ["transformer", "cnn"]}, "DeepReading"))

        # Network agent publishes graph
        events.append(self.publish("network/constructed", {"clusters": 3, "edges": 15}, "CitationNetwork"))

        # Gap agent publishes gaps
        events.append(self.publish("gaps/detected", {"total": 5, "top_score": 18}, "GapDetection"))

        # Hypothesis agent publishes hypotheses
        events.append(self.publish("hypotheses/generated", {"total": 3, "verified": 2}, "HypothesisGeneration"))

        return events
