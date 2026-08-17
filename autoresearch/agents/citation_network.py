"""Citation Network Builder v2 - Graph analysis with cluster detection."""
from typing import Dict, List, Any
from collections import defaultdict
from autoresearch.agents.base_agent import BaseAgent

class CitationNetworkBuilder(BaseAgent):
    """Builds citation networks and identifies research structure."""

    def __init__(self, db_session=None, llm_client=None):
        super().__init__("CitationNetwork", db_session, llm_client)

    def build_network(self, papers: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        self.start_timer()
        self.log(f"Building network for {len(papers)} papers")

        nodes = []
        for i, (paper, analysis) in enumerate(zip(papers, analyses)):
            nodes.append({
                "id": i,
                "external_id": paper.get("external_id", ""),
                "title": paper.get("title", "")[:80],
                "year": paper.get("published_year"),
                "citations": paper.get("citation_count", 0) or 0,
                "domains": analysis.get("domains", []),
                "methods": analysis.get("methodology", []),
                "authors": paper.get("authors", [])[:3],
                "venue": paper.get("venue", "")
            })

        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self._calculate_similarity(nodes[i], nodes[j])
                if similarity > 0.25:
                    edges.append({
                        "source": i,
                        "target": j,
                        "weight": round(similarity, 3),
                        "type": self._edge_type(nodes[i], nodes[j])
                    })

        clusters = self._find_clusters(nodes, edges)
        bridges = self._find_bridge_papers(nodes, edges, clusters)
        influential = self._find_influential_papers(nodes)

        duration = self.stop_timer()

        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "bridge_papers": bridges,
            "influential_papers": influential,
            "network_stats": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_clusters": len(clusters),
                "avg_degree": round(len(edges) * 2 / len(nodes), 2) if nodes else 0,
                "density": round(len(edges) / (len(nodes) * (len(nodes) - 1) / 2), 4) if len(nodes) > 1 else 0
            },
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

    def _calculate_similarity(self, n1: Dict, n2: Dict) -> float:
        score = 0.0
        d1, d2 = set(n1.get("domains", [])), set(n2.get("domains", []))
        if d1 and d2:
            score += (len(d1 & d2) / len(d1 | d2)) * 0.4
        m1, m2 = set(n1.get("methods", [])), set(n2.get("methods", []))
        if m1 and m2:
            score += (len(m1 & m2) / len(m1 | m2)) * 0.3
        a1 = set(a.lower() for a in n1.get("authors", []))
        a2 = set(a.lower() for a in n2.get("authors", []))
        if a1 and a2:
            score += (len(a1 & a2) / len(a1 | a2)) * 0.3
        return min(score, 1.0)

    def _edge_type(self, n1: Dict, n2: Dict) -> str:
        d1, d2 = set(n1.get("domains", [])), set(n2.get("domains", []))
        if d1 == d2:
            return "same_domain"
        elif d1 & d2:
            return "cross_domain"
        return "methodological"

    def _find_clusters(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        parent = list(range(len(nodes)))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for edge in edges:
            union(edge["source"], edge["target"])

        clusters = defaultdict(list)
        for i in range(len(nodes)):
            clusters[find(i)].append(i)

        result = []
        for cid, nids in clusters.items():
            if len(nids) >= 2:
                doms = set()
                for nid in nids:
                    doms.update(nodes[nid].get("domains", []))
                result.append({
                    "id": cid,
                    "size": len(nids),
                    "papers": [nodes[nid]["title"] for nid in nids],
                    "domains": list(doms),
                    "node_ids": nids
                })
        return result

    def _find_bridge_papers(self, nodes: List[Dict], edges: List[Dict], clusters: List[Dict]) -> List[Dict]:
        if len(clusters) < 2:
            return []
        node_to_cluster = {}
        for c in clusters:
            for nid in c["node_ids"]:
                node_to_cluster[nid] = c["id"]

        scores = defaultdict(float)
        for edge in edges:
            if edge["type"] == "cross_domain":
                c1 = node_to_cluster.get(edge["source"])
                c2 = node_to_cluster.get(edge["target"])
                if c1 and c2 and c1 != c2:
                    scores[edge["source"]] += edge["weight"]
                    scores[edge["target"]] += edge["weight"]

        return [
            {"node_id": nid, "title": nodes[nid]["title"], "bridge_score": round(score, 3)}
            for nid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _find_influential_papers(self, nodes: List[Dict]) -> List[Dict]:
        sorted_nodes = sorted(nodes, key=lambda x: x.get("citations", 0), reverse=True)
        return [
            {"node_id": i, "title": n["title"], "citations": n.get("citations", 0), "year": n.get("year")}
            for i, n in enumerate(sorted_nodes[:5])
        ]

    def execute(self, papers: List[Dict], analyses: List[Dict]) -> Dict[str, Any]:
        return self.build_network(papers, analyses)
