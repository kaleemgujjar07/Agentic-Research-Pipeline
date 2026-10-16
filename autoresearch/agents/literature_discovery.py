"""Literature Discovery Agent v2 - Multi-source with DB persistence."""
import requests
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from autoresearch.agents.base_agent import BaseAgent

class LiteratureDiscoveryAgent(BaseAgent):
    """Discovers papers from arXiv, Semantic Scholar, and PubMed."""

    def __init__(self, db_session=None, llm_client=None, config=None):
        super().__init__("LiteratureDiscovery", db_session, llm_client)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AutoResearch/2.0 (Academic Research Tool)"})

        if config:
            self.arxiv_base = config.apis.arxiv_base_url
            self.arxiv_max = config.apis.arxiv_max_results
            self.arxiv_timeout = config.apis.arxiv_timeout
            self.ss_base = config.apis.semantic_scholar_base_url
            self.ss_max = config.apis.semantic_scholar_max_results
            self.ss_timeout = config.apis.semantic_scholar_timeout
            self.pubmed_base = config.apis.pubmed_base_url
            self.pubmed_max = config.apis.pubmed_max_results
            self.pubmed_timeout = config.apis.pubmed_timeout
        else:
            self.arxiv_base = "http://export.arxiv.org/api/query"
            self.arxiv_max = 20
            self.arxiv_timeout = 15
            self.ss_base = "https://api.semanticscholar.org/graph/v1"
            self.ss_max = 20
            self.ss_timeout = 15
            self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            self.pubmed_max = 20
            self.pubmed_timeout = 15

    def search_arxiv(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.arxiv_max
        self.log(f"Searching arXiv: '{query}' (max={max_results})")

        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            response = self.session.get(self.arxiv_base, params=params, timeout=self.arxiv_timeout)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                paper = {
                    "source": "arXiv",
                    "external_id": entry.find("atom:id", ns).text.split("/")[-1] if entry.find("atom:id", ns) is not None else "",
                    "title": entry.find("atom:title", ns).text.strip() if entry.find("atom:title", ns) is not None else "",
                    "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns) if author.find("atom:name", ns) is not None],
                    "published_year": int(entry.find("atom:published", ns).text[:4]) if entry.find("atom:published", ns) is not None else None,
                    "abstract": entry.find("atom:summary", ns).text.strip() if entry.find("atom:summary", ns) is not None else "",
                    "url": entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else "",
                    "pdf_url": "",
                    "citation_count": 0,
                    "venue": "arXiv"
                }
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        paper["pdf_url"] = link.get("href")
                papers.append(paper)

            self.log(f"arXiv: {len(papers)} papers found")
            self.metrics["calls"] += 1
            return papers

        except Exception as e:
            self.log(f"arXiv error: {str(e)}", "ERROR")
            return []

    def search_semantic_scholar(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.ss_max
        self.log(f"Searching Semantic Scholar: '{query}' (max={max_results})")

        try:
            url = f"{self.ss_base}/paper/search"
            params = {
                "query": query,
                "fields": "title,authors,year,abstract,citationCount,referenceCount,url,openAccessPdf,venue",
                "limit": max_results
            }
            response = self.session.get(url, params=params, timeout=self.ss_timeout)
            response.raise_for_status()

            data = response.json()
            papers = []
            for paper in data.get("data", []):
                papers.append({
                    "source": "Semantic Scholar",
                    "external_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "authors": [a.get("name", "") for a in paper.get("authors", [])],
                    "published_year": paper.get("year"),
                    "abstract": paper.get("abstract", "") or "",
                    "url": paper.get("url", ""),
                    "pdf_url": paper.get("openAccessPdf", {}).get("url", "") if paper.get("openAccessPdf") else "",
                    "citation_count": paper.get("citationCount", 0) or 0,
                    "reference_count": paper.get("referenceCount", 0) or 0,
                    "venue": paper.get("venue", "")
                })

            self.log(f"Semantic Scholar: {len(papers)} papers found")
            self.metrics["calls"] += 1
            return papers

        except Exception as e:
            self.log(f"Semantic Scholar error: {str(e)}", "ERROR")
            return []

    def search_pubmed(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.pubmed_max
        self.log(f"Searching PubMed: '{query}' (max={max_results})")

        try:
            # Step 1: Search for IDs
            search_url = f"{self.pubmed_base}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            search_response = self.session.get(search_url, params=search_params, timeout=self.pubmed_timeout)
            search_response.raise_for_status()

            idlist = search_response.json().get("esearchresult", {}).get("idlist", [])

            if not idlist:
                self.log("PubMed: No results")
                return []

            # Step 2: Fetch summaries
            summary_url = f"{self.pubmed_base}/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(idlist),
                "retmode": "json"
            }
            summary_response = self.session.get(summary_url, params=summary_params, timeout=self.pubmed_timeout)
            summary_response.raise_for_status()

            results = summary_response.json().get("result", {})
            papers = []

            for pmid in idlist:
                if pmid in results:
                    article = results[pmid]
                    papers.append({
                        "source": "PubMed",
                        "external_id": f"PMID:{pmid}",
                        "title": article.get("title", ""),
                        "authors": [a.get("name", "") for a in article.get("authors", [])],
                        "published_year": article.get("pubdate", "")[:4] if article.get("pubdate") else None,
                        "abstract": "",  # Would need separate efetch call
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "pdf_url": "",
                        "citation_count": 0,
                        "venue": article.get("fulljournalname", "")
                    })

            self.log(f"PubMed: {len(papers)} papers found")
            self.metrics["calls"] += 1
            return papers

        except Exception as e:
            self.log(f"PubMed error: {str(e)}", "ERROR")
            return []

    def _deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicates by title similarity."""
        seen = {}
        unique = []

        for paper in papers:
            title_key = paper["title"].lower().strip()[:60]
            if title_key not in seen:
                seen[title_key] = paper
                unique.append(paper)
            else:
                # Merge citation counts
                if paper.get("citation_count", 0) > seen[title_key].get("citation_count", 0):
                    seen[title_key]["citation_count"] = paper["citation_count"]

        return unique

    def execute(self, query: str, max_results: int = 10, sources: List[str] = None) -> Dict[str, Any]:
        self.start_timer()

        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        self.log(f"Starting discovery for: '{query}'")
        all_papers = []

        if "arxiv" in sources:
            all_papers.extend(self.search_arxiv(query, max_results))
            time.sleep(1)

        if "semantic_scholar" in sources:
            all_papers.extend(self.search_semantic_scholar(query, max_results))
            time.sleep(0.5)

        if "pubmed" in sources:
            all_papers.extend(self.search_pubmed(query, max_results))

        unique_papers = self._deduplicate(all_papers)
        unique_papers.sort(key=lambda x: x.get("citation_count", 0) or 0, reverse=True)

        duration = self.stop_timer()

        result = {
            "query": query,
            "total_found": len(unique_papers),
            "papers": unique_papers[:max_results],
            "sources_searched": sources,
            "agent": self.name,
            "duration_seconds": round(duration, 2)
        }

        self.log(f"Discovery complete: {len(unique_papers)} unique papers in {duration:.1f}s")
        return result
