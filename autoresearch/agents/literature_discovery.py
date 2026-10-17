"""Literature Discovery Agent v3 - multi-source, resilient, and explains its failures.

Sources: arXiv, Semantic Scholar, OpenAlex (and PubMed titles, optional).
Free public APIs rate-limit shared cloud servers, so this agent:
  * retries with backoff on 429/5xx and network errors,
  * records what every source returned (``source_status``) instead of failing silently,
  * retries once with a broader query if nothing at all was found,
  * caches successful searches for an hour so repeated runs do not re-hit the APIs.

Optional environment variables: SEMANTIC_SCHOLAR_API_KEY, OPENALEX_MAILTO.
"""
import copy
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

from autoresearch.agents.base_agent import BaseAgent

DEFAULT_SOURCES = ["arxiv", "semantic_scholar", "openalex"]
ALL_SOURCES = DEFAULT_SOURCES + ["pubmed"]


class LiteratureDiscoveryAgent(BaseAgent):
    """Discovers papers from arXiv, Semantic Scholar, OpenAlex and (optionally) PubMed."""

    CACHE_TTL_SECONDS = 3600
    _cache: Dict[Any, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, db_session=None, llm_client=None, config=None, max_retries: int = 2):
        super().__init__("LiteratureDiscovery", db_session, llm_client)
        self.max_retries = max_retries
        self.source_status: Dict[str, Dict[str, Any]] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AutoResearch/3.0 (Academic Research Tool)"})

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
            self.arxiv_base = "https://export.arxiv.org/api/query"
            self.arxiv_max = 20
            self.arxiv_timeout = 15
            self.ss_base = "https://api.semanticscholar.org/graph/v1"
            self.ss_max = 20
            self.ss_timeout = 15
            self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            self.pubmed_max = 20
            self.pubmed_timeout = 15
        self.openalex_base = "https://api.openalex.org/works"
        self.openalex_timeout = 15

    # ------------------------------------------------------------------ cache
    @classmethod
    def clear_cache(cls) -> None:
        with cls._cache_lock:
            cls._cache.clear()

    # ------------------------------------------------------------------ http
    @staticmethod
    def _retry_wait(response, attempt: int) -> float:
        ra = str(getattr(response, "headers", {}).get("Retry-After", "")).strip()
        if ra.isdigit():
            return float(min(int(ra), 5))
        return float(2 * (attempt + 1))

    def _get(self, url: str, params: Dict, timeout: int, headers: Optional[Dict] = None):
        """GET with retry/backoff. Raises RuntimeError with a short reason on final failure."""
        last = "request failed"
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=timeout, headers=headers)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    wait = self._retry_wait(r, attempt)
                    last = f"HTTP {r.status_code}"
                    self.log(f"HTTP {r.status_code} from {url.split('/')[2]}; retrying in {wait:.0f}s", "WARNING")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except Exception as e:  # noqa: BLE001 - any failure is reported, never swallowed
                last = f"{type(e).__name__}: {str(e)[:140]}"
                transient = isinstance(e, (requests.Timeout, requests.ConnectionError))
                if transient and attempt < self.max_retries:
                    time.sleep(1 + attempt)
                    continue
                break
        raise RuntimeError(last)

    def _record(self, source: str, papers: Optional[List[Dict]], error: Optional[str] = None) -> List[Dict]:
        if error is None:
            self.source_status[source] = {"ok": True, "count": len(papers or [])}
            self.metrics["calls"] += 1
            self.log(f"{source}: {len(papers or [])} papers found")
            return papers or []
        self.source_status[source] = {"ok": False, "error": error, "count": 0}
        self.log(f"{source} error: {error}", "ERROR")
        return []

    # ------------------------------------------------------------------ sources
    def search_arxiv(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.arxiv_max
        self.log(f"Searching arXiv: '{query}' (max={max_results})")
        words = re.findall(r"\w+", query)[:8]
        if not words:
            return self._record("arxiv", [], None)
        try:
            params = {
                "search_query": " AND ".join(f"all:{w}" for w in words),
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            response = self._get(self.arxiv_base, params, self.arxiv_timeout)
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            def text(entry, tag):
                node = entry.find(f"atom:{tag}", ns)
                return " ".join(node.text.split()) if node is not None and node.text else ""

            papers = []
            for entry in root.findall("atom:entry", ns):
                published = text(entry, "published")
                paper = {
                    "source": "arXiv",
                    "external_id": text(entry, "id").split("/")[-1],
                    "title": text(entry, "title"),
                    "authors": [a.text for a in (au.find("atom:name", ns) for au in entry.findall("atom:author", ns)) if a is not None and a.text],
                    "published_year": int(published[:4]) if published[:4].isdigit() else None,
                    "abstract": text(entry, "summary"),
                    "url": text(entry, "id"),
                    "pdf_url": "",
                    "citation_count": 0,
                    "venue": "arXiv",
                }
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        paper["pdf_url"] = link.get("href")
                papers.append(paper)
            return self._record("arxiv", papers)
        except Exception as e:  # noqa: BLE001
            return self._record("arxiv", None, str(e))

    def search_semantic_scholar(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.ss_max
        self.log(f"Searching Semantic Scholar: '{query}' (max={max_results})")
        try:
            params = {
                "query": query,
                "fields": "title,authors,year,abstract,citationCount,referenceCount,url,openAccessPdf,venue",
                "limit": max_results,
            }
            key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            headers = {"x-api-key": key} if key else None
            response = self._get(f"{self.ss_base}/paper/search", params, self.ss_timeout, headers)
            papers = []
            for paper in response.json().get("data") or []:
                papers.append({
                    "source": "Semantic Scholar",
                    "external_id": paper.get("paperId", ""),
                    "title": paper.get("title", "") or "",
                    "authors": [a.get("name", "") for a in paper.get("authors") or []],
                    "published_year": paper.get("year"),
                    "abstract": paper.get("abstract", "") or "",
                    "url": paper.get("url", "") or "",
                    "pdf_url": (paper.get("openAccessPdf") or {}).get("url", ""),
                    "citation_count": paper.get("citationCount", 0) or 0,
                    "reference_count": paper.get("referenceCount", 0) or 0,
                    "venue": paper.get("venue", "") or "",
                })
            return self._record("semantic_scholar", papers)
        except Exception as e:  # noqa: BLE001
            return self._record("semantic_scholar", None, str(e))

    @staticmethod
    def _openalex_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
        """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
        if not inverted:
            return ""
        by_pos = {}
        for word, positions in inverted.items():
            for p in positions:
                by_pos[p] = word
        return " ".join(by_pos[p] for p in sorted(by_pos))

    def search_openalex(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.ss_max
        self.log(f"Searching OpenAlex: '{query}' (max={max_results})")
        try:
            params = {"search": query, "per-page": min(max_results, 50), "filter": "has_abstract:true"}
            if os.environ.get("OPENALEX_MAILTO"):
                params["mailto"] = os.environ["OPENALEX_MAILTO"]
            response = self._get(self.openalex_base, params, self.openalex_timeout)
            papers = []
            for w in response.json().get("results") or []:
                loc_source = ((w.get("primary_location") or {}).get("source") or {})
                papers.append({
                    "source": "OpenAlex",
                    "external_id": (w.get("id") or "").rsplit("/", 1)[-1],
                    "title": w.get("title") or w.get("display_name") or "",
                    "authors": [((a or {}).get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])][:10],
                    "published_year": w.get("publication_year"),
                    "abstract": self._openalex_abstract(w.get("abstract_inverted_index")),
                    "url": w.get("doi") or w.get("id") or "",
                    "pdf_url": (w.get("open_access") or {}).get("oa_url") or "",
                    "citation_count": w.get("cited_by_count") or 0,
                    "venue": loc_source.get("display_name") or "",
                })
            return self._record("openalex", papers)
        except Exception as e:  # noqa: BLE001
            return self._record("openalex", None, str(e))

    def search_pubmed(self, query: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or self.pubmed_max
        self.log(f"Searching PubMed: '{query}' (max={max_results})")
        try:
            search = self._get(f"{self.pubmed_base}/esearch.fcgi",
                               {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
                               self.pubmed_timeout)
            idlist = search.json().get("esearchresult", {}).get("idlist", [])
            if not idlist:
                return self._record("pubmed", [])
            summary = self._get(f"{self.pubmed_base}/esummary.fcgi",
                                {"db": "pubmed", "id": ",".join(idlist), "retmode": "json"},
                                self.pubmed_timeout)
            results = summary.json().get("result", {})
            papers = []
            for pmid in idlist:
                article = results.get(pmid)
                if not article:
                    continue
                pubdate = article.get("pubdate", "") or ""
                papers.append({
                    "source": "PubMed",
                    "external_id": f"PMID:{pmid}",
                    "title": article.get("title", ""),
                    "authors": [a.get("name", "") for a in article.get("authors", [])],
                    "published_year": int(pubdate[:4]) if pubdate[:4].isdigit() else None,
                    "abstract": "",  # would need a separate efetch call
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "pdf_url": "",
                    "citation_count": 0,
                    "venue": article.get("fulljournalname", ""),
                })
            return self._record("pubmed", papers)
        except Exception as e:  # noqa: BLE001
            return self._record("pubmed", None, str(e))

    # ------------------------------------------------------------------ orchestration
    def _deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicates by title; keep the highest citation count."""
        seen: Dict[str, Dict] = {}
        unique: List[Dict] = []
        for paper in papers:
            key = re.sub(r"\W+", " ", (paper.get("title") or "").lower()).strip()[:60]
            if not key:
                continue
            if key not in seen:
                seen[key] = paper
                unique.append(paper)
            elif (paper.get("citation_count") or 0) > (seen[key].get("citation_count") or 0):
                seen[key]["citation_count"] = paper["citation_count"]
        return unique

    def _search_sources(self, query: str, max_results: int, sources: List[str]) -> List[Dict]:
        searchers = {
            "arxiv": self.search_arxiv,
            "semantic_scholar": self.search_semantic_scholar,
            "openalex": self.search_openalex,
            "pubmed": self.search_pubmed,
        }
        papers: List[Dict] = []
        for name in sources:
            if name not in searchers:
                self.source_status[name] = {"ok": False, "error": "unknown source", "count": 0}
                continue
            papers.extend(searchers[name](query, max_results))
            time.sleep(0.5)  # be polite to free APIs
        return papers

    def execute(self, query: str, max_results: int = 10, sources: List[str] = None) -> Dict[str, Any]:
        self.start_timer()
        sources = [s for s in (sources or DEFAULT_SOURCES)]
        cache_key = (query.strip().lower(), max_results, tuple(sources))

        with self._cache_lock:
            hit = self._cache.get(cache_key)
        if hit and time.time() - hit[0] < self.CACHE_TTL_SECONDS:
            self.log("Discovery served from cache")
            cached = copy.deepcopy(hit[1])
            cached["from_cache"] = True
            self.stop_timer()
            return cached

        self.log(f"Starting discovery for: '{query}'")
        self.source_status = {}
        all_papers = self._search_sources(query, max_results, sources)

        broadened = None
        words = query.split()
        if not any(p.get("title") for p in all_papers) and len(words) > 3:
            broadened = " ".join(words[:3])
            self.log(f"Nothing found; retrying with broader query '{broadened}'", "WARNING")
            all_papers = self._search_sources(broadened, max_results, sources)

        # papers without an abstract carry nothing the downstream agents can read
        with_abstract = [p for p in all_papers if (p.get("abstract") or "").strip()]
        if with_abstract:
            all_papers = with_abstract

        unique = self._deduplicate(all_papers)
        unique.sort(key=lambda x: x.get("citation_count", 0) or 0, reverse=True)
        duration = self.stop_timer()

        result = {
            "query": query,
            "broadened_query": broadened,
            "total_found": len(unique),
            "papers": unique[:max_results],
            "sources_searched": sources,
            "source_status": copy.deepcopy(self.source_status),
            "from_cache": False,
            "agent": self.name,
            "duration_seconds": round(duration, 2),
        }
        if unique:  # never cache a failure
            with self._cache_lock:
                self._cache[cache_key] = (time.time(), copy.deepcopy(result))
        self.log(f"Discovery complete: {len(unique)} unique papers in {duration:.1f}s")
        return result
