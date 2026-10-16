"""Offline stand-ins for arXiv / Semantic Scholar so tests never touch the network."""
from contextlib import contextmanager
from unittest import mock

from fixtures import PAPERS


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    @property
    def content(self):
        entries = "".join(
            f"<entry><id>http://arxiv.org/abs/{i}v1</id><title>{p['title']}</title>"
            f"<published>{p['year']}-01-01T00:00:00Z</published><summary>{p['abstract']}</summary>"
            f"<author><name>Author {i}</name></author></entry>"
            for i, p in enumerate(PAPERS) if p["abstract"]
        )
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'.encode()

    def json(self):
        return {"data": [
            {"paperId": str(i), "title": p["title"] + " (SS)", "abstract": p["abstract"], "year": p["year"],
             "citationCount": 10 * i, "authors": [{"name": "B"}], "url": p["url"], "venue": "V"}
            for i, p in enumerate(PAPERS) if p["abstract"]
        ]}


@contextmanager
def offline_apis():
    with mock.patch("requests.Session.get", lambda self, url, **kw: FakeResponse()), \
         mock.patch("autoresearch.agents.literature_discovery.time.sleep", lambda s: None):
        yield
