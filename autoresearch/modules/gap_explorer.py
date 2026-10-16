"""Gap Explorer core logic (no Streamlit imports, fully unit-testable).

Turns paper abstracts into *fine-grained, source-linked gap statements* and lets a
human and the AI co-organise them:

* the AI proposes a clustering (TF-IDF + KMeans) and names clusters from their top terms
* the human can override: pin gaps into their own named groups, star, remove, rename
* AI re-clustering never touches human-pinned gaps (AI structure is *provisional*,
  human structure is *authoritative*)
* every re-cluster can be previewed as a diff before it is applied

Design follows the implications of MindTrellis (Li, Li, Kuang, Liu, Zhao; ACM DIS 2026, doi:10.1145/3800645.3813045): editable AI structure,
preview before execution, provenance for every item, and respect for manual layout.
"""
from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------- #
# 1. Gap statement extraction (with provenance)
# --------------------------------------------------------------------------- #

# Phrases that signal an author-stated limitation / open problem.
_CUE_PATTERNS = [
    r"\blimitations?\b", r"\blimited\b", r"\blimits?\b",
    r"\bchalleng(?:e|es|ing)\b", r"\bremains?\b", r"\bunderexplored\b",
    r"\bunexplored\b", r"\bfuture work\b", r"\bopen (?:problem|question)s?\b",
    r"\black(?:s|ing)?\b", r"\bsuffers?\b", r"\bstruggles?\b", r"\bfails?\b",
    r"\bpoorly\b", r"\bdifficult(?:y|ies)?\b", r"\bbottleneck\b",
    r"\bcomputationally (?:expensive|costly|intensive)\b", r"\bexpensive\b",
    r"\bnot (?:well|yet|fully)\b", r"\byet to\b", r"\bunsolved\b",
    r"\bhowever\b", r"\bdespite\b", r"\bdrawbacks?\b", r"\bshortcomings?\b",
]
_CUE_RE = re.compile("|".join(_CUE_PATTERNS), re.IGNORECASE)

# Coarse theme tags so each gap carries an interpretable label from the start.
THEMES: Dict[str, List[str]] = {
    "Data & annotation": ["data", "dataset", "annotation", "label", "labeled", "scarce", "small sample"],
    "Compute & efficiency": ["computation", "computational", "efficiency", "memory", "latency", "cost", "expensive", "scale", "scalab"],
    "Generalization": ["generaliz", "transfer", "domain shift", "out-of-distribution", "unseen", "cross-dataset", "robust"],
    "Evaluation": ["evaluation", "benchmark", "metric", "comparison", "protocol", "baseline"],
    "Interpretability & trust": ["interpret", "explain", "explan", "trust", "black box", "transparen", "clinical", "safety"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _theme_of(text: str) -> str:
    low = text.lower()
    best, best_hits = "Other", 0
    for theme, kws in THEMES.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best, best_hits = theme, hits
    return best


theme_of = _theme_of  # public alias used by the gap-detection agent


def source_label(gap: dict) -> str:
    """'Title (2022)' or just 'Title' when the year is unknown."""
    title = gap.get("paper_title", "Untitled")
    year = gap.get("year")
    return f"{title} ({year})" if year else title


def extract_gap_statements(papers: Sequence[dict], min_len: int = 40, max_len: int = 400) -> List[dict]:
    """Pull limitation / open-problem sentences out of abstracts.

    Each result keeps the source paper (title/url/year), so every gap is traceable.
    Papers whose abstracts contain no cue sentence simply contribute nothing; we do
    not invent gaps.
    """
    gaps: List[dict] = []
    for p_idx, paper in enumerate(papers):
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        for sent in _SENT_SPLIT.split(abstract):
            sent = " ".join(sent.split())
            if not (min_len <= len(sent) <= max_len):
                continue
            cues = sorted({m.group(0).lower() for m in _CUE_RE.finditer(sent)})
            if not cues:
                continue
            gaps.append({
                "id": f"g{len(gaps) + 1}",
                "text": sent,
                "cues": cues,
                "theme": _theme_of(sent),
                "paper_title": paper.get("title", "Untitled"),
                "paper_url": paper.get("url", ""),
                "year": paper.get("year", 0),
                "paper_index": p_idx,
            })
    return gaps


# --------------------------------------------------------------------------- #
# 2. Clustering (AI proposal) + 2-D layout
# --------------------------------------------------------------------------- #

# Words that make a sentence *read like* a limitation but say nothing about the topic.
# If they stay in the features, clusters form around "remains / future work / however"
# instead of around what the gap is actually about.
_GENERIC = {
    "work", "future", "remains", "remain", "however", "despite", "model", "models",
    "method", "methods", "approach", "approaches", "paper", "study", "propose",
    "existing", "current", "often", "rarely", "major", "concern", "particularly",
    "make", "making", "address", "challenge", "challenges", "challenging", "limited",
    "limitation", "limitations", "lack", "lacks", "lacking", "suffer", "suffers",
    "struggle", "struggles", "fail", "fails", "unexplored", "underexplored",
    "difficult", "difficulty", "bottleneck", "poorly", "drawback", "shortcoming",
    "unsolved", "open", "problem", "problems", "question", "questions",
}
_STOP = sorted(set(TfidfVectorizer(stop_words="english").get_stop_words()) | _GENERIC)

# Theme -> synthetic token. Appended to the text so sentences about the same concern
# (e.g. "latency" and "memory usage") are pulled together even without shared words.
_THEME_TOKEN = {t: "theme" + re.sub(r"[^a-z]", "", t.lower()) for t in THEMES}


def _features_text(text: str) -> str:
    theme = _theme_of(text)
    tok = _THEME_TOKEN.get(theme)
    return f"{text} {tok} {tok}" if tok else text


def _vectorize(texts: Sequence[str]):
    vec = TfidfVectorizer(stop_words=_STOP, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform([_features_text(t) for t in texts])
    return vec, X


def _top_terms(vec, X, member_idx: Sequence[int], n: int = 3) -> str:
    if not len(member_idx):
        return "empty"
    centroid = np.asarray(X[list(member_idx)].mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top = [t for t in terms[np.argsort(centroid)[::-1][: n * 4]] if "theme" not in t]
    # prefer unigrams, drop terms already covered by a chosen phrase
    picked: List[str] = []
    for t in top:
        if any(t in q or q in t for q in picked):
            continue
        picked.append(t)
        if len(picked) == n:
            break
    return ", ".join(picked)


def layout_2d(texts: Sequence[str]) -> np.ndarray:
    """2-D coordinates for a scatter view (LSA projection, deterministic)."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    _, X = _vectorize(texts)
    comps = min(2, X.shape[1] - 1) if X.shape[1] > 1 else 1
    if comps < 1:
        return np.zeros((n, 2))
    coords = TruncatedSVD(n_components=comps, random_state=0).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((n, 1))])
    return coords


@dataclass
class Clustering:
    """Result of one AI clustering pass over the *unpinned* gaps."""
    assign: Dict[str, str] = field(default_factory=dict)   # gap id -> cluster id ("C1"..)
    labels: Dict[str, str] = field(default_factory=dict)   # cluster id -> auto name (top terms)


def cluster_gaps(gaps: Sequence[dict], k: int, exclude_ids: Optional[Set[str]] = None) -> Clustering:
    """Cluster gaps with TF-IDF + KMeans, skipping ``exclude_ids`` (human-pinned)."""
    exclude_ids = exclude_ids or set()
    free = [g for g in gaps if g["id"] not in exclude_ids]
    if not free:
        return Clustering()
    k = max(1, min(k, len(free)))
    texts = [g["text"] for g in free]
    vec, X = _vectorize(texts)
    if k == 1 or len(free) == 1:
        raw = np.zeros(len(free), dtype=int)
    else:
        raw = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    # order clusters by size (desc) so C1 is always the biggest -> stable, readable ids
    order = sorted(set(raw), key=lambda c: (-int((raw == c).sum()), int(c)))
    remap = {c: f"C{i + 1}" for i, c in enumerate(order)}
    result = Clustering()
    for g, c in zip(free, raw):
        result.assign[g["id"]] = remap[c]
    for c in order:
        members = [i for i, r in enumerate(raw) if r == c]
        result.labels[remap[c]] = _top_terms(vec, X, members)
    return result


def align_to_previous(new: Clustering, old_assign: Dict[str, str]) -> Clustering:
    """Rename the new clusters so they line up with the previous ones.

    KMeans labels are arbitrary between runs. Without alignment, "what moved?" is
    meaningless. We match new clusters to old ones by member overlap (Hungarian
    algorithm), so a cluster the user already knows keeps its id.
    """
    old_ids = sorted({v for v in old_assign.values()})
    new_ids = sorted(set(new.assign.values()))
    if not old_ids or not new_ids:
        return new
    overlap = np.zeros((len(new_ids), len(old_ids)), dtype=int)
    for gid, nc in new.assign.items():
        oc = old_assign.get(gid)
        if oc in old_ids:
            overlap[new_ids.index(nc), old_ids.index(oc)] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    for r, c in zip(rows, cols):
        if overlap[r, c] > 0:
            mapping[new_ids[r]] = old_ids[c]
            used.add(old_ids[c])
    # unmatched new clusters get fresh ids that don't collide
    counter = 1
    taken = set(old_ids) | set(mapping.values())
    for nc in new_ids:
        if nc in mapping:
            continue
        while f"C{counter}" in taken:
            counter += 1
        mapping[nc] = f"C{counter}"
        taken.add(mapping[nc])
    return Clustering(
        assign={g: mapping[c] for g, c in new.assign.items()},
        labels={mapping[c]: lbl for c, lbl in new.labels.items()},
    )


def diff_assignments(old: Dict[str, str], new: Dict[str, str]) -> List[dict]:
    """List gaps whose cluster would change (used for the preview-before-apply step)."""
    moves = []
    for gid, nc in new.items():
        oc = old.get(gid)
        if oc != nc:
            moves.append({"id": gid, "from": oc, "to": nc})
    return moves


# --------------------------------------------------------------------------- #
# 3. Human-in-the-loop state
# --------------------------------------------------------------------------- #

@dataclass
class ExplorerState:
    """Everything the human and the AI co-own. Plain data -> easy to store/test."""
    gaps: List[dict]
    k: int = 3
    ai_assign: Dict[str, str] = field(default_factory=dict)    # AI proposal (provisional)
    ai_labels: Dict[str, str] = field(default_factory=dict)    # AI auto names
    pinned: Dict[str, str] = field(default_factory=dict)       # gap id -> human group name
    renamed: Dict[str, str] = field(default_factory=dict)      # cluster id -> human name
    starred: Set[str] = field(default_factory=set)
    removed: Set[str] = field(default_factory=set)
    history: List[str] = field(default_factory=list)           # human-readable action log

    # ---- construction ---- #
    @classmethod
    def create(cls, gaps: List[dict], k: int = 3) -> "ExplorerState":
        st = cls(gaps=list(gaps), k=max(1, min(k, max(1, len(gaps)))))
        st.recluster(st.k, log=False)
        st.history.append(f"AI grouped {len(gaps)} gaps into {len(set(st.ai_assign.values()))} clusters")
        return st

    # ---- views ---- #
    @property
    def active(self) -> List[dict]:
        return [g for g in self.gaps if g["id"] not in self.removed]

    def group_of(self, gap_id: str) -> str:
        """Display group for a gap: human group > human-renamed cluster > AI label."""
        if gap_id in self.pinned:
            return self.pinned[gap_id]
        cid = self.ai_assign.get(gap_id)
        if cid is None:
            return "Unassigned"
        return self.cluster_name(cid)

    def cluster_name(self, cid: str) -> str:
        return self.renamed.get(cid) or f"{cid}: {self.ai_labels.get(cid, '')}".strip(": ")

    def groups(self) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for g in self.active:
            out.setdefault(self.group_of(g["id"]), []).append(g)
        return out

    # ---- human actions ---- #
    def pin_to_group(self, gap_ids: Sequence[str], group_name: str) -> int:
        group_name = group_name.strip()
        if not group_name:
            raise ValueError("Group name cannot be empty")
        n = 0
        for gid in gap_ids:
            if gid in {g["id"] for g in self.gaps}:
                self.pinned[gid] = group_name
                n += 1
        if n:
            self.history.append(f"You grouped {n} gap(s) as '{group_name}'")
        return n

    def unpin(self, gap_ids: Sequence[str]) -> int:
        n = 0
        for gid in gap_ids:
            if self.pinned.pop(gid, None) is not None:
                n += 1
        if n:
            self.history.append(f"You returned {n} gap(s) to AI grouping")
        return n

    def rename_cluster(self, cid: str, name: str) -> None:
        name = name.strip()
        if not name:
            self.renamed.pop(cid, None)
        else:
            self.renamed[cid] = name
            self.history.append(f"You renamed {cid} to '{name}'")

    def toggle_star(self, gap_id: str) -> None:
        self.starred ^= {gap_id}

    def remove(self, gap_ids: Sequence[str]) -> None:
        self.removed.update(gap_ids)
        for gid in gap_ids:
            self.pinned.pop(gid, None)
        if gap_ids:
            self.history.append(f"You removed {len(gap_ids)} gap(s)")

    def restore_all(self) -> None:
        if self.removed:
            self.history.append(f"You restored {len(self.removed)} removed gap(s)")
        self.removed.clear()

    # ---- AI actions ---- #
    def propose_recluster(self, k: int) -> Clustering:
        """AI proposal for a new k. Does NOT change state; use for preview."""
        exclude = set(self.pinned) | set(self.removed)
        proposal = cluster_gaps(self.gaps, k, exclude_ids=exclude)
        return align_to_previous(proposal, self.ai_assign)

    def preview_recluster(self, k: int) -> List[dict]:
        proposal = self.propose_recluster(k)
        # Only compare gaps the AI is allowed to move.
        old = {g: c for g, c in self.ai_assign.items() if g not in self.pinned and g not in self.removed}
        return diff_assignments(old, proposal.assign)

    def recluster(self, k: int, log: bool = True) -> int:
        """Apply AI clustering. Human-pinned and removed gaps are left alone."""
        proposal = self.propose_recluster(k)
        moves = diff_assignments(
            {g: c for g, c in self.ai_assign.items() if g not in self.pinned and g not in self.removed},
            proposal.assign,
        )
        # keep AI assignments of pinned gaps (so unpinning later restores a sensible home)
        for gid in list(self.ai_assign):
            if gid not in self.pinned and gid not in self.removed:
                self.ai_assign.pop(gid, None)
        self.ai_assign.update(proposal.assign)
        self.ai_labels = dict(proposal.labels) | {
            c: l for c, l in self.ai_labels.items() if c not in proposal.labels and c in set(self.ai_assign.values())
        }
        # drop renames for clusters that no longer exist
        live = set(self.ai_assign.values())
        self.renamed = {c: n for c, n in self.renamed.items() if c in live}
        self.k = k
        if log:
            self.history.append(f"AI re-clustered into k={k} ({len(moves)} gap(s) moved; your groups untouched)")
        return len(moves)

    # ---- export ---- #
    def to_markdown(self) -> str:
        lines = ["# Research gap map", ""]
        for name, items in sorted(self.groups().items(), key=lambda kv: -len(kv[1])):
            tag = " (your group)" if any(g["id"] in self.pinned for g in items) else ""
            lines.append(f"## {name}{tag}  ·  {len(items)} gap(s)")
            for g in items:
                star = "★ " if g["id"] in self.starred else ""
                lines.append(f"- {star}{g['text']}  \n  _Source: {source_label(g)}_")
            lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. Adapter: gaps found by the GapDetection *agent* -> explorer items
# --------------------------------------------------------------------------- #

def gaps_from_agent_output(agent_gaps: Sequence[dict]) -> List[dict]:
    """Turn GapDetectionAgent gaps (any type) into explorer items with provenance."""
    items: List[dict] = []
    for g in agent_gaps:
        text = " ".join(str(g.get("description", "")).split())
        if not text:
            continue
        kind = g.get("type", "gap")
        items.append({
            "id": f"g{len(items) + 1}",
            "text": text,
            "kind": kind,
            "cues": [],
            "theme": g.get("theme") if kind == "stated_limitation" else _theme_of(text),
            "paper_title": g.get("paper_title") or "Cross-paper analysis",
            "paper_url": g.get("paper_url", ""),
            "year": g.get("year"),
            "note": g.get("evidence", ""),
        })
    return items


def _normalise(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape, 0.5)
    for c in range(xy.shape[1]):
        lo, hi = float(xy[:, c].min()), float(xy[:, c].max())
        if hi - lo > 1e-9:
            out[:, c] = (xy[:, c] - lo) / (hi - lo)
    return out


# --------------------------------------------------------------------------- #
# 5. Session service (framework-free; used by the FastAPI app and Streamlit)
# --------------------------------------------------------------------------- #

class GapExplorerService:
    """In-memory store of human+AI gap-organisation sessions.

    Sessions live as long as the server process (bounded to MAX_SESSIONS, oldest
    evicted). Nothing is persisted: this is documented in the README.
    """
    MAX_SESSIONS = 25
    MAX_NAME_LEN = 80
    MAX_K = 8

    def __init__(self) -> None:
        self._sessions: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.RLock()

    # ---- lifecycle ---- #
    def create(self, agent_gaps: Sequence[dict], topic: str = "") -> Optional[str]:
        items = gaps_from_agent_output(agent_gaps)
        if len(items) < 2:
            return None
        xy = _normalise(layout_2d([g["text"] for g in items]))
        coords = {g["id"]: (float(xy[i][0]), float(xy[i][1])) for i, g in enumerate(items)}
        state = ExplorerState.create(items, k=min(4, len(items)))
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[sid] = {"state": state, "coords": coords, "topic": topic}
            while len(self._sessions) > self.MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return sid

    def _get(self, sid: str) -> dict:
        with self._lock:
            if sid not in self._sessions:
                raise KeyError(sid)
            return self._sessions[sid]

    def get_state(self, sid: str) -> ExplorerState:
        return self._get(sid)["state"]

    def get_coords(self, sid: str) -> Dict[str, tuple]:
        return self._get(sid)["coords"]

    def topic(self, sid: str) -> str:
        return self._get(sid).get("topic", "")

    # ---- helpers ---- #
    @staticmethod
    def _max_k(st: ExplorerState) -> int:
        free = len([g for g in st.active if g["id"] not in st.pinned])
        return max(1, min(GapExplorerService.MAX_K, free))

    def _clean_ids(self, st: ExplorerState, ids: Sequence[str]) -> List[str]:
        known = {g["id"] for g in st.gaps}
        return [i for i in dict.fromkeys(str(x) for x in ids) if i in known]

    # ---- views ---- #
    def view(self, sid: str) -> Dict[str, Any]:
        with self._lock:
            sess = self._get(sid)
            st: ExplorerState = sess["state"]
            coords = sess["coords"]
            human_names = set(st.pinned.values())
            items = []
            for g in st.active:
                grp = st.group_of(g["id"])
                items.append({
                    "id": g["id"], "text": g["text"], "kind": g.get("kind", ""),
                    "theme": g.get("theme", ""), "paper_title": g["paper_title"],
                    "paper_url": g.get("paper_url", ""), "year": g.get("year"),
                    "note": g.get("note", ""), "group": grp,
                    "human": g["id"] in st.pinned, "starred": g["id"] in st.starred,
                    "x": coords[g["id"]][0], "y": coords[g["id"]][1],
                })
            groups = [{"name": n, "count": len(v), "human": n in human_names}
                      for n, v in sorted(st.groups().items(), key=lambda kv: -len(kv[1]))]
            ai_clusters = [{"id": c, "name": st.cluster_name(c)} for c in sorted(
                {st.ai_assign[g["id"]] for g in st.active if g["id"] not in st.pinned and g["id"] in st.ai_assign})]
            return {
                "session_id": sid, "topic": sess.get("topic", ""),
                "k": min(st.k, self._max_k(st)), "max_k": self._max_k(st),
                "counts": {"gaps": len(st.active), "groups": len(groups),
                           "pinned": len([g for g in st.active if g["id"] in st.pinned]),
                           "starred": len([g for g in st.active if g["id"] in st.starred]),
                           "removed": len(st.removed)},
                "items": items,
                "removed": [{"id": g["id"], "text": g["text"]} for g in st.gaps if g["id"] in st.removed],
                "groups": groups, "ai_clusters": ai_clusters,
                "history": st.history[-25:],
            }

    def preview(self, sid: str, k: int) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            max_k = self._max_k(st)
            k = max(1, min(int(k), max_k))
            proposal = st.propose_recluster(k)
            old = {g: c for g, c in st.ai_assign.items() if g not in st.pinned and g not in st.removed}
            by_id = {g["id"]: g for g in st.gaps}
            moves = []
            for mv in diff_assignments(old, proposal.assign):
                moves.append({
                    "id": mv["id"], "text": by_id[mv["id"]]["text"],
                    "from": st.cluster_name(mv["from"]) if mv["from"] else "unassigned",
                    "to": st.renamed.get(mv["to"]) or f"{mv['to']}: {proposal.labels.get(mv['to'], '')}",
                })
            return {"k": k, "moves": moves, "moved": len(moves),
                    "ai_grouped": len(proposal.assign),
                    "human_grouped": len([g for g in st.active if g["id"] in st.pinned]),
                    "unchanged": k == min(st.k, max_k)}

    # ---- actions ---- #
    def recluster(self, sid: str, k: int) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            k = max(1, min(int(k), self._max_k(st)))
            moved = st.recluster(k)
            out = self.view(sid)
            out["moved"] = moved
            return out

    def group(self, sid: str, ids: Sequence[str], name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("Group name cannot be empty")
        if len(name) > self.MAX_NAME_LEN:
            raise ValueError(f"Group name too long (max {self.MAX_NAME_LEN})")
        with self._lock:
            st = self.get_state(sid)
            ids = [i for i in self._clean_ids(st, ids) if i not in st.removed]
            if not ids:
                raise ValueError("No valid gaps selected")
            st.pin_to_group(ids, name)
            return self.view(sid)

    def unpin(self, sid: str, ids: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            st.unpin(self._clean_ids(st, ids))
            return self.view(sid)

    def rename_cluster(self, sid: str, cluster_name_or_id: str, name: str) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            cid = cluster_name_or_id.split(":")[0].strip()
            if cid not in set(st.ai_assign.values()):
                raise ValueError("Unknown cluster")
            if len((name or "").strip()) > self.MAX_NAME_LEN:
                raise ValueError(f"Name too long (max {self.MAX_NAME_LEN})")
            st.rename_cluster(cid, name or "")
            return self.view(sid)

    def toggle_star(self, sid: str, ids: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            for i in self._clean_ids(st, ids):
                st.toggle_star(i)
            return self.view(sid)

    def remove(self, sid: str, ids: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            st = self.get_state(sid)
            st.remove(self._clean_ids(st, ids))
            return self.view(sid)

    def restore_all(self, sid: str) -> Dict[str, Any]:
        with self._lock:
            self.get_state(sid).restore_all()
            return self.view(sid)

    def export_markdown(self, sid: str) -> str:
        with self._lock:
            return self.get_state(sid).to_markdown()
