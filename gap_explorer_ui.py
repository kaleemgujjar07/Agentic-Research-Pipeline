"""Streamlit UI for the interactive Gap Explorer (human-AI co-organisation of research gaps)."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from autoresearch.modules.gap_explorer import ExplorerState, source_label

_PALETTE = ["#2a6fdb", "#e0701a", "#1f9d6b", "#a24bd6", "#d6336c", "#8a6d1d", "#0e8fa8", "#6b7280"]
_NONCE = "gap_explorer_nonce"


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _selected_ids_from_event(event) -> list[str]:
    ids: list[str] = []
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return ids
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd:
            ids.append(str(cd))
    return ids


def _build_figure(state: ExplorerState, coords: dict[str, tuple[float, float]]) -> go.Figure:
    fig = go.Figure()
    groups = state.groups()
    pinned_names = set(state.pinned.values())
    for i, name in enumerate(sorted(groups)):
        items = groups[name]
        color = _PALETTE[i % len(_PALETTE)]
        is_human = name in pinned_names
        fig.add_trace(go.Scatter(
            x=[coords[g["id"]][0] for g in items],
            y=[coords[g["id"]][1] for g in items],
            mode="markers",
            name=("✋ " if is_human else "🤖 ") + _short(name, 38),
            customdata=[g["id"] for g in items],
            text=[f"<b>{html.escape(_short(g['text'], 160))}</b><br><i>{html.escape(_short(source_label(g), 70))}</i>" for g in items],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=[20 if g["id"] in state.starred else 13 for g in items],
                symbol=["star" if g["id"] in state.starred else ("diamond" if is_human else "circle") for g in items],
                color=color,
                line=dict(width=1, color="white"),
            ),
        ))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10), dragmode="lasso",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.05), hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_gap_explorer(state: ExplorerState, coords: dict[str, tuple[float, float]], session_id: str = "") -> None:
    """Render the co-creation panel for gaps the 7-agent pipeline detected.

    ``state``/``coords`` come from the same GapExplorerService the FastAPI app uses.
    """
    st.header("🧩 Gap Explorer: co-organise the detected gaps with the AI")
    st.caption(
        "The AI proposes a structure; you stay in control. Lasso-select gaps on the map to group them "
        "yourself, and the AI will never move gaps you've grouped. Every gap shows where it came from."
    )

    # Widget keys carry a counter so cleared selections really clear; reset it for a new pipeline run.
    if st.session_state.get("gap_explorer_sid") != session_id:
        st.session_state["gap_explorer_sid"] = session_id
        st.session_state[_NONCE] = 0
    nonce = st.session_state.get(_NONCE, 0)

    by_id = {g["id"]: g for g in state.gaps}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gaps (from abstracts)", len(state.active))
    m2.metric("Groups", len(state.groups()))
    m3.metric("Your groupings", len(state.pinned))
    m4.metric("Starred", len(state.starred & {g["id"] for g in state.active}))

    left, right = st.columns([3, 2], gap="large")

    # ------------------------------------------------------------- map + selection
    with left:
        st.markdown("**Gap map** · 🤖 circles = AI groups · ✋ diamonds = your groups · ★ = starred")
        event = st.plotly_chart(
            _build_figure(state, coords),
            key=f"gap_map_{nonce}",
            on_select="rerun",
            selection_mode=("lasso", "box", "points"),
            use_container_width=True,
        )
        from_map = [i for i in _selected_ids_from_event(event) if i in by_id and i not in state.removed]

        options = [g["id"] for g in state.active]
        from_list = st.multiselect(
            "…or pick gaps from a list",
            options=options,
            format_func=lambda i: f"{_short(by_id[i]['text'], 80)}  ·  {_short(by_id[i]['paper_title'], 25)}",
            key=f"gap_pick_{nonce}",
        )
        selected = list(dict.fromkeys(from_map + from_list))
        st.caption(f"{len(selected)} gap(s) selected")

        a, b = st.columns([3, 2])
        with a:
            new_group = st.text_input("Group name", placeholder="e.g. Data scarcity in clinical imaging",
                                      key=f"gap_group_name_{nonce}", label_visibility="collapsed")
        with b:
            if st.button("✋ Group selected", use_container_width=True, disabled=not selected):
                if not new_group.strip():
                    st.warning("Give the group a name first.")
                else:
                    state.pin_to_group(selected, new_group)
                    st.session_state[_NONCE] = nonce + 1
                    st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("★ Star / unstar", use_container_width=True, disabled=not selected):
            for i in selected:
                state.toggle_star(i)
            st.session_state[_NONCE] = nonce + 1
            st.rerun()
        if c2.button("↩ Return to AI", use_container_width=True,
                     disabled=not any(i in state.pinned for i in selected)):
            state.unpin(selected)
            st.session_state[_NONCE] = nonce + 1
            st.rerun()
        if c3.button("🗑 Remove", use_container_width=True, disabled=not selected):
            state.remove(selected)
            st.session_state[_NONCE] = nonce + 1
            st.rerun()

    # ------------------------------------------------------------- AI controls
    with right:
        st.markdown("**Ask the AI to re-organise**")
        max_k = max(2, min(8, len(state.active) - len(state.pinned)))
        cur_k = min(state.k, max_k)  # the slider's range shrinks as you pin gaps; don't fake a change
        k = st.slider("Number of AI clusters", 1, max_k, cur_k, key=f"gap_k_{nonce}")
        moves = state.preview_recluster(k) if k != cur_k else []
        free = len(state.active) - len(state.pinned)
        if k == cur_k:
            st.caption("This is the current setting. Move the slider to preview a change.")
        else:
            st.info(
                f"**Preview:** k={k} would move **{len(moves)}** of {free} AI-grouped gap(s). "
                f"Your {len(state.pinned)} grouped gap(s) stay exactly where you put them."
            )
            if moves:
                with st.expander("See what would move"):
                    for mv in moves:
                        g = by_id[mv["id"]]
                        st.markdown(f"- {_short(g['text'], 90)}  \n  `{mv['from']}` → `{mv['to']}`")
            if st.button("✅ Apply AI re-clustering", use_container_width=True):
                state.recluster(k)
                st.session_state[_NONCE] = nonce + 1
                st.rerun()

        st.markdown("**Rename an AI cluster**")
        cids = sorted(set(state.ai_assign[g["id"]] for g in state.active
                          if g["id"] not in state.pinned and g["id"] in state.ai_assign))
        if cids:
            cid = st.selectbox("Cluster", cids, format_func=lambda c: state.cluster_name(c), key=f"gap_rn_{nonce}")
            nm = st.text_input("New name", key=f"gap_rn_name_{nonce}", placeholder="leave empty to reset")
            if st.button("Rename", use_container_width=True):
                state.rename_cluster(cid, nm)
                st.session_state[_NONCE] = nonce + 1
                st.rerun()

        if state.removed and st.button(f"Restore {len(state.removed)} removed gap(s)", use_container_width=True):
            state.restore_all()
            st.session_state[_NONCE] = nonce + 1
            st.rerun()

    # ------------------------------------------------------------- grouped view
    st.markdown("---")
    st.subheader("Your gap map, grouped")
    for name, items in sorted(state.groups().items(), key=lambda kv: -len(kv[1])):
        mine = any(g["id"] in state.pinned for g in items)
        with st.expander(f"{'✋' if mine else '🤖'} {name}  ·  {len(items)} gap(s)", expanded=False):
            for g in items:
                star = "★ " if g["id"] in state.starred else ""
                url = g.get("paper_url", "")
                src = f"[{g['paper_title']}]({url})" if url.startswith(("http://", "https://")) else g["paper_title"]
                year = f" ({g['year']})" if g.get("year") else ""
                st.markdown(f"- {star}{g['text']}  \n  _Source: {src}{year} · {str(g.get('kind', '')).replace('_', ' ')}_")

    with st.expander("📜 What you and the AI have done so far"):
        for line in reversed(state.history):
            st.markdown(f"- {line}")

    st.download_button("📥 Download gap map (.md)", data=state.to_markdown(),
                       file_name="research_gap_map.md", mime="text/markdown")
