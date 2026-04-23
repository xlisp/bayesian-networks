"""Bayesian belief updates + network analysis.

Each node's belief (probability it's on the right path) is modelled with a
Beta(alpha, beta) posterior. Expected value = alpha / (alpha + beta).

Prior is encoded into initial (alpha, beta). Each observation is a Bernoulli
trial: success increments alpha, failure increments beta. This is the standard
conjugate update for Beta/Bernoulli.

A node's effective belief also depends on the parent chain: the hypothesis
`jim_k is correct` is only useful if its ancestors are correct too. We propagate
this with a product over the ancestor chain (independence assumption — a
reasonable first-order model for planning; users can revise with observations).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

DIVERGENCE_SIBLING_THRESHOLD = 3  # >=N untested children under one parent
LOW_BELIEF_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class NodeView:
    id: int
    code: str
    name: str
    description: str
    parent_id: Optional[int]
    status: str
    alpha: float
    beta: float
    prior: float

    @property
    def local_belief(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self) -> int:
        # alpha/beta start with prior weight; round to integer observation count
        return int(round(self.alpha + self.beta - 2))


def to_view(row) -> NodeView:
    return NodeView(
        id=row["id"],
        code=row["code"],
        name=row["name"],
        description=row["description"] or "",
        parent_id=row["parent_id"],
        status=row["status"],
        alpha=row["alpha"],
        beta=row["beta"],
        prior=row["prior"],
    )


def observe(alpha: float, beta: float, result: str) -> tuple[float, float]:
    if result == "success":
        return alpha + 1.0, beta
    if result == "fail":
        return alpha, beta + 1.0
    raise ValueError(f"result must be 'success' or 'fail', got {result!r}")


def status_from_belief(alpha: float, beta: float, current: str) -> str:
    """Derive a status label from posterior. 'obsolete' is sticky."""
    if current == "obsolete":
        return "obsolete"
    belief = alpha / (alpha + beta)
    evidence = alpha + beta - 2
    if evidence < 1:
        return "pending"
    if belief >= HIGH_CONFIDENCE_THRESHOLD:
        return "success"
    if belief <= LOW_BELIEF_THRESHOLD:
        return "fail"
    return "active"


def build_index(nodes: List[NodeView]) -> Dict[int, NodeView]:
    return {n.id: n for n in nodes}


def children_of(nodes: List[NodeView], parent_id: Optional[int]) -> List[NodeView]:
    return [n for n in nodes if n.parent_id == parent_id]


def ancestors(nodes: List[NodeView], node: NodeView) -> List[NodeView]:
    idx = build_index(nodes)
    chain = []
    cur = node
    while cur.parent_id is not None:
        cur = idx[cur.parent_id]
        chain.append(cur)
    return chain  # nearest-parent first


def path_belief(nodes: List[NodeView], node: NodeView) -> float:
    """Belief that the chain root -> ... -> node is all correct."""
    b = node.local_belief
    for anc in ancestors(nodes, node):
        b *= anc.local_belief
    return b


def optimal_path(nodes: List[NodeView]) -> List[NodeView]:
    """Greedy: from each root, pick the live child with highest local belief."""
    roots = children_of(nodes, None)
    if not roots:
        return []
    roots = [r for r in roots if r.status != "obsolete"] or roots
    start = max(roots, key=lambda n: n.local_belief)
    path = [start]
    cur = start
    while True:
        kids = [c for c in children_of(nodes, cur.id) if c.status != "obsolete"]
        if not kids:
            break
        nxt = max(kids, key=lambda n: n.local_belief)
        path.append(nxt)
        cur = nxt
    return path


def divergence_warnings(nodes: List[NodeView]) -> List[str]:
    """Flag parents with too many untested siblings, and low-belief actives."""
    warnings = []
    idx = build_index(nodes)

    # Parents with many pending children
    by_parent: Dict[Optional[int], List[NodeView]] = {}
    for n in nodes:
        if n.status == "obsolete":
            continue
        by_parent.setdefault(n.parent_id, []).append(n)

    for pid, kids in by_parent.items():
        pending = [k for k in kids if k.evidence == 0]
        if len(pending) >= DIVERGENCE_SIBLING_THRESHOLD:
            parent_name = "root" if pid is None else f"{idx[pid].code} ({idx[pid].name})"
            codes = ", ".join(k.code for k in pending)
            warnings.append(
                f"发散警告: {parent_name} 下有 {len(pending)} 个未验证分支 "
                f"[{codes}] — 建议聚焦一条先验证，再扩展。"
            )

    # Low-belief active nodes suggest rollback
    for n in nodes:
        if n.status == "obsolete":
            continue
        if n.evidence >= 1 and n.local_belief <= LOW_BELIEF_THRESHOLD:
            parent = idx.get(n.parent_id) if n.parent_id else None
            back_to = parent.code if parent else "root"
            warnings.append(
                f"回归提醒: {n.code} ({n.name}) 信念 {n.local_belief:.2f} 已低于阈值 "
                f"{LOW_BELIEF_THRESHOLD} — 建议回到 {back_to}, 尝试其他分支。"
            )

    return warnings


def next_focus(nodes: List[NodeView]) -> Optional[NodeView]:
    """Suggest the single node to work on next.

    Prefer: a pending or active node on the optimal path, with fewest observations
    (so new evidence has the most impact).
    """
    path = optimal_path(nodes)
    candidates = [n for n in path if n.status in ("pending", "active")]
    if not candidates:
        candidates = [n for n in nodes
                      if n.status in ("pending", "active")]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (n.evidence, -n.local_belief))
