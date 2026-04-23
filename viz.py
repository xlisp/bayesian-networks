"""Graphviz DOT generation for the belief network."""

import shutil
import subprocess
from pathlib import Path
from typing import List, Set

from bayes import NodeView, optimal_path

STATUS_COLOR = {
    "pending": "#e0e0e0",
    "active": "#fff4b3",
    "success": "#a8e6a3",
    "fail": "#f4a5a5",
    "obsolete": "#d0d0d0",
}


def escape(s: str) -> str:
    return s.replace('"', '\\"').replace("\n", "\\l")


def node_label(n: NodeView) -> str:
    b = n.local_belief
    ev = n.evidence
    desc = f"\\n{escape(n.description)}" if n.description else ""
    return (
        f"{n.code}: {escape(n.name)}"
        f"{desc}"
        f"\\nbelief={b:.2f}  obs={ev}  status={n.status}"
    )


def build_dot(nodes: List[NodeView], project_name: str, goal: str = "") -> str:
    path_ids: Set[int] = {n.id for n in optimal_path(nodes)}
    lines = [
        f'digraph "{escape(project_name)}" {{',
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    if goal:
        lines.append(
            f'  label="{escape(project_name)} — goal: {escape(goal)}";'
            ' labelloc="t"; fontsize=14;'
        )

    for n in nodes:
        color = STATUS_COLOR.get(n.status, "#ffffff")
        penwidth = 3 if n.id in path_ids else 1
        style = '"rounded,filled,bold"' if n.id in path_ids else '"rounded,filled"'
        dashed = ',style="rounded,filled,dashed"' if n.status == "obsolete" else ''
        lines.append(
            f'  n{n.id} [label="{node_label(n)}", fillcolor="{color}", '
            f'penwidth={penwidth}, style={style}{dashed}];'
        )

    for n in nodes:
        if n.parent_id is None:
            continue
        bold = ' [penwidth=2.5, color="#333333"]' if (
            n.id in path_ids and n.parent_id in path_ids
        ) else ''
        lines.append(f'  n{n.parent_id} -> n{n.id}{bold};')

    lines.append("}")
    return "\n".join(lines)


def render(dot_source: str, out_path: Path, fmt: str = "png") -> Path:
    """Write DOT and, if `dot` is installed, render to `fmt`."""
    dot_file = out_path.with_suffix(".dot")
    dot_file.write_text(dot_source)
    if not shutil.which("dot"):
        return dot_file
    img = out_path.with_suffix(f".{fmt}")
    subprocess.run(
        ["dot", f"-T{fmt}", str(dot_file), "-o", str(img)],
        check=True,
    )
    return img
