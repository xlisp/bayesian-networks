"""Bayesian belief-network planner for scheme design & retrospective.

Use case: LLM training scheme analysis. Model each design choice as a node,
record outcomes (success/fail), let Bayesian updates guide you to the optimal
path — and warn you when exploration diverges too far from what's working.

CLI:
  python planner.py init <name> --goal "..."
  python planner.py use <name>
  python planner.py projects
  python planner.py add <name> [--parent jim0] [--prior 0.6] [--desc "..."]
  python planner.py observe <code> <success|fail> [--notes "..."]
  python planner.py list
  python planner.py show <code>
  python planner.py path
  python planner.py focus
  python planner.py check
  python planner.py rollback <code>
  python planner.py graph [--out graph] [--format png|svg]
"""

import argparse
import sys
from pathlib import Path

import db
import bayes
import viz


def require_project(conn):
    proj = db.get_active_project(conn)
    if not proj:
        print("没有活动项目。先运行: python planner.py init <name> --goal ...")
        sys.exit(1)
    return proj


def cmd_init(args):
    db.init_db()
    with db.connect() as conn:
        if db.find_project(conn, args.name):
            print(f"项目 '{args.name}' 已存在。使用 `use` 切换。")
            sys.exit(1)
        proj = db.create_project(conn, args.name, args.goal)
        print(f"创建并切换到项目: {proj['name']} (id={proj['id']})")
        print(f"目标: {proj['goal']}")


def cmd_use(args):
    with db.connect() as conn:
        proj = db.find_project(conn, args.name)
        if not proj:
            print(f"未找到项目 '{args.name}'")
            sys.exit(1)
        db.set_active_project(conn, proj["id"])
        print(f"已切换到项目: {proj['name']}")


def cmd_projects(_args):
    db.init_db()
    with db.connect() as conn:
        active = db.get_active_project(conn)
        active_id = active["id"] if active else None
        for p in db.list_projects(conn):
            mark = "*" if p["id"] == active_id else " "
            print(f"{mark} {p['name']}  —  {p['goal'] or ''}")


def cmd_add(args):
    with db.connect() as conn:
        proj = require_project(conn)
        node = db.add_node(
            conn,
            project_id=proj["id"],
            name=args.name,
            description=args.desc or "",
            parent_code=args.parent,
            prior=args.prior,
        )
        parent = f" <- {args.parent}" if args.parent else " (root)"
        print(f"添加节点 {node['code']}: {node['name']}{parent}  prior={args.prior}")


def cmd_observe(args):
    with db.connect() as conn:
        proj = require_project(conn)
        node = db.get_node(conn, proj["id"], args.code)
        if not node:
            print(f"节点 '{args.code}' 不存在")
            sys.exit(1)
        alpha, beta = bayes.observe(node["alpha"], node["beta"], args.result)
        status = bayes.status_from_belief(alpha, beta, node["status"])
        db.record_observation(conn, node["id"], args.result, args.notes or "")
        db.update_node_belief(conn, node["id"], alpha, beta, status)
        new_belief = alpha / (alpha + beta)
        print(
            f"{node['code']} ({node['name']}): {args.result}  "
            f"-> belief {new_belief:.3f}  status={status}"
        )
        if args.result == "fail" and new_belief <= bayes.LOW_BELIEF_THRESHOLD:
            parent_code = "root"
            if node["parent_id"]:
                p = conn.execute(
                    "SELECT code FROM nodes WHERE id=?", (node["parent_id"],)
                ).fetchone()
                parent_code = p["code"]
            print(f"  ⚠ 回归提醒: 建议回到 {parent_code}, 尝试其他分支。")


def _load_views(conn, project_id):
    rows = db.list_nodes(conn, project_id)
    return [bayes.to_view(r) for r in rows]


def cmd_list(_args):
    with db.connect() as conn:
        proj = require_project(conn)
        views = _load_views(conn, proj["id"])
        if not views:
            print("(还没有节点。用 `add` 添加第一个。)")
            return
        print(f"项目: {proj['name']}  目标: {proj['goal'] or ''}")
        print(f"{'code':<6} {'status':<9} {'belief':>7} {'obs':>4}  name")
        print("-" * 60)
        for n in views:
            print(
                f"{n.code:<6} {n.status:<9} {n.local_belief:>7.3f} "
                f"{n.evidence:>4}  {n.name}"
            )


def cmd_show(args):
    with db.connect() as conn:
        proj = require_project(conn)
        node = db.get_node(conn, proj["id"], args.code)
        if not node:
            print(f"节点 '{args.code}' 不存在")
            sys.exit(1)
        v = bayes.to_view(node)
        views = _load_views(conn, proj["id"])
        print(f"{v.code}: {v.name}")
        print(f"  描述: {v.description or '(无)'}")
        print(f"  状态: {v.status}")
        print(f"  prior: {v.prior:.3f}   posterior belief: {v.local_belief:.3f}")
        print(f"  alpha={v.alpha:.2f} beta={v.beta:.2f} (Beta posterior)")
        print(f"  path belief (chain): {bayes.path_belief(views, v):.3f}")
        obs = db.list_observations(conn, node["id"])
        if obs:
            print("  观察记录:")
            for o in obs:
                print(f"    [{o['created_at']}] {o['result']}  {o['notes'] or ''}")
        else:
            print("  观察记录: (无)")


def cmd_path(_args):
    with db.connect() as conn:
        proj = require_project(conn)
        views = _load_views(conn, proj["id"])
        path = bayes.optimal_path(views)
        if not path:
            print("(还没有节点。)")
            return
        print(f"当前最优路径 (贪心 · 按 belief):")
        for i, n in enumerate(path):
            prefix = "  " * i + ("└─ " if i else "")
            print(f"{prefix}{n.code} {n.name}  belief={n.local_belief:.3f}")
        print(f"\n全链联合信念: {bayes.path_belief(views, path[-1]):.3f}")


def cmd_focus(_args):
    with db.connect() as conn:
        proj = require_project(conn)
        views = _load_views(conn, proj["id"])
        n = bayes.next_focus(views)
        if not n:
            print("没有待验证的节点 — 要么全部验证完，要么用 `add` 继续扩展。")
            return
        print(f"下一步聚焦: {n.code}  {n.name}")
        print(f"  当前 belief={n.local_belief:.3f}  obs={n.evidence}")
        print(f"  行动: 运行这个方案，然后 `observe {n.code} success|fail --notes ...`")


def cmd_check(_args):
    with db.connect() as conn:
        proj = require_project(conn)
        views = _load_views(conn, proj["id"])
        warns = bayes.divergence_warnings(views)
        if not warns:
            print("✓ 网络健康: 没有发散或低信念分支需要处理。")
            return
        for w in warns:
            print("⚠ " + w)


def cmd_rollback(args):
    with db.connect() as conn:
        proj = require_project(conn)
        touched = db.rollback_subtree(conn, proj["id"], args.code)
        print(f"已回滚 {args.code} 及其 {len(touched)-1} 个后代节点 (标记为 obsolete)。")
        print("提示: 它们仍在历史中可查, 但不再参与最优路径计算。")


def cmd_graph(args):
    with db.connect() as conn:
        proj = require_project(conn)
        views = _load_views(conn, proj["id"])
        if not views:
            print("(还没有节点。)")
            return
        warnings = bayes.divergence_warnings(views)
        dot = viz.build_dot(views, proj["name"], proj["goal"] or "")
        out = Path(args.out).resolve()
        result = viz.render(dot, out, fmt=args.format)
        print(f"已生成: {result}")
        if warnings:
            print("\n附加提醒:")
            for w in warnings:
                print("  ⚠ " + w)


def build_parser():
    p = argparse.ArgumentParser(prog="planner", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="创建并激活一个项目")
    s.add_argument("name")
    s.add_argument("--goal", default="", help="项目目标")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("use", help="切换活动项目")
    s.add_argument("name")
    s.set_defaults(func=cmd_use)

    s = sub.add_parser("projects", help="列出所有项目")
    s.set_defaults(func=cmd_projects)

    s = sub.add_parser("add", help="添加节点 (方案/假设)")
    s.add_argument("name")
    s.add_argument("--parent", default=None, help="父节点 code, 例 jim0")
    s.add_argument("--prior", type=float, default=0.5, help="先验 [0,1]")
    s.add_argument("--desc", default="", help="描述")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("observe", help="记录观察并更新信念")
    s.add_argument("code")
    s.add_argument("result", choices=["success", "fail"])
    s.add_argument("--notes", default="", help="备注")
    s.set_defaults(func=cmd_observe)

    s = sub.add_parser("list", help="列出节点")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="节点详情 + 观察历史")
    s.add_argument("code")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("path", help="当前最优路径")
    s.set_defaults(func=cmd_path)

    s = sub.add_parser("focus", help="下一步应该做什么")
    s.set_defaults(func=cmd_focus)

    s = sub.add_parser("check", help="发散 / 回归检查")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("rollback", help="将节点及后代标记为 obsolete")
    s.add_argument("code")
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser("graph", help="生成 graphviz 图")
    s.add_argument("--out", default="graph", help="输出路径 (不含后缀)")
    s.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    s.set_defaults(func=cmd_graph)

    return p


def main():
    db.init_db()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
