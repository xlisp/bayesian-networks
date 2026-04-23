"""Web UI for the Bayesian planner. Run: `python web.py` → http://127.0.0.1:5000

Thin Flask wrapper over db/bayes/viz. One page per project with:
  - project switcher / create
  - embedded SVG graph
  - focus, path, divergence warnings
  - node table (add / observe / rollback inline)
  - node detail drawer with observation history
"""

import shutil
import subprocess
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template_string,
    request,
    url_for,
)

import bayes
import db
import viz

app = Flask(__name__)
db.init_db()


# ---------- helpers ----------

def load_project_or_404(name):
    with db.connect() as conn:
        proj = db.find_project(conn, name)
    if not proj:
        abort(404, f"project {name!r} not found")
    return proj


def load_views(conn, project_id):
    return [bayes.to_view(r) for r in db.list_nodes(conn, project_id)]


def dot_to_svg(dot_source: str) -> str:
    """Render DOT to inline SVG. Strip the <?xml?> / <!DOCTYPE> header so it
    embeds cleanly in HTML."""
    if not shutil.which("dot"):
        return '<pre>(graphviz `dot` 未安装，只能显示 DOT 源码)\n' + dot_source + "</pre>"
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_source,
        capture_output=True,
        text=True,
        check=True,
    )
    svg = result.stdout
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


# ---------- templates ----------

BASE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
       margin: 0; background: #f6f7f9; color: #222; }
header { background: #222; color: #fff; padding: 12px 20px; display: flex;
         align-items: center; gap: 16px; }
header a { color: #9cf; text-decoration: none; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; }
header form { margin: 0; }
main { max-width: 1280px; margin: 0 auto; padding: 20px; }
.grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
.card { background: #fff; border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06); margin-bottom: 16px; }
.card h2 { margin: 0 0 12px 0; font-size: 15px;
           text-transform: uppercase; letter-spacing: 0.5px; color: #666; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 8px 6px; text-align: left; border-bottom: 1px solid #eee; }
th { color: #666; font-weight: 500; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.5px; }
td code { background: #eef; padding: 1px 6px; border-radius: 3px; font-size: 13px; }
.status { display: inline-block; padding: 2px 8px; border-radius: 10px;
          font-size: 11px; font-weight: 600; text-transform: uppercase; }
.status.pending  { background: #e0e0e0; color: #555; }
.status.active   { background: #fff4b3; color: #7a5a00; }
.status.success  { background: #a8e6a3; color: #1d5f1a; }
.status.fail     { background: #f4a5a5; color: #7a1a1a; }
.status.obsolete { background: #d0d0d0; color: #666; text-decoration: line-through; }
.belief-bar { width: 60px; height: 6px; background: #eee; border-radius: 3px;
              display: inline-block; vertical-align: middle; margin-right: 6px; }
.belief-bar > span { display: block; height: 100%; border-radius: 3px;
                     background: linear-gradient(90deg, #f4a5a5, #fff4b3, #a8e6a3); }
input, select, textarea { font: inherit; padding: 6px 8px;
                          border: 1px solid #ccc; border-radius: 4px;
                          background: #fff; }
textarea { width: 100%; resize: vertical; }
button { font: inherit; padding: 6px 14px; border: 0; border-radius: 4px;
         background: #2a6df4; color: #fff; cursor: pointer; font-weight: 500; }
button.small { padding: 3px 10px; font-size: 12px; }
button.danger { background: #c73a3a; }
button.success { background: #2c9542; }
button.ghost { background: #eee; color: #333; }
form.inline { display: inline; margin: 0; }
.warn { background: #fff4d6; border-left: 4px solid #f5a623; padding: 10px 14px;
        margin-bottom: 8px; border-radius: 4px; font-size: 14px; }
.ok   { background: #e6f6e6; border-left: 4px solid #2c9542; padding: 10px 14px;
        border-radius: 4px; font-size: 14px; }
.path-item { padding: 4px 0; }
.path-item .depth { color: #aaa; }
.graph-wrap { overflow: auto; max-height: 600px; border: 1px solid #eee;
              border-radius: 6px; background: #fafafa; padding: 10px; }
.graph-wrap svg { max-width: 100%; height: auto; }
.row-form { display: flex; gap: 6px; align-items: center; }
.row-form input[type=text] { flex: 1; }
label { font-size: 12px; color: #666; display: block; margin-top: 8px; }
.muted { color: #888; font-size: 12px; }
.node-obs li { margin: 4px 0; font-size: 13px; }
"""

LAYOUT = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>{{ title }} · Bayesian Planner</title>
  <style>{{ css|safe }}</style>
</head>
<body>
<header>
  <h1><a href="{{ url_for('home') }}">🧠 Bayesian Planner</a></h1>
  {% if projects %}
    <form method="post" action="{{ url_for('use_project') }}">
      <select name="name" onchange="this.form.submit()">
        {% for p in projects %}
          <option value="{{ p.name }}" {% if active and p.id == active.id %}selected{% endif %}>
            {{ p.name }}
          </option>
        {% endfor %}
      </select>
    </form>
  {% endif %}
  <span style="flex:1"></span>
  <a href="{{ url_for('projects_page') }}">+ 新项目</a>
</header>
<main>{{ body|safe }}</main>
</body>
</html>
"""

PROJECTS_PAGE = """
<div class="card">
  <h2>新建项目</h2>
  <form method="post" action="{{ url_for('create_project') }}">
    <div class="row-form">
      <input type="text" name="name" placeholder="项目名, 例 llm-train" required>
      <input type="text" name="goal" placeholder="目标, 例 loss<2.0" style="flex:2">
      <button type="submit">创建</button>
    </div>
  </form>
</div>
<div class="card">
  <h2>所有项目</h2>
  {% if projects %}
  <table>
    <tr><th></th><th>名称</th><th>目标</th><th>创建时间</th><th></th></tr>
    {% for p in projects %}
    <tr>
      <td>{% if active and p.id == active.id %}★{% endif %}</td>
      <td><a href="{{ url_for('project_view', name=p.name) }}">{{ p.name }}</a></td>
      <td>{{ p.goal or '' }}</td>
      <td class="muted">{{ p.created_at }}</td>
      <td>
        <form class="inline" method="post" action="{{ url_for('use_project') }}">
          <input type="hidden" name="name" value="{{ p.name }}">
          <button class="small ghost">设为活动</button>
        </form>
      </td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
    <p class="muted">还没有项目。</p>
  {% endif %}
</div>
"""

PROJECT_PAGE = """
<div class="card">
  <div style="display:flex; align-items:baseline; gap:12px;">
    <h2 style="margin:0; color:#222; font-size:20px; text-transform:none; letter-spacing:0;">
      {{ project.name }}
    </h2>
    <span class="muted">{{ project.goal or '（无目标）' }}</span>
  </div>
</div>

<div class="grid">
  <div>
    <div class="card">
      <h2>信念网络图</h2>
      {% if views %}
        <div class="graph-wrap">{{ svg|safe }}</div>
        <div class="muted" style="margin-top:8px;">
          粗框 = 当前最优路径 · 颜色 = 状态 · 虚线框 = obsolete
        </div>
      {% else %}
        <p class="muted">还没有节点，先在右边添加第一个。</p>
      {% endif %}
    </div>

    <div class="card">
      <h2>节点</h2>
      {% if views %}
      <table>
        <tr>
          <th>code</th><th>名称</th><th>状态</th>
          <th>belief</th><th>obs</th><th>父</th><th>操作</th>
        </tr>
        {% for n in views %}
        <tr>
          <td><a href="{{ url_for('node_view', name=project.name, code=n.code) }}"><code>{{ n.code }}</code></a></td>
          <td>{{ n.name }}</td>
          <td><span class="status {{ n.status }}">{{ n.status }}</span></td>
          <td>
            <span class="belief-bar"><span style="width: {{ (n.local_belief*100)|round(0, 'floor') }}%;"></span></span>
            {{ '%.2f' % n.local_belief }}
          </td>
          <td>{{ n.evidence }}</td>
          <td>
            {% if n.parent_id %}
              <code>{{ parent_code[n.parent_id] }}</code>
            {% else %}
              <span class="muted">root</span>
            {% endif %}
          </td>
          <td>
            <form class="inline" method="post"
                  action="{{ url_for('observe_node', name=project.name, code=n.code) }}">
              <input type="hidden" name="result" value="success">
              <button class="small success">✓</button>
            </form>
            <form class="inline" method="post"
                  action="{{ url_for('observe_node', name=project.name, code=n.code) }}">
              <input type="hidden" name="result" value="fail">
              <button class="small danger">✗</button>
            </form>
            {% if n.status != 'obsolete' %}
            <form class="inline" method="post"
                  action="{{ url_for('rollback_node', name=project.name, code=n.code) }}"
                  onsubmit="return confirm('rollback {{ n.code }} 和所有后代?');">
              <button class="small ghost">rollback</button>
            </form>
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </table>
      <p class="muted" style="margin-top:10px;">
        ✓/✗ 记录一次观察（不带备注）。要加备注点 <code>{{ 'jimN' }}</code> 进详情页。
      </p>
      {% else %}
        <p class="muted">还没有节点。</p>
      {% endif %}
    </div>
  </div>

  <div>
    <div class="card">
      <h2>下一步聚焦</h2>
      {% if focus %}
        <p><code>{{ focus.code }}</code> — <strong>{{ focus.name }}</strong></p>
        <p class="muted">belief={{ '%.2f' % focus.local_belief }} · obs={{ focus.evidence }}</p>
        <p>跑这个方案，然后在下方记录结果。</p>
      {% else %}
        <p class="muted">没有待验证节点。</p>
      {% endif %}
    </div>

    <div class="card">
      <h2>最优路径</h2>
      {% if path %}
        {% for n in path %}
          <div class="path-item">
            <span class="depth">{{ '  ' * loop.index0 }}{{ '└ ' if not loop.first else '' }}</span>
            <code>{{ n.code }}</code> {{ n.name }}
            <span class="muted">{{ '%.2f' % n.local_belief }}</span>
          </div>
        {% endfor %}
        <p class="muted" style="margin-top:8px;">
          全链联合信念：{{ '%.2f' % path_belief }}
        </p>
      {% else %}
        <p class="muted">无路径。</p>
      {% endif %}
    </div>

    <div class="card">
      <h2>发散 / 回归检查</h2>
      {% if warnings %}
        {% for w in warnings %}<div class="warn">⚠ {{ w }}</div>{% endfor %}
      {% else %}
        <div class="ok">✓ 网络健康</div>
      {% endif %}
    </div>

    <div class="card">
      <h2>添加节点</h2>
      <form method="post" action="{{ url_for('add_node_route', name=project.name) }}">
        <label>名称 (必填)</label>
        <input type="text" name="name" required style="width:100%">
        <label>父节点（留空为 root）</label>
        <select name="parent" style="width:100%">
          <option value="">（root）</option>
          {% for n in views if n.status != 'obsolete' %}
            <option value="{{ n.code }}">{{ n.code }} — {{ n.name }}</option>
          {% endfor %}
        </select>
        <label>先验 prior (0..1)</label>
        <input type="number" name="prior" value="0.5" step="0.05" min="0" max="1" style="width:100%">
        <label>描述</label>
        <textarea name="desc" rows="2"></textarea>
        <div style="margin-top:10px;"><button type="submit">添加</button></div>
      </form>
    </div>
  </div>
</div>
"""

NODE_PAGE = """
<div class="card">
  <h2>
    <a href="{{ url_for('project_view', name=project.name) }}">← 返回 {{ project.name }}</a>
  </h2>
  <h3><code>{{ node.code }}</code> {{ node.name }}
    <span class="status {{ node.status }}">{{ node.status }}</span>
  </h3>
  <p>{{ node.description or '（无描述）' }}</p>
  <p class="muted">
    prior={{ '%.3f' % node.prior }} ·
    posterior belief={{ '%.3f' % node.local_belief }} ·
    α={{ '%.2f' % node.alpha }} β={{ '%.2f' % node.beta }} ·
    path belief={{ '%.3f' % path_belief }}
  </p>

  <form method="post" action="{{ url_for('observe_node', name=project.name, code=node.code) }}">
    <label>记录观察（带备注）</label>
    <div class="row-form">
      <select name="result">
        <option value="success">success</option>
        <option value="fail">fail</option>
      </select>
      <input type="text" name="notes" placeholder="备注, 例 loss 从 3.5 降到 2.2" style="flex:1">
      <button type="submit">提交</button>
    </div>
  </form>
</div>

<div class="card">
  <h2>观察历史</h2>
  {% if observations %}
    <ul class="node-obs">
      {% for o in observations %}
        <li>
          <span class="status {{ 'success' if o.result == 'success' else 'fail' }}">{{ o.result }}</span>
          <span class="muted">{{ o.created_at }}</span>
          {{ o.notes or '' }}
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p class="muted">还没有观察记录。</p>
  {% endif %}
</div>
"""


def render_layout(title, body_template, **ctx):
    with db.connect() as conn:
        projects = db.list_projects(conn)
        active = db.get_active_project(conn)
    body = render_template_string(body_template, **ctx)
    return render_template_string(
        LAYOUT, title=title, css=BASE_CSS, body=body,
        projects=projects, active=active,
    )


# ---------- routes ----------

@app.route("/")
def home():
    with db.connect() as conn:
        active = db.get_active_project(conn)
    if active:
        return redirect(url_for("project_view", name=active["name"]))
    return redirect(url_for("projects_page"))


@app.route("/projects")
def projects_page():
    with db.connect() as conn:
        projects = db.list_projects(conn)
        active = db.get_active_project(conn)
    return render_layout(
        "Projects", PROJECTS_PAGE,
        projects=projects, active=active,
    )


@app.route("/projects/new", methods=["POST"])
def create_project():
    name = request.form["name"].strip()
    goal = request.form.get("goal", "").strip()
    if not name:
        abort(400, "name required")
    with db.connect() as conn:
        if db.find_project(conn, name):
            abort(400, f"项目 {name!r} 已存在")
        db.create_project(conn, name, goal)
    return redirect(url_for("project_view", name=name))


@app.route("/projects/use", methods=["POST"])
def use_project():
    name = request.form["name"]
    with db.connect() as conn:
        proj = db.find_project(conn, name)
        if not proj:
            abort(404)
        db.set_active_project(conn, proj["id"])
    return redirect(url_for("project_view", name=name))


@app.route("/p/<name>")
def project_view(name):
    proj = load_project_or_404(name)
    with db.connect() as conn:
        db.set_active_project(conn, proj["id"])
        views = load_views(conn, proj["id"])
    parent_code = {v.id: v.code for v in views}
    path = bayes.optimal_path(views)
    focus = bayes.next_focus(views)
    warnings = bayes.divergence_warnings(views)
    svg = dot_to_svg(viz.build_dot(views, proj["name"], proj["goal"] or "")) if views else ""
    pb = bayes.path_belief(views, path[-1]) if path else 0.0
    return render_layout(
        proj["name"], PROJECT_PAGE,
        project=proj, views=views, parent_code=parent_code,
        path=path, focus=focus, warnings=warnings,
        svg=svg, path_belief=pb,
    )


@app.route("/p/<name>/nodes", methods=["POST"])
def add_node_route(name):
    proj = load_project_or_404(name)
    node_name = request.form["name"].strip()
    parent = request.form.get("parent") or None
    try:
        prior = float(request.form.get("prior") or 0.5)
    except ValueError:
        prior = 0.5
    prior = max(0.0, min(1.0, prior))
    desc = request.form.get("desc", "").strip()
    if not node_name:
        abort(400, "name required")
    with db.connect() as conn:
        db.add_node(conn, proj["id"], node_name, desc, parent, prior)
    return redirect(url_for("project_view", name=name))


@app.route("/p/<name>/nodes/<code>/observe", methods=["POST"])
def observe_node(name, code):
    proj = load_project_or_404(name)
    result = request.form["result"]
    notes = request.form.get("notes", "").strip()
    with db.connect() as conn:
        node = db.get_node(conn, proj["id"], code)
        if not node:
            abort(404)
        alpha, beta = bayes.observe(node["alpha"], node["beta"], result)
        status = bayes.status_from_belief(alpha, beta, node["status"])
        db.record_observation(conn, node["id"], result, notes)
        db.update_node_belief(conn, node["id"], alpha, beta, status)
    referrer = request.referrer or url_for("project_view", name=name)
    return redirect(referrer)


@app.route("/p/<name>/nodes/<code>/rollback", methods=["POST"])
def rollback_node(name, code):
    proj = load_project_or_404(name)
    with db.connect() as conn:
        db.rollback_subtree(conn, proj["id"], code)
    return redirect(url_for("project_view", name=name))


@app.route("/p/<name>/nodes/<code>")
def node_view(name, code):
    proj = load_project_or_404(name)
    with db.connect() as conn:
        node = db.get_node(conn, proj["id"], code)
        if not node:
            abort(404)
        views = load_views(conn, proj["id"])
        observations = db.list_observations(conn, node["id"])
    v = bayes.to_view(node)
    pb = bayes.path_belief(views, v)
    return render_layout(
        f"{proj['name']} · {code}", NODE_PAGE,
        project=proj, node=v, observations=observations, path_belief=pb,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
