---
name: bayes-plan
description: Guide the user through scheme design and retrospective using the project's Bayesian planner. Trigger when the user wants to plan, debug, or retrospect on an approach (LLM training runs, hyperparameter search, experiment design, bug-fix strategies) and frame it as a belief network — or when they explicitly ask to run `/bayes-plan`, mention jim0/jimN nodes, "belief network", "divergence", "regression/回归", or reference planner.py / web.py in this repo.
user-invocable: true
allowed-tools:
  - Read
  - Bash(python3 planner.py *)
  - Bash(python3 web.py *)
  - Bash(ls *)
  - Bash(open *)
  - Bash(sqlite3 bayes_planner.db *)
---

# /bayes-plan — Bayesian scheme design & retrospective

This project is a Bayesian-belief-network planner (`planner.py` CLI + `web.py`
Flask UI, SQLite at `bayes_planner.db`). The goal: frame an open-ended design
question (e.g. "how do I train this LLM efficiently?") as a tree of hypotheses
where each node `jimN` is one decision; record success/fail observations;
Bayesian updates reveal the best path and flag divergent exploration.

Your role when this skill is active: be the user's analyst. Keep them on a
disciplined loop — **frame hypotheses → run one → observe → update → check for
divergence → focus next step**. Don't let them accumulate dozens of untested
branches (that's the "无头苍蝇" failure mode the tool exists to prevent).

Arguments passed: `$ARGUMENTS`

---

## First thing to do: read state

Before suggesting anything, check what exists. Run **in parallel**:

```bash
python3 planner.py projects    # any active project?
python3 planner.py list        # nodes + beliefs in active project
python3 planner.py check       # divergence / regression warnings
python3 planner.py focus       # the single best next step
```

If there is no active project, move to **Phase 1**. Otherwise jump to the phase
that matches what the user is asking.

---

## Phase 1 — Framing a new scenario

When the user describes an open problem ("I want to train a domain LLM",
"这段代码调不对想复盘一下"), help them frame it as a network before they start
adding nodes. Ask (concise, bulleted):

1. **Goal** — a measurable success criterion (`loss<2.0`, `acc>0.9`,
   `bug fixed and tests green`). This becomes the project goal.
2. **Root-level decisions** — the 2–4 choices that dominate everything
   downstream (base model, algorithm family, dataset, architecture).
3. **Prior strength** — for each candidate, how confident are they on a 0–1
   scale, and why (literature, past experience, gut feel)? Push back on 0.9+
   priors without concrete evidence; those bias the tree.

Then create:

```bash
python3 planner.py init <name> --goal "<measurable criterion>"
python3 planner.py add "<decision>" --prior 0.7 --desc "<why this prior>"
python3 planner.py add "<child decision>" --parent jim0 --prior 0.6
```

**Hold the line on adding children**: if a parent has ≥3 untested siblings,
stop and tell the user to run the most promising one first. The tool's
divergence warning exists because adding branches is cheap and testing them
isn't.

---

## Phase 2 — Recording a run

When the user reports the outcome of trying a node ("jim2 worked, loss went to
2.1"; "jim4 diverged again"):

1. Record it with a specific, evidence-rich note — not just "ok" / "bad":
   ```bash
   python3 planner.py observe jim2 success --notes "loss 3.5→2.2 over 3 epochs, no overfitting"
   python3 planner.py observe jim4 fail    --notes "loss oscillated, grad norm 12→nan"
   ```
2. Immediately re-read state — `list`, `check`, `focus` — and report back:
   - New belief for that node
   - Whether a regression warning just fired (belief ≤ 0.3)
   - The suggested next focus
3. If a node hit fail status, **ask the user whether to `rollback`** its
   subtree. Don't auto-rollback: siblings might still be worth trying, and
   rollback is semantic ("abandon this line"), not statistical.

---

## Phase 3 — Checking convergence & divergence

Call `check` at every non-trivial turn. React to its output:

- **发散警告** (too many pending siblings): tell the user *which* branch to
  validate first — pick the one with the highest `prior` × `path_belief`.
  Don't let them add yet another sibling before running something.
- **回归提醒** (node belief ≤ 0.3): surface the parent code and say
  "jimK looks like a dead end; go back to jimParent and try the other child."
  If no viable sibling exists, that's the cue to add one now.

Also call `path` periodically to show the current best end-to-end chain, and
read `path_belief`: if the whole chain is below ~0.3, the plan is still
speculative — flag it explicitly, don't pretend things are on track.

---

## Phase 4 — Visualizing

Generate a graph when:
- The user asks for one (`show the graph`, `给我图`)
- You're about to summarize a long retrospective
- The tree has grown past ~8 nodes (visual gets faster than the table)

```bash
python3 planner.py graph --out runs/<date> --format png
open runs/<date>.png    # on macOS
```

Or launch the web UI for live interaction:

```bash
python3 web.py   # serves http://127.0.0.1:5000
```

Mention that the web UI has inline ✓/✗ buttons, an add-node form, and the
same focus/warnings panel — useful when the user wants to iterate fast.

---

## Commands reference (call via Bash)

| Command                                                            | Purpose                                  |
|--------------------------------------------------------------------|------------------------------------------|
| `python3 planner.py init <name> --goal "..."`                      | new project (becomes active)             |
| `python3 planner.py use <name>`                                    | switch active project                    |
| `python3 planner.py projects`                                      | list projects (`*` = active)             |
| `python3 planner.py add <name> --parent jimN --prior P --desc ...` | add hypothesis node                      |
| `python3 planner.py observe <code> success\|fail --notes "..."`    | record outcome, Bayesian update          |
| `python3 planner.py list`                                          | node table with belief / obs / status    |
| `python3 planner.py show <code>`                                   | node detail + observation history        |
| `python3 planner.py path`                                          | current greedy-optimal path              |
| `python3 planner.py focus`                                         | single node to work on next              |
| `python3 planner.py check`                                         | divergence / regression warnings         |
| `python3 planner.py rollback <code>`                               | mark node + descendants obsolete         |
| `python3 planner.py graph --out <base> --format png\|svg`          | render graphviz                          |
| `python3 web.py`                                                   | start web UI at :5000                    |

---

## Interpretation cheatsheet

- **belief** — `α/(α+β)` from Beta posterior. First observation swings it a
  lot (thin priors); later observations move it less. That's intended.
- **path_belief** — product of beliefs from root to node, assuming
  conditional independence. Useful as a plan-level confidence score, not a
  calibrated probability. Long chains naturally depress it.
- **status** transitions are derived from belief after each observe:
  `pending` → `active` (first evidence) → `success` (≥0.75) or `fail` (≤0.3).
  `obsolete` only comes from manual `rollback`.
- **prior** is baked into `Beta(2·prior, 2·(1-prior))` (weight 2 = lightly
  informative). If the user insists on a very strong prior, they should add
  `--desc` citing the evidence — otherwise nudge them toward 0.5 and let
  observations do the work.

---

## Anti-patterns to push back on

- **Adding nodes without running any**: the whole tool fights this. If the
  user keeps saying "and also we could try X, Y, Z" — stop them, run one.
- **Vague observation notes**: "it worked" / "didn't work" rots the
  retrospective value. Ask for the number, the symptom, the run ID.
- **Re-creating a dead branch under a new name**: if they abandoned jim4 as
  `lr=5e-4` and now want to add `lr=5e-4 with warmup` as jim9, fine — but
  make it a *child of the right parent* so the structural diff is visible.
- **Declaring success on 1 observation**: success status fires at belief≥0.75,
  which needs ~2 successes on a 0.5 prior. Don't override with `--prior 0.9`
  just to hit success faster; that defeats the Bayesian update.

---

## Ending a turn

When wrapping up a planning session, summarize in this order:
1. **What we just learned** (latest observation + its belief delta)
2. **Current best path** (from `path`)
3. **Next action** (from `focus`)
4. **Open warnings** (from `check`)

Keep it to 4 lines unless the user asks for detail.
