# bayesian-networks — 方案设计与复盘工具

用贝叶斯信念网络的思想组织你的方案尝试。每个节点是一个假设/决策（jim0,
jim1, …），子节点依赖父节点成立。每次 `observe` 用 Beta/Bernoulli 共轭更新
节点后验信念；发散太多或分支信念过低时工具会提醒你"回归"，指向更有希望的
分支。

**场景**：大模型训练方案分析，AI 辅助复盘，找最优路径，避免无头苍蝇。

## 依赖

- Python 3.10+（只用标准库）
- `graphviz` 二进制（`brew install graphviz` / `apt install graphviz`）
  用于把 `.dot` 渲染成 `.png` / `.svg`；没装也能生成 `.dot`。

## 核心概念

| 概念          | 说明                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| 节点 jimN     | 一个方案/假设。自动编号。                                                                            |
| prior         | 手动给的先验 [0,1]，编码成 `Beta(2·prior, 2·(1-prior))`。                                            |
| belief        | 后验 `α/(α+β)`。success +1α，fail +1β。                                                              |
| status        | pending（没证据）/ active（有证据但未定论）/ success（≥0.75）/ fail（≤0.3）/ obsolete（手动 rollback）|
| 发散警告      | 同一父节点下 ≥3 个 pending 分支未验证 → 建议聚焦。                                                   |
| 回归提醒      | 节点 belief ≤ 0.3 → 建议回到父节点，试别的分支。                                                     |
| 最优路径      | 每层贪心选 belief 最高的子节点，从 root 一直到叶子。图里加粗标出。                                   |

## 快速上手（LLM 训练复盘示例）

```bash
# 1. 开项目
python planner.py init llm-train --goal "8xA100 训领域 LLM, loss<2.0"

# 2. 铺假设（先验 = 你对它的信心）
python planner.py add "选择 base: Llama3-8B"   --prior 0.7 --desc "社区资源多"
python planner.py add "LoRA 微调, r=16"       --parent jim0 --prior 0.65
python planner.py add "lr=2e-4 cosine"        --parent jim1 --prior 0.6
python planner.py add "lr=1e-4 linear"        --parent jim1 --prior 0.5
python planner.py add "lr=5e-4 cosine"        --parent jim1 --prior 0.4

# 3. 工具提醒发散 — 3 个 pending 兄弟节点
python planner.py check
# ⚠ 发散警告: jim1 下有 3 个未验证分支 — 建议聚焦一条先验证。

# 4. 跑实验，记录结果
python planner.py observe jim0 success --notes "模型加载正常"
python planner.py observe jim1 success --notes "显存 35GB"
python planner.py observe jim4 fail    --notes "loss 震荡"
python planner.py observe jim4 fail    --notes "梯度爆炸"
# ⚠ 回归提醒: jim4 belief=0.20, 回到 jim1 试别的。

python planner.py observe jim2 success --notes "loss 3.5 → 2.2"

# 5. 看当前最优路径
python planner.py path
# jim0 → jim1 → jim2  belief 依次 0.80 / 0.77 / 0.73

# 6. 下一步做什么？
python planner.py focus
# -> 继续验证 jim2

# 7. 关掉已证失败的分支（保留历史）
python planner.py rollback jim4

# 8. 出图（用于复盘报告）
python planner.py graph --out runs/r1 --format png
```

## CLI 速查

```
init <name> --goal "..."              新建项目
use <name>                            切换项目
projects                              列出项目（* 是活动项目）
add <name> --parent jimN --prior P    加节点
observe <code> success|fail --notes   记观察, 贝叶斯更新
list                                  表格列出节点 + belief
show <code>                           节点详情 + 观察历史
path                                  当前最优路径
focus                                 建议下一步
check                                 发散/回归检查
rollback <code>                       标节点及后代为 obsolete
graph --out <base> --format png|svg   生成 graphviz 图
```

## 数据存储

- SQLite 文件：`bayes_planner.db`（和 `planner.py` 同目录）
- 表：`projects`, `nodes`, `observations`, `state`
- 纯文本 `.dot` + 渲染 `.png` 写到你指定的 `--out` 路径

## 贝叶斯更新公式

节点的成功率建模为 Beta 分布：

```
先验:  Beta(α₀, β₀) = Beta(2·prior, 2·(1-prior))
观察:  Bernoulli(θ), θ = P(该方案成功)
后验:  α = α₀ + #success,   β = β₀ + #fail
belief E[θ] = α / (α + β)
```

路径联合信念（独立假设）：`∏ belief(节点)` 沿 root→节点 的链。
`path belief` 告诉你"这整条链都成立"的概率 — 链越长越保守，提示你越往后越
需要证据支撑。
