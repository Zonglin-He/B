# TRACE 新稿首轮独立实质评审

## 结论与评分

**整体 5/10，Weak Reject；confidence 4/5。** 采用用户指定标尺：6 为 Weak Accept，7 为 Accept。分数评价当前已提供材料，不预测进行中的实验结果，也不是五维分数的机械平均。

方法的数学定义与所核实现基本一致，四个完整评估 split 的历史结果可以核实。当前拒绝倾向的主要原因是：用来区分所提方法与普通区域熵适应的核心实验全部未完成，common support 与有益更新之间的联系也没有完成验证。明确承认证据边界减少了过度宣称，但不等于相关贡献已经被验证。没有发现需要直接宣布方法公式错误的证据。

| 维度 | 分数 / 10 | 判断 |
|---|---:|---|
| 贡献 | 5 | 候选级双探针和区域归一化形成清楚的算法设计；min 的顺序性质并不能证明该设计提供新增实证知识。未做外部文献优先权检索，不声称其已被前人完整覆盖。 |
| 正确性 | 7 | 所核候选、变换、支持读出、detach 与损失定义一致，小张量 CPU 验证通过；未认证准许源码之外的完整继承调用链。 |
| 实验支撑 | 4 | 有四个完整 split 的历史单次结果；所有关键机制控制、风险分析和新确定性全量验证仍未完成。 |
| 比较与解释边界 | 8 | 已清楚区分文献报告值与本地值、不同预算/后端、非校准与非因果解释；少量机制段落仍有已完成式措辞。 |
| 清晰与可复现 | 6 | 方法主链可以逐段读懂；预测分支仍依赖 inherited/established 描述，精确 checkpoint、层权重和扩展映射算子不够自足；运行确定性应进入正式协议。 |

Confidence 4/5 表示对当前稿件、计数核验和明确的证据缺口有较高把握，不是对完整推理链、全部标签隔离、文献优先权或未来实验结果的认证。

## 审查范围

审查开始于 2026-10-28 13:34:05（Asia/Shanghai），限定十分钟内。对象是本 worktree 的新稿，不是桌面原稿。已逐段阅读 main.tex 的摘要、引言、相关工作、方法、实验和局限，以及 appendix.tex、preamble.tex、全部 generated/*.tex、references.bib、generated/provenance.json；阅读两个 PDF 的文本，并直接检查主稿第 3 页和 supplement 第 2 页渲染。没有因必要比较而使用旧稿。

源码只读 trace_ovss/method/two_view.py 与 trace/{engine,candidates,objective,interventions}.py。未读 docs/self-evolution-contract、其他代理评审、其他分数、adapter/dense_clip/estimator 或历史 research wrapper 的实现。按环境要求启动时检索过工作区历史证据索引，但没有把旧方法历史审计作为新稿的机制证据或评分依据。

补充读取限于 generated/provenance 引用的 completed evidence：result、protocol、suite seal、fixed config 等，以及用户直接指定的最新 repeatability summary。未读取预测/模型 payload 或 GT 内容，未运行 GPU 或模型 forward。只执行计数重算、哈希核对、内存 PDF 渲染和小张量 CPU 测试。本报告不重新认证全部历史标签隔离流程。

## 逐段主链与方法可读性

| 位置 | 实际读到的定义与判断 |
|---|---|
| main.tex:14-20 | 动机链是“预测置信度不等于更新效用 → 上下文改变下的区域响应 → 相对熵权重”。逻辑问题不是读不懂，而是第二步到第三步的效用联系尚无完成证据。 |
| main.tex:50-63 | 清楚区分 query vocabulary 与最终 category，以及初始无梯度诊断与可微 prompt-specific 适应分支。这是必要定义，不应把两个分布混作同一张量。精确层聚合仍有复现缺口，见 W4。 |
| main.tex:65-75 | crop 内四连通、按 source image 跨 crop top-K、query extension 不先合并、候选固定，均能按文字追踪；附录给了 tie-breaking。没有隐含实例分割或 GT mask 假设。 |
| main.tex:77-99 | 两视图都保留区域，灰度模糊与低对比度通道循环变换的定义具体；在原 token 位置读 query probability，而不重新抽取候选。附录明确了平均模糊 padding、RGB 数值及候选像素可能进入上下文变换统计量。 |
| main.tex:101-119 | min、活跃阈值、区域内及 prompt 平均、episode 内权重归一化均明确。c_R 用于诊断而不是交叉熵伪标签。所核 objective 的小张量结果与公式一致。min 的最大共同下界解释是正确的代数刻画，不构成性能定理。 |
| main.tex:121；appendix.tex:75-96 | 只更新视觉 LayerNorm、episode reset、无 CLS loss 的边界清楚；区域平均不等于梯度等量，全部低分不自动减小步长，singleton 的 rho 会抵消。稿件已经说明，不能再当作未披露 bug。 |
| main.tex:123-161 | 完整 split 的分数与 operating-point 比较可检查；机制控制与风险分析是计划而不是结果。速度段只有输入计数恒等式和历史计时，不应要求作者证明其没有声称的三分之一总延迟下降。 |
| main.tex:163-171 | 局限基本准确；但“controlled studies isolate”不能指向全 pending 的表，需改语态。结论描述算法结构成立，不能外推为已证明可靠或因果定位。 |

## 已完成与未完成证据

### 已完成：历史完整 split 测量

从 20 个本地 completed result.json 的类别 intersection/union 重算 pooled mIoU；四个新方法结果及 16 个 native baseline 结果与表格一致。所核 result、两份 suite seal、fixed_configs 的哈希与 generated/provenance.json 一致。这是算术和工件身份核查，不是新推理或预测 payload 复验。

| 数据集 | 图数 | TRACE 历史 mIoU | 最佳本地 native | 差值 pp |
|---|---:|---:|---:|---:|
| PC60 | 5,105 | 30.584325 | MLMP 29.600177 | +0.984148 |
| VOC21 | 1,449 | 52.396489 | MLMP 50.292803 | +2.103687 |
| COCO-CD | 500 | 75.189561 | TENT 72.200299 | +2.989262 |
| COCO Object | 5,000 | 32.678270 | MLMP 30.355801 | +2.322468 |

主表另与 MLMP 论文报告值比较，VOC21/PC60/COCO Object 差值为 +1.62/+2.59/+3.84 pp。主表及 native 表的差值不能相互替换。generated/reported_comparison.tex:2 明示来源、方法不同设置和未主张的 checkpoint/query-extension 等价性。这种报告值比较协议可接受，**不因未重跑全部文献 baseline 而扣分**。已完成结果支持“这些 operating points 上取得更高数值”，不支持“同协议、同预算或公式级因果优越”。

历史 protocol 的 algorithm_definition 明确为只评估 target/counterfactual、使用 min(T,C)、无 dominance attenuation；variant=full 的旧元数据不能据此误判为实际运行了三视图。所核 engine/candidates/objective/interventions 的当前哈希与历史 protocol 中对应源哈希一致；这不等于已核历史 wrapper 的所有代码。

### 已完成：局部确定性诊断

用户提供的 /home/wwww/hzl/results/trace_evolution_repeatability_20261028_1332/summary.json 显示：

- 默认 dispatch 的两组 A/A support 相同，但最大 loss 差分别为 0.0189732015 和 0.0019092560，梯度和参数不一致，标签改变数分别为 11,024 和 4,583。这些是原始标签改变数，不是错误率或 mIoU 变化。
- deterministic_default 下所列三组比较，loss 差均为 0，support、gradient、trainable 参数均相等，changed_labels 均为 0；deterministic_math 所列三组也相同。
- gt_contents_opened=false。四图与“旧旧/新新/旧新”的运行身份说明来自用户；summary 独立支持的是所列比较字段。

因此，不能把此前默认 A/A 不一致归因于新旧公式不等价，也不能继续声称确定性默认 dispatch 的局部门槛仍失败。相反，不能把四图通过升级为全 split 性能、跨配置稳定性或已完成机制控制。

### 已完成：有限 CPU 事实核查

AST 提取准许的 objective 类，在 CPU、无模型加载的小张量上验证：prompt/region 平均与 Eq. 9-11 一致（合成 loss=8.9）；全体权重共同正缩放不改变损失；整个 episode 只有一个活跃候选时 rho 抵消；空候选损失为零。此测试不触及继承读出或任何模型 forward。

### 尚未完成

新确定性实验正在 freeze 的状态仅按用户提供说明记录，没有将其计为 completed。region-uniform、candidate-confidence、单视图、mean/product、uniform-token、matched shuffle 和 update-risk 分析，在审查稿中均无已完成结果。它们将改善还是削弱论文，现有材料不足以判断。

## 主要 Weaknesses

### W1 / High：核心设计没有完成的匹配控制

**位置：** main.tex:101-121、142-150；generated/mechanism_status.tex:2-13；appendix.tex:101-106。

**事实与推理：** 完整方法同时改变连通候选、top-K、区域平均、双探针、min 权重和 operating-point 设置。所有用于区分这些因素的控制都 pending。完整方法优于 MLMP 或 TENT 不能判断探针是否提供超出初始置信度的信息、第二视图是否必要、min 是否优于 mean/product、区域平均是否足以解释收益。现有证据兼容“region-uniform 已能达到同样水平”，不是已经证明这种情况发生。附录最大共同下界的证明不能排除这些替代解释。

**对结论的影响：** 这是当前新颖性和有效贡献的主要缺口，而非笼统的调参公平性问题。方法写得完整，不代表其决定性设计已经验证。

**最小修复：** 确定性 reference gate 通过后，至少在预先声明的两个不同完整 split 上，固定候选、episode、读出、可训练参数、LR 和步数，完成 region-uniform、candidate-confidence、T-only、C-only、mean、product 与 min 比较；另在相同选中 token 集合上测试 uniform-token 目标。权重控制保留相同诊断 forward。使用 paired episode 区间，不挑选已经看过结果后最有利的子集，保留负结果。

**需要：实验 + 文字；不预设需要改算法。** 若 min 或第二视图不占优，应按实际结果缩小贡献，不能保留对应的必要性叙事。

### W2 / High：common support 到有益更新的联系未被验证

**位置：** main.tex:16-20、104-121、150；appendix.tex:75-99、108-113；two_view.py:74-85。

**事实与推理：** rho=min(T,C) 测量修改后 query 的共同绝对支持，不直接测正确性、上下文依赖的因果来源或更新效用。诊断和优化还使用不同的 prompt/层聚合分支。共同错误、被变换保留的上下文、大 mask，都可能给出高支持。极端全 crop 候选令两视图退化为原输入；真实数据中是否有足够多此类候选，本轮没有证据，不能当作观察到的失败。稿件诚实列出这些局限，但没有任何已完成分析说明该 score 确实区分了动机中的有害自强化风险。

**对结论的影响：** W1 问“组件是否提高最终分数”；本项问“分配信号是否与论文提出的更新风险问题有关”。两者可以独立出现正、负或空结果。当前只能称为 common-support 分配启发式，不能把 reliability 当已验证性质。

**最小修复：** 预先定义离线有害更新结局，在相近初始置信度和面积层内比较 rho 与风险；正确类别概率变化与 C-to-W/W-to-C 分开。用完整 episode 内匹配的 score shuffle 保持候选几何和 score multiset，核实候选特定分配是否改变、是否改善结果。先 seal 预测再读标签，按完整 episode 估计不确定性。不能用单候选独立归一化更新检验 rho，因为其会抵消。若只观察到分配改变而区间跨零，结论只能是描述性改变。

**需要：实验 + 文字；不是已确认算法失效。** 没有这些结果时，标题、动机和结论应继续限定为行为支持启发式，而非已经证明可靠更新。

### W3 / Medium-High：局部确定性修复尚未回答历史性能的重复性

**位置：** main.tex:125、152-168；appendix.tex:39-54；generated/provenance.json:33-39；最新 repeatability summary。

**事实与推理：** 默认 A/A 已显示即使 support 不变，优化和预测也会变化。deterministic_default 的旧旧、新新、旧新局部 exact 是应计入的已完成修复，不应继续当作失败项。它没有给出历史单次全 split mIoU 的运行变异，也没有给新冻结运行的完整分数。历史分数仍是完成的测量，不能据此宣告无效、推断收益方向或凭标签改变数虚构 mIoU 波动。

**对结论的影响：** 这是重复性和未来单因子控制的前提，不是要求所有方法天生 bit-exact。尤其 PC60 最佳本地差值是 0.984 pp，而运行变异对该差值究竟有多大影响仍未知；没有证据说它一定能解释该差值。

**最小修复：** 冻结 deterministic 开关、dispatch、精度、软件硬件和 episode 顺序；明确分开“四图 gate”“历史单次全量”“新确定性全量”。新控制使用同一冻结 runtime，并补方法重复运行及 episode/seed 稳定性、paired 区间。只复测实际参与新 matched 比较的相应臂；不要求因引用文献数值而重跑所有 baseline。不要把新的 pending 结果提前并入表格，也不要直接把旧 provenance 的笼统验证字段改成全量通过。

**需要：实验 + 运行配置 + 文字；无需修改算法公式。**

### W4 / Medium：预测接口缺少自足的复现定义

**位置：** main.tex:53-63、127；appendix.tex:15、24-25；engine.py:69-80。

**事实与推理：** 文稿将影响 rho 与梯度的关键操作称作 inherited adaptive weighted layer combination、aggregated prompt embeddings、established extension map。给出“18 层、7 模板”不足以决定层权重公式、embedding/score 聚合和 logit 尺度，也没有把精确 checkpoint、扩展汇聚算子及对应词表工件的入口写得自足。provenance 的源/词表哈希有审计价值，但不是这些操作的数学或可执行说明。本轮未越界读 adapter/dense_clip/estimator，因此这是文字与工件入口缺口，不是关于未见代码错误的指控。

**对结论的影响：** 读者可以理解新损失，却仍难独立重建产生候选、权重和最终类别的准确预测分布；这些细节也正是解释 operating-point 差异所需的身份信息。

**最小修复：** 附录补诊断与优化的层聚合公式、归一化和温度；给 checkpoint 标识/哈希、词表及 extension-map 工件与具体汇聚顺序算子、历史 quality_only 和当前入口的精确版本及命令；运行开关与 W3 对齐。

**需要：文字 + 工件清单；不要求改算法或再做大型实验。**

## 次要问题

**M1，文字状态不一致：** main.tex:144-150、168 用 examine、are held fixed、controlled studies isolate 等已执行式语气描述全 pending 的机制包。统一改为计划语态或将计划移至附录；不能让计划承担已完成结果的功能。只需文字修复。

**M2，成本列名：** generated/cost_snapshot.tex:2-12 的 Views 实际为 diagnostic inputs/image，不是公式中的总输入数。改成 Diag. inputs/img 更明确。当前没有 speedup 宣称，不能以缺少 speedup 对照本身否定现有计时。

没有把缺少所有 backbone/corruption/vocabulary 扩展当作独立否决条件；它们限制外推范围，不能替代当前核心验证。

## 必须修复与决策边界

1. **W1：完成决定贡献的匹配控制。** 至少证明探针信息、第二视图/min、区域平均各自的作用，或按负结果收缩方法叙事。
2. **W2：建立或否定更新风险联系。** 提供预先定义的离线结局与匹配分配控制，不把 common support 的代数性质当可靠性证据。
3. **W3：将已修复 gate 变成明确的确定性运行协议。** 历史单次与新运行分开，完成的局部诊断保留其应有范围，pending 不入账。
4. **W4 + M1：补齐接口身份与文字状态。** 给精确聚合/映射/工件入口，所有未完成研究用未完成语态。

当前 5 分不是对实验未来结果的预测，也不是因采用 MLMP 报告值扣分。若匹配控制只产生零效应或反向结果，分数不会仅因“实验完成”自动上升；若得到稳定、可归因且与更新风险相符的证据，才具备重新评估到 6 分及以上的依据。**现有材料不足以支持当前接收。**

## 快照

本报告绑定以下新稿快照，完整 SHA-256 见 round01.json：main.tex e491c1383f1fdd30；appendix.tex 6ab2f67daccd876c；main.pdf 73a7a3f01463736f；supplement.pdf d4ca2274f002d373。所有评分依据是本轮实际核查材料，不使用其他评审或作者目标分数。

