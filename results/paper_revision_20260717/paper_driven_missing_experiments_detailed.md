# 基于所给 OVSS / OVSS-TTA 论文的补实验审阅

日期：2026-07-17

## 结论先行

现有实验已经覆盖 clean 主表、三随机种子、shortcut-oriented 数据集、matched bridge ablation、source-risk 机制指标、完整模块消融、small-object regression、正式效率、第二 backbone、same-class multi-object、五种 counterfactual perturbation 和 paired bootstrap。继续增加普通 benchmark 或 corruption 的边际价值已经较低。

当前最值得补的三项证据是：

1. **Candidate-oracle + baseline anchors**：定位候选几何与更新策略的瓶颈，并在同一 240-image 子集上给出 NoAdapt/TENT/MLMP/CFC-TTA 参照。内部 2x2 和四数据集 baseline anchors 均已完成。
2. **Thing/stuff/background × baseline**：不再只报告 CFC-TTA 自身的角色统计，而是在同一冻结候选上比较 NoAdapt/TENT/MLMP/CFC-TTA。九数据集对照已完成；八个数据集使用 240 张图，Cityscapes 因每图产生 36 个 patch，使用 60 张图并单独披露。
3. **Vocabulary-composition stress**：这是目前最明显、最 OVSS-specific、且尚未被现有实验覆盖的缺口。它检验文本标签集合改变后，confidence-source reliability 是否仍然有效。

另有一项高价值诊断可作为主文或 appendix：**semantic-correct but localization-wrong subset**，用于证明“模型知道类别”并不等于“当前局部区域可作为可靠伪标签”。

## 逐篇论文对应关系

| 论文 | 论文中关键实验 | 对当前论文的直接启发 | 当前覆盖 | 需要补什么 |
|---|---|---|---|---|
| LaVG, ECCV 2024 | Table 3 比较 Normalized Cut、CutLER、Panoptic Cut 与 GT masks；Table 4 比较 hard masked attention、Gaussian blur、region prototype；Table 5 报告 cost | 分离 candidate discovery 与 region grounding；同时检查 intervention view 的输入域偏移和效率 | Candidate-oracle 2x2 已完成；正式效率已完成 | 给 2x2 增加同子集 baseline anchors；在 appendix 做 hard zero / soft blur / prototype 的 intervention sanity |
| Mask-adapted CLIP, CVPR 2023 | Figure 1 用 GT mask/GT class 两个 oracle 定位瓶颈；Table 2 比较 GT/generated masks 与 GT/caption vocabulary；Table 3 检查 mask-domain adaptation | 证明“更准确的 mask”不必然解决 region-language classification；hard mask 会造成 CLIP input-domain gap | Candidate-oracle 已覆盖第一部分；当前方法使用 soft intervention | 增加 hard-zero 对照，证明 soft blur 不是任意工程选择；不要把 GT candidate 称为性能 upper bound |
| CLIPSelf, ICLR 2024 | Table 2 分别报告 boxes、thing masks、stuff masks，并显示 proposal supervision 对 thing 有利、对 stuff 可能不利；Table 1 有 resolution sanity check | 角色与 spatial-unit coverage 必须分开；proposal 机制天然偏 thing | TRACE-only 九数据集角色诊断与 NoAdapt/TENT/MLMP/CFC-TTA matched role comparison 均已完成 | 如需进一步展开，只附加 role-specific candidate recall/size bin，不需要新模块 |
| DenseVLM, ICCV 2025 | Tables 1--3 分 Boxes/Masks-T/Masks-S；Table 4 改变 category set 的规模和 thing/stuff 组成；Table 2 同时报时间和显存 | foreground/background bias 与 vocabulary composition 都是 dense VLM 的核心变量 | 角色诊断和效率已覆盖一半 | 必做 vocabulary-composition stress；报告 added-label false-positive area 和 role-specific harm，而不只报 mIoU |
| CASS / DSG, CVPR 2025 | Table 1 覆盖八数据集；Table 2 报 pAcc；Table 3 组件消融；可视化分析 object parts 和 intra-object coherence | 多 object / object-part fragmentation 是 candidate-level 方法的重要边界 | same-class multi-object 和 fragment guard 已完成 | 把已有 same-class 结果放入 experiment chain；在 role table 旁报告 candidate fragmentation/coverage，而非引入 graph 模块 |
| PACL, CVPR 2023 | Table 1 与 ROC 先验证 patch-text alignment proxy，再报告下游 segmentation；核心观察是 image-level class knowledge 不保证 patch-level localization | 你的机制也需要一个 OVSS-specific local-alignment failure subset | 当前 source-risk AUC 控制了 confidence，但未直接隔离“类别对、位置错” | 补 semantic-correct/localization-wrong subset，比较 NoAdapt/MLMP/CFC-TTA 的 region IoU delta 和 correct-to-harm |
| CLIPtrase, ECCV 2024 | Table 2 在四数据集上逐模块、层、分辨率、聚类方式和 FLOPs；Figure 7 做参数面敏感性 | 实验应把 performance gain 与 local spatial mechanism、cost 联系起来 | bridge ablation、正式 efficiency、第二 backbone 已覆盖 | 不建议照搬聚类参数面；只需保留 candidate budget versus cost Pareto 作为 appendix |
| NERVE | Tables 2--4 分析 pseudo-mask、entropy fusion 与 affinity；报告有/无 background 数据集 | entropy 不是唯一局部信号，且背景设置会改变结论 | confidence-controlled AUC 与角色表已覆盖 | 可补 adaptation-step trajectory：step 0/1/3/5/10 的 mIoU 与 harm，检验错误是否被逐步放大；优先级低于 vocabulary stress |
| MLMP | 主表比较 clean/shift；Tables 17--20 分 layer range、prompt integration、template count 和组件；附录有不同 backbone、single-sample、online/episodic 和 cost | 主要 baseline 必须保持 prompt/backbone/protocol 可解释；不能只引用论文数字做机制比较 | clean 主表、三种子、matched bridge、效率、第二 backbone 已覆盖 | 当前 matched role run 使用官方 baseline optimizer 与同一候选；报告时明确 CFC-TTA 为 dataset-specific canonical config，不宣称 optimizer 完全统一 |
| Seg-TTO | Tables 3--8 分 framework、prompt、TTO loss、visual/textual contribution；22-domain MESS 检查专门领域 | naive prompt/TPT 可能损伤 dense localization；prompt 是公平性变量 | 主方法与 MLMP 使用同一 prompt bank；多个 baseline 已跑 | appendix 做 1-template / shared-7-template 两种 prompt bank 下的 MLMP-CFC 差值即可；不必新增 22-domain MESS |
| FreeCP | Figure 1 比较 full vocabulary 与 GT vocabulary；Table 3 分 redundancy/ambiguity purification；Table 6 多 backbone | overcomplete vocabulary 会改变 activation competition，是 OVSS 特有而非普通 TTA 问题 | 当前 real-class extension 仅做局部分组，未系统改变 vocabulary | 必做 related-vs-random distractor stress，并分开 class-count effect 与 semantic competition |
| VocAlign | Table 1 报 best/worst classes；Figure 5 改变 Top-K class count 和 memory，并比较随机 class；还用 crop 验证 ego-vehicle context failure | 需要 per-class failure 与 vocabulary budget/control，不应只报均值 | 有全局 mIoU 和机制统计，没有 vocabulary budget | vocabulary stress 中加入随机等量 distractor；报告最受影响的 5 类与最受益的 5 类 |
| SemLA | 20-domain leave-one-out；Figure 3 contribution heatmap；Figure 5 检验 CLIP distance 与实际 adaptation gain 的相关性 | reliability/proxy 应与实际收益相关，而不能只展示分数分布 | source-risk AUC、reliability quartile、paired bootstrap 已覆盖 | 不需要再跑 20 domains；可在角色表中报告 reliability bin 与 region-IoU delta，已有数据足够重算 |
| Self-supervised Augmentation Consistency | Table 4 消融 augmentation consistency；另有 per-class 与 transformation 分析 | counterfactual view 应验证 perturbation choice，而不是只选一种 | 五种 counterfactual perturbation 已完成 | 增加 hard-zero intervention 作为 domain-gap negative control；其余 perturbation 不必重跑 |

## 应该如何把 Experiments 串起来

### E1. End-to-end effectiveness

先回答“方法是否有效”：七个标准 clean 数据集、COCO-CD、Waterbirds-Seg，三随机种子与 paired bootstrap。这里不讲机制细节。

### E2. Prediction utility versus update reliability

用已有 context-dominant matched-region、high-CFC subset、confidence-controlled AUC 和 reliability quartiles 回答：context 可以预测正确，但不应自动获得完整 update authority。

### E3. Candidate bottleneck

用 Candidate-oracle 2x2 回答：错误来自 candidate geometry 还是 reliability weighting。四个内部 cell 保持 CLS 关闭；NoAdapt/TENT/MLMP/CFC-TTA 作为同子集 complete-method anchors，单独列在表左侧，不能把它们伪装成 2x2 cell。

### E4. Semantic-role bottleneck

在冻结的 pre-adaptation CFC-TTA candidates 上比较 NoAdapt/TENT/MLMP/CFC-TTA，并按 thing/stuff/background 报告：

- region accuracy before/after 和 delta；
- correct-to-harm rate；
- prediction flip rate / flip incidence；
- subset mIoU。

这一表应独立于原 Table 4(c)，不合并。它回答的是“不同适应方法如何改变同一批语义角色候选”，而原 Table 4(c) 回答的是“source-risk 是否在控制 confidence 后预测 harm”。

### E5. OVSS-specific vocabulary bottleneck

这是新实验的最高优先级。建议 VOC20、PC60、COCO Object、COCO-Stuff 四个数据集，方法为 NoAdapt、MLMP、CFC-TTA，四种 vocabulary：

1. benchmark labels；
2. benchmark labels + synonyms/class extensions，并映射回 real class；
3. benchmark labels + 等量 semantically related/co-occurring distractors；
4. benchmark labels + 等量 random unrelated distractors。

固定图像、backbone、prompt template 数量、optimizer 和 adaptation step。报告 base-class mIoU retention、added-label false-positive area、thing/stuff/background correct-to-harm、target-context margin shift、最受影响/最受益类别。条件 3 与 4 的差异才是 semantic competition，不能把单纯增加类别数误写成 context effect。

### E6. Semantic-correct but localization-wrong

以 adaptation 前结果构造 diagnostic subset：图像级 top vocabulary 包含 GT 类，但 candidate 对相应 GT component 的 IoU 低于 0.3，且 region confidence 高于匹配数据集的中位数。比较 NoAdapt、MLMP、CFC-TTA 的 candidate IoU delta、correct-to-harm 和 wrong-location confident area。GT 只用于离线分组，不参与适应。

该实验把 PACL 的 patch-alignment问题与本文的 update reliability 直接连接：模型可以“知道类别”，但错误位置上的低熵局部预测仍不是可靠伪标签。

### E7. Practical boundary

最后报告已有 runtime/memory、candidate-budget Pareto、second backbone、same-class multi-object、counterfactual robustness、small-object regression 和 failure cases。它们是边界与部署证据，不应抢在核心机制之前。

## 不建议继续增加的实验

- 不再重启完整 15-corruption suite；它不能补上 vocabulary/source 的机制缺口。
- 不需要照搬 Seg-TTO 的 22-domain MESS；当前论文不主张 specialized-domain adaptation。
- 不需要新增 graph、SAM 或 proposal network 作为主方法组件；Candidate-oracle 的作用是定位瓶颈，不是引出更复杂的方法。
- 不需要把每篇论文的全部 ablation 复制一遍。只借用与 confidence source、candidate unit、semantic role、vocabulary competition、local alignment 直接相关的实验逻辑。

## 当前证据状态

| 实验 | Evidence type | 状态 | 主要产物 |
|---|---|---|---|
| Candidate-oracle 2x2 | newly_run | 完成 | `candidate_oracle_4datasets_240/decomposition.csv` |
| Candidate geometry quality | newly_run | 完成 | `candidate_oracle_quality_4datasets_240/candidate_quality_summary.csv` |
| Candidate-oracle baseline anchors | newly_run | 四数据集完成 | `candidate_oracle_baseline_comparison/candidate_oracle_with_baseline_anchors.csv` |
| TRACE-only 9-dataset role profile | newly_run | 完成 | `table4c_9dataset_role_expansion/dataset_role_breakdown.csv` |
| Matched baseline role comparison | newly_run | 九数据集完成 | `matched_baseline_role_9datasets_240_city60/*/baseline_role_diagnosis.csv` |
| Vocabulary-composition stress | planned | 未运行 | 尚无 |
| Semantic-correct/localization-wrong | planned | 未运行 | 尚无 |
