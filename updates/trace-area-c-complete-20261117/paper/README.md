# TRACE area_C: Complete Science and Independent Paper

本次固定范围内的科学实验已完成。新结果于2026年11月17日07:24:33 CST
完成实际成功退出核验，不是仅生成计划、目录或合成检查。

## 文件

- `main.pdf`: 独立论文及完整附录；不是覆盖旧稿，也不是声称投稿就绪。
- `main.tex`, `preamble.tex`, `references.bib`: 可重新编译的论文源文件。
- `complete_results.md`: 26个主条件、126个诊断消融单元、全部精确trial、
  全量组件/时间/上下文/TENT/分组结果、历史R03及全部22字段参数。
- `complete_results.json`: 完整机器可读结果、共享bootstrap draws和来源。
- `EXPERIMENT_INDEX.json`: 科学父阶段hash、真实终态与进程绑定。
- `self_review.md`: 新一轮五维审稿，保留无法通过写作解决的科学风险。
- `qa/`: 每页渲染、视觉检查、字体、引用和边界检查。
- `DELIVERY_SEAL.json`: 交付文件逐项SHA256和大小，不是额外模型运行授权。

## 结论边界

area_C 在8个L14 clean均值和15种S5 corruption均值上高于固定论文配置MLMP，
S5算术macro提高2.108889个百分点；B16 Object和PC60出现退化。

完整Object和PC60中，area_C减去no-region的配对95%区间均含0。
两组正式时间预算对比的精度区间也均含0。不能将整体均值优势归因于C，
不能声称等预算优越或校准后的更新可靠性。保留有益上下文、历史R03 NULL、
无法识别的原始跨episode区间、稀少分组undefined及所有失败记录。

TRACE使用训练来源调参而MLMP不调参；PC59/60存在历史开发曝光。
Waterbirds-Seg与COCO-CD为本地声明扩展，不是标准分类或未曝光独立公开测试。
Object的NoAdapt/MLMP数值与论文仍有差距，不能宣称完全复现论文数字。

## 重新编译

在此目录运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`。
结果格式化与QA仅读取已完成的派生结果，不开启RGB/GT/权重或预测数组。
最新QA状态见 `qa/report.json`，SHA256绑定见 `DELIVERY_SEAL.json`。

本轮没有上传GitHub，没有覆盖旧论文、原结果或主checkout，也没有重跑成功实验。
原监控已结束，无合法活跃科学unit；原trace自动化保持PAUSED，不新建自动化。
