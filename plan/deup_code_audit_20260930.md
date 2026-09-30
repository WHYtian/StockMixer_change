# DEUP 金融实现：定点源码审计

日期：2026-09-30。范围：静态阅读，没有安装或执行第三方代码，没有复现其业绩。

仓库：[ursinasanderink/deup](https://github.com/ursinasanderink/deup)
固定提交：`95affa2e1afaec7dd262ab00d3e96933a2f26d74`。

## 对此前调研的修正

此前只记录《When Alpha Breaks》论文所给项目链接无法访问。此次确认作者另有
可访问的 DEUP 库，不应把旧链接失效解释为没有开源实现。
原方法研究仓库 [MJ10/DEUP](https://github.com/MJ10/DEUP) 与这个金融扩展库应区分。

## 核实的实现事实及使用边界

1. `CrossSectionalDEUP` 确实存在，内部使用 `DEUPRanker`，具有按日期分组的
   walk-forward 接口。不是只能从 README 猜测存在的功能。
2. `horizon` 在该封装中用于选择目标列名；默认 `embargo=1`，没有根据 horizon
   自动扩大。`PurgedWalkForward` 按日期组留出 embargo，未接收逐条标签成熟时间。
   **使用多日收益标签时必须显式设置并检查边界，不能只设置 horizon 就假定无泄漏。**
3. `health_report` 会读取目标列，缺少现成 rank_loss 时还用真实目标计算误差。
   因此这是依赖标签的诊断接口，不能直接当成预测当日可用的实时交易开关。
   这不意味着所有 `predict` 接口都读未来标签。

来源：[finance.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/domains/finance.py)、
[splitters.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/splitters.py)。

4. `finance_walkforward.py` 提供对已生成排名误差拟合 LightGBM 的路径：50 棵树、
   深度 3、8 叶；按 fold 标识扩展训练。该函数自身不重新审核上游 checkpoint
   选择，也没有按标签成熟日进行 purge；特征可用率筛选读取整个输入面板。
   接入时需要由外层流程保证时间隔离，不能把封装等同于完整无泄漏实验。

来源：[finance_walkforward.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/domains/finance_walkforward.py)。

5. `RankResidualizer` 已经处理排名/信号强度的机械耦合，因此**“控制排名几何”
   不能宣称是本项目首创**。我们的有限样本几何对照和分层评估属于具体实现与检验。
6. `DEUPRanker` 默认 `decompose=False`；错误风险预测不自动等于已经识别出认知
   不确定性。库中噪声估计器输出方差尺度，排名绝对误差不能不经量纲检查直接相减。
   我们的 A1/A2 明确只估计排名错误风险，不声称完成认知/偶然不确定性分解。

来源：[decompose.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/core/decompose.py)、
[estimators.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/estimators.py)、
[aleatoric.py](https://github.com/ursinasanderink/deup/blob/95affa2e1afaec7dd262ab00d3e96933a2f26d74/src/deup/core/aleatoric.py)。

## 对当前实验的决定

- 不在已冻结的 A2 中临时替换风险预测器或添加新方法；避免看结果后修改比较范围。
- 本轮先修正上游时间隔离，保留与 A1 一致的二次岭回归，加入容量匹配的打乱对照。
- 后续若做 LightGBM/DEUP 对照，须另行冻结协议，显式分离基础模型和风险模型的
  时间边界，保持标签单位一致，重新登记比较数量。
- 不从库的命名、README 或其原论文收益数字推断它会提升本项目 RankIC。
