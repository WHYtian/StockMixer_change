# 股票预测与排序中的不确定性研究

基于 StockMixer 的研究快照：考察 MC Dropout、跨种子分歧和异方差建模，能否为股票收益预测与横截面排序提供额外信息。本仓库不是 StockMixer 官方实现，也不声称已经获得稳定的样本外收益提升。

## 当前结果（2026-09-30）

A2 已启动：两个时间窗口、五种子共十次训练，将基础训练、checkpoint 选择、
风险拟合和评价分离；增加容量匹配的打乱特征对照及多重检验校正。
目前没有 A2 效果结论。见 [A2 固定协议](plan/protocol_rank_uncertainty_a2_20260930.md)
和 [DEUP 源码审计](plan/deup_code_audit_20260930.md)。下文为已完成的 A1 结果。

最新 A1 实验使用 NASDAQ 的 5 个随机种子，在开发期进行顺序交叉拟合，评估 156 个交易日、158,497 个有效股票日。

- **主要指标未通过正向验证**：加入四种不确定性特征配置后，排名错误预测 MSE 相对控制变量对照的配对 95% 区间均包含零。
- **重要诊断**：只使用预测排名位置的几何基线，与实际排名错误的相关性为 0.231933，高于原始 MC 标准差的 0.124027。因此，“不确定性与错误相关”本身不足以证明增量信息。
- **探索性观察**：加入跨种子排名分歧，在预测排名十分位内保留 60% 股票时，RankIC 从 0.025877 变为 0.028993。该结果是辅助指标、未经多重比较校正；基础模型使用整个验证期选取检查点，因此不能当作独立样本外提升。

完整统计、对照与限制见 [A1 结果报告](plan/results_rank_uncertainty_a1_20260930.md) 和 [预先固定的协议](plan/protocol_rank_uncertainty_a1_20260930.md)。预测区间覆盖率升高也不能单独作为预测能力提升的证据，必须同时考虑区间宽度与朴素基线。

## 仓库内容

- `StockMixer/research/`：训练、MC Dropout、概率评估、因子控制、回测诊断、A 股数据预处理及排名风险分析。
- `StockMixer/tests/`：54 项测试，覆盖模型模式切换、数据边界、指标、筛选、排名风险及 A2 时间隔离与对照检验。
- `StockMixer/research/results/`：A1 汇总、逐日指标、输入指纹，以及此前 NASDAQ E3 的基础汇总。
- `plan/`：文献与开源项目调研、研究计划、NASDAQ / A 股报告、A1 协议及结果。

这是研究代码与轻量结果的整理版，不是本地目录的完整备份。原始数据、逐股票预测、模型权重、旧探索性 notebook、机器专用调度脚本和运行日志没有上传。历史文档及 manifest 保留原始运行路径，以便追溯；在其他机器复现时须替换路径。报告引用的部分额外回测产物未包含在此快照中。

A2 的轻量结果保存在 `StockMixer/research/results/rank_uncertainty_a2_20260930/`；完整逐日表未上传，报告和汇总 JSON 已保留。

## 安装与测试

已测试环境为 Python 3.12；依赖版本记录在 [requirements.txt](requirements.txt)。按机器的 CPU/CUDA 环境安装合适的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd StockMixer
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m pytest tests -q
```

测试使用合成数据，不需要下载股票数据。通过测试不代表实验假设成立。

## 运行实验

NASDAQ 加载器需要 `eod_data.pkl`、`gt_data.pkl`、`mask_data.pkl`、`price_data.pkl` 四个对齐文件。数据来源可参考 [上游 StockMixer](https://github.com/SJTU-DMTai/StockMixer)，但历史本地数据经过变更；只有与 A1 manifest 中 SHA256 一致的数据和预测文件，才对应本次记录的输入。不要加载不可信的 pickle 文件。

在 `StockMixer/` 目录中，训练单个随机种子的示例：

```bash
python -m research.run_experiment \
  --dataset /path/to/NASDAQ --output-dir /path/to/new-run \
  --seed 20260813 --dropout 0.1 --dropout-sites mixer \
  --select-metric rank_ic --select-smoothing 5 \
  --epochs 300 --patience 30 --min-epochs 30 \
  --mc-samples 50 --device auto
```

为 20260813–20260817 分别训练并生成验证期预测后，可运行 A1：

```bash
python -m research.rank_uncertainty \
  --run-dirs /path/to/run-20260813 /path/to/run-20260814 \
    /path/to/run-20260815 /path/to/run-20260816 /path/to/run-20260817 \
  --dataset /path/to/NASDAQ \
  --protocol ../plan/protocol_rank_uncertainty_a1_20260930.md \
  --output-dir /path/to/new-a1-output
```

A1 分析只读取验证期预测；输出目录必须不存在。训练脚本本身会生成验证期和测试期输出。由于未附数据、权重和预测，克隆仓库后可以直接运行单元测试，但不能直接重算历史实验结果。

## 尚未解决的问题

外层时间验证 A2 正在运行，尚无结果；标准化标签与原始收益的概率指标单位、MSE 与 beta-NLL 的损失权重可比性、MC 偶然不确定性的聚合方式，以及简化回测的成交/权重假设，仍需要进一步审计。详见研究计划。当前结果不应写成“显著提升样本外预测准确率”或实盘表现。

A2 训练入口（在 `StockMixer/` 内，五个种子分别运行，输出路径不得已存在对应运行）：

```bash
python -m research.rank_a2 train --seed 20260813 \
  --root /path/to/a2-runs --dataset /path/to/NASDAQ \
  --protocol ../plan/protocol_rank_uncertainty_a2_20260930.md
```

全部十次训练完成后统一分析：

```bash
python -m research.rank_a2 analyse \
  --root /path/to/a2-runs --dataset /path/to/NASDAQ \
  --protocol ../plan/protocol_rank_uncertainty_a2_20260930.md \
  --output /path/to/new-a2-analysis
```

## 来源

模型骨干基于 AAAI 2024 的 StockMixer；来源、基准提交及方法参考见 [ATTRIBUTION.md](ATTRIBUTION.md)。研究用途，不构成投资建议。
