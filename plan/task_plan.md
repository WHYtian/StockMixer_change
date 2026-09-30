# Task Plan: 重跑 StockMixer 不确定性项目,产出可写进简历的结果

创建日期:2026-09-29
状态:**方案待用户确认,尚未开始实现**

## Goal

在 `StockMixer/research/` 新框架下重跑全部实验,使简历上的每个数字都有
多种子均值±标准差、显著性检验和日志文件支撑。

## 已核实的事实(2026-09-29)

- 数据:NASDAQ 1026 只 × 1245 天 × 5 特征,mask 有效率 99.86%;SP500 474 只 × 2526 天。NYSE 为 0 字节,不可用。
- 切分:train 0–756 / valid 756–1008 / test 1008–1245(237 个测试日)。
- 唯一完整运行:`research/results/baseline_seed20260813`,test IC 0.0216,RankIC 0.0355,RankICIR 0.303。
- 该运行 100 epoch 用时约 17 分钟(A30)。best_model.pt 在最后 1 分钟才保存,说明 valid MSE 到第 100 epoch 仍在下降,early stopping 未触发。
- 旧日志 `src/r2_dropout/`、`src/stockmixer_new.csv` 使用了错误的 RIC 定义和未关闭的 Dropout,不能引用。
- 登录节点 beech13 只有 GTX 1050 Ti,驱动过旧,无 Slurm 客户端。
- `/vol/bitbucket/hw2025/env/bin/python`(3.14)当前 `import torch` 失败,也没有 pandas。
- home 配额 11184M / 11719M,实验输出必须写到 `/vol/bitbucket/hw2025/`。

## 补充核实(2026-09-29,第二轮)

- GPU:gpu32–gpu36 各有 1 张 RTX 4080 16GB,均空闲,16 核 / 60GB 内存;gpu31 不通(No route to host)。
- 环境损坏原因:`/vol/bitbucket/hw2025/env` 建于 Python 3.12.3,系统 Python 已升级为 3.14.4,
  `bin/python` 指向 `/usr/bin/python3`,因此 `lib/python3.12/site-packages` 里的 torch/scipy/pandas 都不可见。
  登录节点和 gpu32 上表现一致。系统中已无 python3.12。`uv` 可用(`~/.local/bin/uv`)。
- 数据源连通性(从登录节点):Yahoo chart API 可用(需浏览器 UA,否则 429);
  东方财富 K 线 API 可用(A 股,已取到 2026-09 数据);Stooq 被 JS 验证拦截;baostock 端口超时。
- `dataset/SP500/sp500_ticker.csv` 含 505 只股票的 Sector 字段,可用于行业中性分析。

## 数据方案(待用户选择市场)

- 保留 NASDAQ 2013–2017 作为与论文可比的复现基准。
- 新建一份 2016–2026 的数据集作为主结果,用 walk-forward 评估。

## 用户决定(2026-09-29)

- 主数据集:A 股 沪深300 + 中证500,2015-01-01 至 2026-09-28。
- 环境:修复原 env(已完成,见下)。
- GPU:ssh gpu32–gpu36,RTX 4080。

## 已执行(2026-09-29)

- [x] 环境修复:`env/bin/python` 改指向 `/vol/bitbucket/hw2025/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`;
      `pyvenv.cfg` 已更新,原文件备份为 `pyvenv.cfg.bak-20260929`。gpu32 上验证 torch 2.11.0+cu130,CUDA 可用。
- [x] 在 env 中新增 `xlrd`(读取中证指数成分股 xls)。
- [x] 新增 `StockMixer/research/ashare_fetch.py`;3 只股票冒烟测试通过(每只约 2820–2850 行)。
- [ ] 全量下载 800 只,输出 `/vol/bitbucket/hw2025/stock/data/ashare_csi800_20150101_20260928/`,约 90 分钟。

- [x] E1 已启动(09:57 前后,gpu32):NASDAQ 基线 × 5 种子,epochs 300,patience 30。
      输出 `/vol/bitbucket/hw2025/stock/runs/nasdaq_e1_baseline_20260929/`。
- [x] `research/run_experiment.py`:MC 运行同时保存 `point_prediction` 和 `point_*` 指标。
- [x] E2 已启动(09:04,gpu33/34/35 分别跑 dropout 0.1/0.2/0.3,各 5 种子,MC=50)。
      输出 `/vol/bitbucket/hw2025/stock/runs/nasdaq_e2_mcdropout_20260929/`(含 code_snapshot 和 code_diff.patch)。
- [x] 新增 `research/aggregate.py`:分组均值±标准差、种子集成、对基线的逐日 RankIC 配对 t 检验和 Newey-West t。冒烟数据上通过。

- [x] 根因取证(09:19):在已训练的 dropout=0.2 模型上逐个位置打开 dropout,测收益率的 MC 标准差:
      Mixer 块内 0.0167;`mixed`(channel_fc 之前)0.0523;`time_fc` 输入 0.0603;`stock_fc` 输入 0.0029;全部打开 0.0864。
      同一天 |真实收益| 均值 0.0160。结论:输出头前的两处 dropout 贡献了绝大部分方差。
- [x] `research/model.py` 增加 `head_dropout`;`run_experiment.py` 增加 `--dropout-sites {all,mixer}` 和 `--select-metric {mse,rank_ic}`。默认值保持旧行为。
- [x] 新增 `tests/test_model_dropout.py`(4 项通过);env 中新增 `pytest`。
- [x] 新增 `research/uncertainty_eval.py`(U1、U2):波动率基线、偏秩相关、PICP、ENCE、RankIC-覆盖率曲线、按交易日 bootstrap。
- [ ] 探针运行(09:21,gpu36):dropout 0.1 / 0.2,`--dropout-sites mixer`,seed 20260815,选择指标仍为 mse(一次只改一个变量)。
      输出 `/vol/bitbucket/hw2025/stock/runs/nasdaq_e2b_mixerdropout_probe_20260929/`。

- [x] 探针验证(09:27):仅 Mixer 内 dropout 时验证 MSE 持续下降到约 0.0005(旧设置升到 0.018)。
      取探针 25–29 epoch 的检查点在 60 个测试日上测得 sigma 均值 0.0031 / 0.0035(dropout 0.1 / 0.2),
      |真实收益| 均值 0.0118,|点预测| 均值 0.0022。量级恢复正常。
- [x] E2c 已启动(09:28,gpu33/34/35):dropout 0.1/0.2/0.3 × 5 种子,`--dropout-sites mixer --select-metric rank_ic`,MC=50。
      输出 `/vol/bitbucket/hw2025/stock/runs/nasdaq_e2c_mixerdropout_rankic_20260929/`。
      注意:E1 基线按 mse 选模型,E2c 按 rank_ic 选,二者不能直接比较;需要补一组按 rank_ic 选的基线。

- [x] 代码审查(09:27 返回)。已确认正确:收益率定义、train/valid/test 目标日不重叠、测试预测第 j 列对应目标日 valid_end + j、
      波动率窗口止于目标日前一天、Newey-West 实现、偏秩相关实现、MC 采样的模式切换。
- [x] 已按审查修复(09:30):
      `aggregate.py` 按除种子外的全部配置分组,校验各运行的测试数据一致,新增种子层面的 Welch 检验;
      `uncertainty_eval.py` 改为按交易日的移动块 bootstrap(块长 10 天)、对"sigma 减波动率"的差值给区间、
      波动率剔除缺失日相邻的收益、两种 ENCE 用同一误差、数据集取自运行配置;
      `mc_dropout.py` 恢复调用前的模式;`run_experiment.py` 拒绝 dropout=0 配 MC、拒绝 horizon≠1、处理 NaN 选择分数;
      `ashare_build.py` 复牌日掩码置 0、5% 涨跌停启发式单独成列且不用于 20% 板块。
- [x] 新增 `tests/test_evaluation.py`(7 项);全部 11 项测试通过。
- [ ] 审查指出但未解决:A 股成分股用的是当前名单(审查评级 critical)。需要历史成分股数据源,待与用户确认。
- [ ] 审查指出但未解决:回测指标的掩码用到了目标日是否停牌(NASDAQ 上平均每天 0.10 只,A 股上影响更大)。在写 backtest.py 时处理。
- [ ] 审查指出但未解决:`panel.npz` 还没有加载器,A 股实验尚不能运行。

- [x] 用户选择方案 A(历史成分股)。数据源改为 chenditc/investment_data 的 Qlib 格式发布包(release 2026-09-28)。
      归档 sha256 与官方 manifest 一致(73f17f04…dede5)。位置 `/vol/bitbucket/hw2025/stock/data/qlib_cn_20260928/`。
      校验:每个交易日 csi300 恰好 300 只、csi500 为 500–501 只;2026-09-28 的成分与中证官网文件 300/300、500/500 完全一致。
- [x] 发现腾讯 fqkline 的后复权收益率有系统性偏差:120 只股票平均每只约 833 个交易日与 Qlib 收益率相差超过 0.002
      (例 sh600000 2015-02-02:Qlib −0.0297,未复权 −0.0297,腾讯 −0.0259)。Qlib 复权收益与未复权收益只在除权日不同(平均每只 11 天)。
      结论:腾讯数据不可用于算收益率。已停止下载(停在 292/800),`ashare_fetch.py` 已移到 `temp/trash/`。
- [x] `research/ashare_build.py` 重写为读取 Qlib 数据;`research/data.py` 新增 `load_panel`、`load_dataset` 和按窗口归一化。
      面板:1579 只(2015 年以来曾属于两个指数的全部股票,含已退市)× 2853 个交易日;
      每日可用股票数 最小 465 / 中位 797 / 最大 800;成分股交易日中 97.6% 可用;收益率范围 −20.09% 至 +20.16%。
      输出 `/vol/bitbucket/hw2025/stock/data/ashare_pit_csi800_20150101_20260928/`。
- [x] 测试增至 13 项,全部通过。A 股 CPU 冒烟训练通过。
- [ ] A 股切分:train 2015–2023(索引 0–2189)/ valid 2024(2189–2431)/ test 2025-01 至 2026-09(2431–2853,422 天)。
      等 NASDAQ E2c 跑完、GPU 空出后启动。

- [x] E1 完成(09:42):NASDAQ 基线,按 mse 选,5 种子。测试 RankIC 0.0243 ± 0.0081,IC 0.0193 ± 0.0059;5 种子集成 RankIC 0.0318。
- [x] E2c 完成(10:02):按 rank_ic 选,仅 Mixer 内 dropout,5 种子,237 个测试日。汇总在
      `/vol/bitbucket/hw2025/stock/runs/nasdaq_e2c_mixerdropout_rankic_20260929/summary/`。
      | 组 | 点预测 RankIC | 集成 RankIC | 与基线之差 | 种子层面 Welch p |
      | dropout 0.0 | 0.0327 ± 0.0058 | 0.0395 | — | — |
      | dropout 0.1 | 0.0325 ± 0.0075 | 0.0406 | −0.0002 | 0.96 |
      | dropout 0.2 | 0.0254 ± 0.0095 | 0.0409 | −0.0073 | 0.19 |
      | dropout 0.3 | 0.0234 ± 0.0105 | 0.0424 | −0.0094 | 0.13 |
      结论:没有证据表明 dropout 提升 RankIC。"IC 提升 8%"不成立。
- [x] 简单因子基线(同一测试期):NASDAQ 1 日反转 RankIC 0.0325(RankICIR 0.249),5 日反转 0.0135,16 日反转 0.0150,低波动 0.0063。
      A 股(2025-01 至 2026-09,成分股内):1 日反转 0.0389,5 日反转 0.0326,16 日反转 0.0281,低波动 0.0321。
      StockMixer 单模型的 RankIC 与 1 日反转因子持平。
- [x] U1/U2 在 15 个 MC 运行上完成。训练充分的 11 个模型:sigma 均值约 0.003–0.008;
      控制波动率后 sigma 与 |误差| 的偏秩相关 0.04–0.15,区间均不含 0。
      保留 sigma 最低的 60% 股票时 RankIC 上升(dropout 0.2 组均值 0.0254 → 0.0340),与按波动率筛选相比的差值区间多数包含 0。
      90% 高斯区间的实际覆盖率 0.11–0.40:MC sigma 不能直接当预测区间用。
- [ ] 问题:15 个 MC 运行中有 4 个在第 1 个 epoch 取得最高验证 RankIC 并在第 31 个 epoch 停止
      (dropout0.1 seed 0813、0817;dropout0.2 seed 0817;dropout0.3 seed 0817)。
      这些模型 sigma 为 0.044–0.056,测试 RankIC 反而最高(0.039–0.041)。属于模型选择规则的缺陷,未从统计中剔除。
      待定:改用验证 RankIC 的滑动平均并设最少 epoch 数后重跑。

## 预先登记的协议(2026-09-29 11:55 定稿,用户已同意;看到新结果后不得修改)

- 叙事方向:不确定性用于筛选和风险控制,不声称提升 IC。
- 模型选择:验证集 RankIC 的 5 个 epoch 尾部滑动平均取最大;在第 e 个 epoch 取得最大值时保存第 e 个 epoch 的权重。
- 提前停止:滑动平均连续 30 个 epoch 无改善,且已训练不少于 30 个 epoch。
- epoch 上限:NASDAQ 300,A 股 200。
- 种子:20260813–20260817,共 5 个,全部报告,不剔除任何运行。
- 分组:dropout 0.0 / 0.1 / 0.2 / 0.3;dropout 仅在 Mixer 块内;MC 采样 50 次。
- 主要指标:测试集 RankIC(单模型均值 ± 标准差,以及 5 种子集成)。
- 主要检验:种子层面 Welch 检验;交易日层面检验只作补充。
- 不确定性检验:控制波动率后的偏秩相关;保留 sigma 最低 60% 股票时的 RankIC 与按波动率筛选之差(10 天块 bootstrap)。
- 必须同时报告的基线:1 日、5 日、16 日反转,16 日低波动。
- 命令行:`--select-metric rank_ic --select-smoothing 5 --min-epochs 30 --patience 30 --dropout-sites mixer --mc-samples 50`。

- [x] 11:56 停止旧规则下的 A 股基线(gpu32)和探针(gpu36);它们的结果不再使用。
- [x] 11:56 按协议启动:NASDAQ 20 个运行(gpu33–36,`runs/nasdaq_e3_protocol_20260929/`);
      A 股 dropout 0.0 × 5(gpu32,`runs/ashare_a2_protocol_20260929/`)。
- [ ] NASDAQ 跑完后在 gpu33–35 启动 A 股 dropout 0.1 / 0.2 / 0.3。
- [x] `run_experiment.py` 新增 `--select-smoothing`、`--min-epochs`;测试增至 14 项,全部通过。

- [x] 11:58 重启全部 25 个协议运行:训练脚本改为同时保存验证集预测(`valid_predictions.npz`),
      回测的规则参数只能在验证集上选。11:56 启动的那批移到 `runs/_superseded/`。协议内容未变。
- [x] 新增 `research/backtest.py`(U3):种子均值预测;sigma 取 MC Dropout / 种子集成标准差 / 历史波动率;
      规则 mu、z(mu) − eta·z(sigma)(eta ∈ 0.25/0.5/1.0)、剔除 sigma 最高的 10%/20%/40%;
      参数按验证期 10 bps 成本下的净 Sharpe 选;Top-50 等权、每日调仓;成本 0/10/20 bps;
      决策日可买性只用当日收盘前的信息(A 股剔除决策日涨停),目标日停牌的持仓收益记 0。
- [x] 新增 `tests/test_backtest.py`(4 项);全部 18 项测试通过。

## 预先登记的附加分析(2026-09-29 定稿,用户指定按 X1→X6 顺序执行;全部报告,不论结果)

数据:协议运行 `nasdaq_e3_protocol_20260929` 与 `ashare_a2_protocol_20260929` 的测试集预测。所有阈值和参数只在验证集上选。

- X1 因子中性化:每个交易日把预测对 1 日、5 日、16 日反转和 16 日波动率(均取横截面秩)做回归,取残差,
  报告残差的 RankIC(单模型均值 ± 标准差、5 种子集成)以及预测与各因子的平均秩相关。区间用 10 天块 bootstrap。
- X2 sigma 与 |预测| 的每日秩相关:报告每个种子的均值及 5 个种子间的标准差;同时报告 sigma 与波动率的相关。
- X3 市场层面开关:当日全市场 sigma 中位数高于其过去 60 个交易日的第 80 百分位时空仓。
  对照:用全市场波动率中位数做同样的开关;随机空仓同样比例的交易日。指标:净 Sharpe、最大回撤、空仓天数占比。成本 10 bps。
- X4 排名位移预测:在验证集上用 LightGBM 预测 |预测秩百分位 − 真实秩百分位|,输入为 sigma、波动率、|预测|、16 日平均成交量的横截面秩;
  验证集前 70% 训练、后 30% 选超参数;测试集上用预测的位移代替 sigma 做 60% 保留筛选,与 sigma、波动率筛选比较 RankIC。
- X5 一致性筛选:只在 5 个种子中至少 4 个都排进当日前 20% 的股票里,按种子均值预测选 Top-50;不足 50 只时全部持有。
- X6 异方差输出头:beta-NLL(beta=0.5)+ 原排序损失,5 个种子,按协议选模型。检查 RankIC 不低于基线(种子层面 Welch 检验);
  5 个模型组成集成,分解偶然与认知不确定性,分别做 X1–X3 同样的检验。

- [x] 12:08 X1、X2 脚本 `research/factor_analysis.py` 完成;X3、X5 已加入 `research/backtest.py`(参数固定:60 日回看、80% 分位、5 选 4、前 20%)。
      在已作废的 E2c 运行上试跑通过(仅用于调试脚本,不作为结论)。
- [x] 12:12 X6 代码完成:`model.py` 增加方差输出头和 `heteroscedastic_loss`;`run_experiment.py` 增加 `--heteroscedastic --beta`;
      `uncertainty_eval.py` 增加 `--sigma {mc_std,aleatoric_std,total}`,输出文件名改为 `uncertainty_eval_<sigma>.json`。
      测试增至 22 项,全部通过。等 GPU 空出后启动。
- [ ] X4 需要在 env 中安装 lightgbm,尚未安装。

- [x] 12:16 用户假设:此前观察到的 MSE 下降来自 dropout 本身(正则化),不是来自 MC 平均。已有证据:
      (a) 全零预测的测试 MSE 为 0.000378;E1 基线(按 mse 选)为 0.000384 ± 0.000003。所有模型的 MSE 都不低于全零预测。
      (b) 旧日志(单种子,`src/r2_dropout/`)在最优验证 epoch 处:原版 test_mse 0.000386;dropout 各组 0.000387–0.000453;
          同一模型的 MC 与非 MC 的 test_mse 相差不超过 0.000003(0.4、0.5 两组除外,MC 更高)。
      (c) 协议运行中训练充分的模型:点预测与 MC 均值的 MSE 几乎相同(例 dropout0.1 seed 0814:0.000521 对 0.000524)。
          在第 35 个 epoch 就停止的模型:MC 均值的 MSE 明显更低(例 dropout0.3 seed 0817:0.003925 对 0.001672)。
- [ ] E4(12:16 启动,gpu33/34):检验 dropout 作为正则项对 MSE 的影响。dropout 0.0/0.1/0.2/0.3 × 5 种子,仅 Mixer 内,
      按验证 MSE 的 5 epoch 滑动平均选模型,其余同协议。输出 `runs/nasdaq_e4_mse_selected_20260929/`。
      预先定下的比较:测试 MSE,(i) 基线点预测 对 dropout 点预测(dropout 的作用);(ii) dropout 点预测 对 MC 均值(MC 的作用);
      种子层面 Welch 检验;同时报告全零预测的 MSE。
- [x] 12:16 A 股 dropout 0.1(gpu35)、0.2(gpu36)按协议启动。dropout 0.3 和 X6 待机器空出。

- [x] 12:17 NASDAQ 协议运行 20/20 完成。结果见 `plan/results_nasdaq_protocol_20260929.md`。
      要点:dropout 不提升 RankIC;集成剔除四因子后仍有 0.026–0.034;计入 10 bps 成本后全部规则跑输等权基准;X3、X5 为阴性结果。
- [ ] 12:20 用户要求精读 Gal & Ghahramani(ICML 2016)和单股数据对应的论文,寻找实验灵感。两个阅读任务进行中。

## 预先登记:X7 补上观测噪声项(2026-09-29 12:35 定稿,来自 Gal & Ghahramani 式 6–8)

数据:NASDAQ 协议运行中 15 个 dropout 模型的验证集和测试集预测。只用已保存的 MC 均值和 MC 标准差(单次前向未保存,
因此用高斯近似:均值取 MC 均值,方差取下列各式)。参数只在验证集上用最大对数似然拟合,测试集只评估一次。
- (a) 仅 MC:方差 = MC 方差
- (b) 加法(论文做法):方差 = c + MC 方差
- (c) 乘法(附录 5.1.1):方差 = k² · MC 方差
- (d) 常数:零均值,方差 = c
- (e) 波动率:零均值,方差 = k² · 16 日波动率²
- (f) 波动率 + MC 均值:均值取 MC 均值,方差 = k² · 16 日波动率²
指标:测试集每个股票日的平均对数似然、90% 区间覆盖率。按交易日配对比较 (b) 与 (e)、(f)。

- [x] 12:40 X7 完成(`research/noise_term.py`)。补上噪声项后覆盖率 0.34–0.37 → 0.94;MC 方差只占总方差 2%–17%;
      对数似然低于零均值波动率基线(15/15 显著)。详见结果文件第 8 节。
- [ ] 待定(未登记):对模型均值做收缩 mean = a·mu,a 在验证集拟合。X7 显示预测幅度偏大是对数似然差的主因。

## 预先登记:X8 换手控制(2026-09-29 12:45 定稿)

依据:只在 NASDAQ 验证期(252 天,基线 5 种子集成)上探索,未接触测试集。验证期结果(成本 10 bps 的净 Sharpe):
每日 Top-50 1.20(换手 0.41);退出阈值 100/200/400 分别 1.56/1.57/1.56;5 日平均得分 1.40;
5 日平均得分 + 退出阈值 200 为 2.00(换手 0.09);等权基准 1.30。
另:1 日预测对未来 1/2/3/5/10 日收益的验证期 RankIC 为 0.0315/0.0348/0.0388/0.0434/0.0580。

固定规则(不再调整):得分取最近 5 个交易日预测的平均;持有 50 只;已持有的股票只有在当日排名跌出前 200 才卖出;空位按排名从高到低补足。
测试:NASDAQ 测试期评估一次,四个 dropout 组都报告,成本 0/10/20 bps,与每日 Top-50 和等权基准比较。
同一规则原样用于 A 股(不在 A 股上重新调参)。

- [x] 12:50 X8 完成。NASDAQ 测试期:换手 0.37–0.49 → 0.08–0.12;10 bps 成本下四组年化 22.0%–29.5%,均高于等权基准 15.8%。结果文件第 9 节。
- [x] 12:39 A 股协议运行 14/15 完成。单模型测试 RankIC 在 −0.019 至 0.029 之间,均值接近 0,低于简单因子(0.028–0.039)。结果文件第 10 节。

## 预先登记:D1 横截面标准化标签(2026-09-29 定稿,启动训练前写入)

背景:A 股协议运行的测试 RankIC 接近 0,低于简单因子。假设:原始收益率标签使回归损失主要拟合当日市场整体涨跌。
改动:训练目标改为当日可交易股票收益率的横截面 z-score,截断到 ±3(`--label cs_zscore`)。评估仍用原始收益率。其余同协议。
分组:dropout 0.0 和 0.1(仅 Mixer 内,MC 50 次),各 5 个种子;A 股和 NASDAQ 都跑。
主要比较:测试 RankIC,cs_zscore 对 raw(同 dropout、同数据集),种子层面 Welch 检验;并与四个简单因子并列报告。
判定:若 A 股 5 种子集成的测试 RankIC 仍低于 1 日反转因子(0.0389),则记为"标签不是主要原因",转查输入特征。
说明:A 股测试期的基线结果已经看过;本项改动是在看到该结果之后提出的。

## 预先登记:D2 用不确定性决定仓位(2026-09-29 约 12:56 定稿,先于测试期评估)

基础组合:X8 的换手控制规则(已冻结)。在其上比较两种用法,sigma 取最近 5 日的平均:
- 加权:持仓内权重 ∝ 1/sigma^gamma,gamma ∈ {0.5, 1.0},单只权重上限为等权的 3 倍后重新归一。
- 过滤:先从候选中剔除 sigma 最高的 q,q ∈ {0.1, 0.2, 0.4},再按 X8 规则选股,持仓等权。
sigma 来源:MC Dropout、种子集成标准差、16 日历史波动率(对照)。
调参:gamma、q 在验证期按 10 bps 净 Sharpe 选,每个来源各选一个;测试期评估一次。
指标:10 bps 下的年化收益、波动、Sharpe、最大回撤、换手;与等权的 X8 组合逐日配对,
报告 Sharpe 之差的 10 天块 bootstrap 95% 区间。
判定:若 MC Dropout 或种子集成的结果不优于历史波动率对照,则结论为"模型不确定性在仓位上没有提供波动率之外的价值"。

- [x] 12:55 D1 启动:A 股 cs_zscore dropout 0.0(gpu35)、0.1(gpu36);NASDAQ cs_zscore dropout 0.0、0.1(gpu32)。
      输出 `runs/ashare_d1_cszscore_20260929/`、`runs/nasdaq_d1_cszscore_20260929/`。
- [x] 约 12:57 D2 完成:阴性结果。22 个组合与等权 X8 的 Sharpe 之差的区间全部包含 0。结果文件第 11 节。

## 预先登记:D3 多日收益预测(2026-09-29 约 12:58 定稿,先于训练启动)

改动:训练目标改为未来 h 个交易日的累计收益,h ∈ {5, 10};决策日不变(窗口最后一天收盘)。
目标期内任一天停牌的股票不参与该样本。训练/验证/测试的目标期互不重叠(各段起点处自动空出 h−1 天)。
分组:NASDAQ,dropout 0.0,原始标签,5 个种子,其余同协议(模型选择用对 h 日收益的验证 RankIC)。A 股待 D1 结果确定标签后再跑。
比较(与 h=1 的协议基线):
- (i) 5 种子集成预测对"下一日收益"的测试 RankIC,以及对各自 h 日收益的 RankIC;
- (ii) X8 规则下 10 bps 的净 Sharpe、年化收益、换手;
- (iii) 每日 Top-50(不做换手控制)的换手,用来衡量预测本身的稳定性。
统计:h 日收益标签相互重叠,RankIC 的区间用 10 天块 bootstrap。
判定:若 (ii) 的净 Sharpe 不高于 h=1 基线,记为阴性。

- [x] 约 12:59 D3 代码完成(`data.forward_return`,`split_offsets` 起点改为 start − lookback;h=1 时与原结果逐位一致)。测试 30 项全部通过。
- [x] 12:59 D3 启动:NASDAQ horizon 5(gpu35)、horizon 10(gpu36),各 5 个种子。输出 `runs/nasdaq_d3_horizon_20260929/`。

- [x] 13:26 D1 完成。A 股集成 RankIC 0.0005 → 0.0250(dropout 0.0),仍低于 1 日反转 0.0389;NASDAQ 上无帮助。
      按判定转查输入特征。结果见 `plan/results_ashare_20260929.md`。

## 预先登记:D4 输入特征(2026-09-29 13:30 定稿,先于训练启动)

依据:原 StockMixer 的 NASDAQ 数据用的是 5/10/20/30 日均线和收盘价共 5 个特征;我们的 A 股输入是 OHLCV 原始水平。
改动:A 股输入改为 [MA5, MA10, MA20, MA30, close](复权收盘价计算,均线至少需要一半窗口的有效交易日),
每个窗口内全部除以窗口最后一天的收盘价(`--features ma`)。其余同协议。
分组:features=ma × label ∈ {raw, cs_zscore},dropout 0.0,5 个种子,共 10 个运行。
选择规则:A 股的四种组合({ohlcv, ma} × {raw, cs_zscore})按"验证期 5 种子集成 RankIC"选出一个作为主结果;
四种组合的测试结果全部报告。此后 A 股上的配置选择一律只看验证期。
判定:被选中的组合,其测试期集成 RankIC 若仍低于 1 日反转(0.0389),则记为"StockMixer 在 A 股日频上不优于简单因子",
A 股部分改为报告模型与因子的组合,不再继续改模型。

## A 股数据的已知限制

- (已作废)最初用腾讯 fqkline + 当前成分股名单,见上方 2026-09-29 的两条记录。
- 现行数据:Qlib 格式众包数据,历史成分股。未识别 ST;涨跌停由复权收益率和收盘=最高/最低推断;无行业字段。
- 众包数据的上游是 tushare 等,未与交易所官方数据逐日核对。

## Phases

- [ ] P0-0 环境与备份:修复/新建环境;输出目录放 bitbucket;代码推到私有 GitHub 仓库
- [ ] P0-1 E1 基线:原版 StockMixer × 5 种子,epoch 上限 300
- [ ] P0-2 E2 不确定性模型:MC Dropout p∈{0.1,0.2,0.3} × 5 种子;Deep Ensemble(复用 E1 的 5 个模型)
- [ ] P1-1 E3 不确定性质量:sigma 与 |误差| 的相关性、分位误差、PICP/MPIW、滚动 conformal 校准
- [ ] P1-2 E4 不确定性调整选股 + 含成本回测
- [ ] P2   E5 SP500 复现 / walk-forward;LightGBM、MASTER、nFBST(本轮不做,除非时间富余)
- [ ] 汇总表、图、简历描述

## 不确定性实验扩展(2026-09-29 调研后新增,详见 notes.md)

- [ ] U1 评估框架:波动率基线、按日计算的 Spearman(sigma, |误差|)、ENCE、PIT、IC-覆盖率曲线、按交易日 bootstrap
- [ ] U2 sigma 与 |mu|、与过去 16 天已实现波动率的相关性
- [ ] U3 `mu - eta*sigma` 选股,eta 在验证集上选;sigma 取 MC Dropout / 集成 / 波动率基线三种
- [ ] U4 滚动 conformal(归一化残差)+ ACI
- [ ] U5 异方差输出头 + beta-NLL(beta=0.5)× 5 种子;检查 IC 不低于 MSE 训练的基线
- [ ] U6 由 U5 组成 Deep Ensemble,分解偶然/认知不确定性

U1–U4 只读已保存的预测,不需要 GPU。U3、U4 取代原计划的 E3、E4 中对应部分。

## 实验矩阵

| ID | 内容 | 运行数 | 估计 GPU 时间 |
|----|------|--------|----------------|
| E1 | baseline,dropout=0,5 seeds | 5 | ~4 h(300 epoch 上限) |
| E2 | dropout 0.1/0.2/0.3,5 seeds,MC=50 | 15 | ~12 h |
| E3 | 对 E1/E2 已保存预测做离线分析 | 0 | CPU |
| E4 | 对已保存预测做离线回测 | 0 | CPU |

种子:20260813, 20260814, 20260815, 20260816, 20260817。

## 指标

- 预测:IC、ICIR、RankIC、RankICIR(逐日横截面),RMSE
- 显著性:逐日 IC 序列的配对检验(模型 vs 基线),Newey-West t 值
- 不确定性:Spearman(sigma, |误差|)、按 sigma 十分位的 RMSE、PICP、MPIW、Interval Score
- 组合:Top-10 / Top-50 等权多头,年化收益、波动、Sharpe、最大回撤、换手率;成本 0 / 10 / 20 bps

## E4 排序规则(lambda 与过滤比例只在验证集上选)

1. `score = mu`
2. `score = mu - lambda * sigma`
3. `score = mu / (sigma + eps)`
4. 剔除 sigma 最高的 10% / 20% 后按 mu 排序

## 需要新增的代码(待确认后实现)

- `research/run_experiment.py`:加 `--select-metric`(mse / rank_ic)
- `research/ensemble.py`:多个已保存预测求均值与方差
- `research/calibration.py`:PICP/MPIW/conformal
- `research/backtest.py`:Top-K 组合、成本、换手
- `research/aggregate.py`:多种子汇总表
- `tests/`:metrics、mc_dropout、backtest 的单元测试

## Key Questions

1. 用哪台 GPU 机器?(gpuvm 手动运行,还是 Slurm a30 分区)
2. 截止日期?
3. 服务器访问权限何时失效?

## Decisions Made

- 不引用任何旧日志数字:评估实现有误。
- 不再扩充模型变体(`model_bayes_*` 共 8 个):先把评估和回测做完整。

## Risks

- 结果可能显示 Dropout 对 IC 没有显著提升。此时简历改写为不确定性过滤对回撤/换手的实际效果,按真实结果写。
- 测试期只有 237 天,Sharpe 置信区间宽。
- NASDAQ 数据是 2013–2017 年的公开基准,非实盘数据,无行业/市值信息,做不了中性化。

## Errors Encountered

- E2(2026-09-29)结果不能作为 MC Dropout 的有效检验。15 个运行在第 31–39 个 epoch 提前停止,最优 epoch 为 1–9。
  现象:带 dropout 的模型在验证集(eval 模式)上的 MSE 随训练上升(dropout 0.2, seed 20260815:0.0030 → 0.0219),
  测试集 mc_std 均值 0.087,而 |真实收益| 均值只有 0.012。
  假设(未验证):dropout 直接作用在输出价格水平的最后一层线性层的输入上,而收益率 = (预测价 − 基准价)/基准价,
  价格水平上的小扰动被放大成收益率上的大误差。
  待做:只在 Mixer 块内保留 dropout,去掉输出层前的 dropout,重跑对比。
- 基线的验证 IC 随 epoch 大幅波动(seed 20260815:第 17 epoch 0.0307,第 65 epoch −0.0003),而验证 MSE 基本持平。
  按 MSE 选模型与 IC 几乎无关。待做:加 `--select-metric rank_ic`,或对最后若干 epoch 取平均。

- `/vol/bitbucket/hw2025/env` 中 torch 无法导入(2026-09-29),原因未查。

## Status

**NASDAQ 第一轮完成;A 股基线训练中(gpu32)** — 待用户决定模型选择规则和叙事方向。

## 2026-09-30 更新：不确定性研究 A1 开发诊断

本节为最新补充；以上历史状态保留，不代表当前仍有相同训练任务运行。

- [x] 用户确认研究主线仍是“不确定性对于股票收益/排序预测的作用”，不以交易
  平滑、退出缓冲或单独提升覆盖率作为主要研究贡献。
- [x] 固定 A1 协议 `protocol_rank_uncertainty_a1_20260930.md`；只使用 NASDAQ E3
  dropout=0.1 五种子的验证输出，不打开测试预测，不重新训练基础模型。
- [x] 新增 `research/rank_uncertainty.py` 和 12 项测试；全套 43 项测试通过。
- [x] 完成 156 天顺序交叉拟合评估：四个不确定性增量版本的主要 MSE 差值
  95% 区间均包含零。只按排名位置构造的基线已与排名错误有 0.231933 的相关性。
- [x] 完整报告：`results_rank_uncertainty_a1_20260930.md`。加入排名分歧的分层
  筛选 RankIC 有小幅正向辅助信号，但上游 checkpoint 曾使用整个验证期选择，
  因而不能声称独立样本外提升，也不据此挑选测试集方案。
- [ ] A2：冻结新的外层时间边界和预算后，重新生成基础模型的顺序预测并独立
  确认；保留所有增量对照，增加伪特征对照。尚未运行。
- [ ] B 组：先处理异方差输出单位、损失相对尺度和 MC 总方差估计口径，再训练。
  本轮未开始 B 组，未修改任何交易规则。
