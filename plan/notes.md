# Notes: 不确定性实验方向调研

更新日期:2026-09-29

证据标签:
- **[原文]** 调研时直接读了 PDF 正文或表格
- **[页面摘要]** 打开了页面,数字由抓取工具摘要得到;引用前须对照 PDF 复核
- **[仅摘要]** 只看到摘要或落地页

文献信息(标题、作者、会议、年份)均已打开原始页面核实。标为"判断"的是评估,不是论文事实。

## 一、金融方向:不确定性在股票预测中的用法

### A. 与本项目任务最接近(横截面日频排序)

| 论文 | 不确定性来源 | 怎么用 | 数据 | 代码 | 主要结果 |
|---|---|---|---|---|---|
| FactorVAE. Duan, Wang, Zhang, Li. AAAI 2022, 36(4):4468-4476. https://ojs.aaai.org/index.php/AAAI/article/view/20369 | 预测收益为高斯分布,方差来自 VAE 隐因子 [原文] | 选股得分 `mu - eta*sigma`(TDrisk),k=50,n=5 [原文] | A 股日频;Alpha158 中选 20 个特征,T=20 [原文] | 无官方代码;非官方 github.com/x7jeon8gi/FactorVAE、github.com/leejoonhun/factor-vae | RankIC 0.055,RankICIR 0.568;TDrisk 年化 16.32%、Sharpe 2.09,对比不加风险项 15.32%、1.92 [原文] |
| Yang, Fang, Zhang, Zhang, Zhou. Enhancing stock ranking forecasting by modeling returns with heteroscedastic Gaussian Distribution. Physica A 664, 2025. DOI 10.1016/j.physa.2025.130442 | 网络同时输出均值和标准差,极大似然训练 [仅摘要] | 改善排序;Top-20 组合评估 [仅摘要] | CSI 100/300/500;特征集未核实(全文付费墙) | 未找到 | Top-20 年化收益提升 2%、20%、50% [仅摘要] |
| Sanderink. When Alpha Breaks. arXiv:2603.13252, 2026(单作者预印本) | 二级模型预测排名位移,扣除偶然不确定性下限;另有 split conformal [页面摘要] | 策略级开关(是否交易)、波动率定仓、对最不确定的尾部限权 | 美股约 100 只,2016–2025;OHLCV 派生特征 + VIX 分位 | 论文给的 GitHub 链接 404 | AUROC 0.72;47% 弃权时精度 80%;不确定性与 \|score\| 相关 0.616,因此按不确定性反比定仓会削掉最强信号(摘要原文) |
| AlphaMix. Sun, Wang, An. arXiv:2207.07578, 2022(预印本) | 多专家集成的分歧 [原文] | 只在分类和回归符号一致的"确定"集合里买 top-k [原文] | ACL18(87 只美股)、SZ50(47 只);OHLC 派生 11 个特征 | 未找到 | ACL18 Sharpe 2.53;加不确定性损失后 Sharpe 2.08 → 2.42 [原文] |

### B. 最接近的先例,但数据不可得

| 论文 | 要点 | 为什么不能复现 |
|---|---|---|
| Chauhan, Alberg, Lipton. Uncertainty-Aware Lookahead Factor Models for Quantitative Investing. ICML 2020, PMLR 119:1489-1499. https://proceedings.mlr.press/v119/chauhan20a.html | 异方差回归 + dropout 启发式,用不确定性折减预测。年化 17.7% vs 14.0%,Sharpe 0.84 vs 0.52(摘要原文) | Compustat 基本面,专有数据 |
| Liao, Ma, Neuhierl, Schilling. The Uncertainty of Machine Learning Predictions in Asset Pricing. arXiv:2503.00549, 2025 | 预测标准误 + 不交易区间(\|预测\| 小于置信界则权重为 0) | 123 个公司特征,月频 |
| Filipović, Pasricha. Empirical Asset Pricing via Ensemble Gaussian Process Regression. arXiv:2212.01048 | 按预测不确定性加权的均值方差组合 | 94 个公司特征 + 宏观 |
| Barunik, Hronec, Tobek. arXiv:2408.07497, 2024 | 37 个分位数的分位数网络 | 194 个特征含基本面 |
| Zhang, Zohren, Roberts. BDLOB. NeurIPS 2018 Bayesian DL Workshop. arXiv:1811.10041 | MC Dropout 用于定仓 | 限价订单簿数据 |

### C. 其他已核实但相关性较弱

- D-Va(Koa, Ma, Ng, Chua. CIKM 2023. arXiv:2309.00073):扩散 + VAE,多步预测,逐股而非横截面。官方代码 github.com/koa-fin/dva。
- Wang 等 arXiv:2503.06929:9 分量高斯混合,需要 5 分钟数据算已实现波动。
- Gao 等 arXiv:2509.22088、Diffolio arXiv:2511.07014:扩散模型 + 组合优化,无代码或需宏观变量。
- Kato. Conformal Predictive Portfolio Selection. arXiv:2410.16333:6 只股票月频,规模很小。
- Kaya, Nguyen. PMLR 266, 2025:conformal 预测集,3 页摘要,随机切分存在时间泄漏(调研时的观察)。
- Chalkidis, Savani. Trading via Selective Classification. ICAIF 2021. arXiv:2110.14914:日内商品期货,准确率-覆盖率评估方式可借用。
- FactorVQVAE(Kim, Ock, Song. Knowledge-Based Systems 318, 2025):FactorVAE 后续,官方代码 github.com/finxlab/FactorVQVAE,但不是不确定性方法。

### D. 未能核实,已排除

- "Portfolio Selection with Adaptive Conformal Prediction"(Springer 章节,页面跳转登录)。
- HireVAE(IJCAI 2023)、RAGIC(arXiv:2402.10760)、FinCLASS(UAI 2021):文献信息无误,但与本任务不符或未打开页面。

### 其他事实

- Qlib 模型库中没有输出风险或不确定性估计的模型(查阅 Qlib README)。

## 二、方法方向:MC Dropout 之外的不确定性方法

文献信息均已打开原始页面核实。"失效风险"一列是针对本任务(IC 0.02–0.04、厚尾、非平稳)的判断,不是论文结论。

| 方法 | 核实的参考文献 | 是否重训 | 成本 | 给出哪种不确定性 | 在本任务上的失效风险(判断) |
|---|---|---|---|---|---|
| Deep Ensemble | Lakshminarayanan, Pritzel, Blundell. NIPS 2017. arXiv:1612.01474 | 是 | M 倍 | 偶然 + 认知 | 认知部分远小于噪声 |
| 异方差高斯 NLL | Nix, Weigend. ICNN 1994. DOI 10.1109/ICNN.1994.374138;Kendall, Gal. NIPS 2017. arXiv:1703.04977 | 是 | 1 倍 | 偶然 | 均值头塌缩,IC 下降;高斯尾部 |
| beta-NLL | Seitzer, Tavakoli, Antic, Martius. ICLR 2022. arXiv:2203.09168。官方代码 github.com/martius-lab/beta-nll | 是 | 1 倍 | 偶然 | 同上,beta=0.5 缓解 |
| 保均值的异方差回归 | Stirn 等. AISTATS 2023, PMLR 206:5593-5613 | 是 | 1 倍 | 偶然 | — |
| 分位数回归 | Koenker, Bassett. Econometrica 46(1):33-50, 1978;Tagasovska, Lopez-Paz. NeurIPS 2019. arXiv:1811.00908 | 是 | 1 倍 | 偶然 | 分位数交叉;极端分位不稳;点预测变成中位数 |
| Deep Evidential Regression | Amini, Schwarting, Soleimany, Rus. NeurIPS 2020. arXiv:1910.02600 | 是 | 1 倍 | 有争议 | 正则系数直接决定输出的不确定性 |
| 对 DER 的批评 | Meinert, Gawlikowski, Lavin. AAAI 2023. arXiv:2205.10060;Jürgens 等. ICML 2024. arXiv:2402.09056;Shen 等. NeurIPS 2024. arXiv:2402.06160 | — | — | — | — |
| SWAG | Maddox 等. NeurIPS 2019. arXiv:1902.02476 | 改训练日程 | 约 1 倍 | 认知 | 针对 SGD 设计,调参多 |
| Last-layer Laplace | Daxberger 等. NeurIPS 2021. arXiv:2106.14806。库 github.com/aleximmer/Laplace | 否 | 一次数据遍历 | 认知 | 同方差似然;输出形状需适配 |
| Split conformal | Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. JASA 113(523):1094-1111, 2018 | 否 | 可忽略 | 总体 | 可交换性不成立 |
| CQR | Romano, Patterson, Candès. NeurIPS 2019. arXiv:1905.03222。官方代码 github.com/yromano/cqr | 否(需分位数头) | 可忽略 | 总体 | 同上 |
| ACI | Gibbs, Candès. NeurIPS 2021. arXiv:2106.00170 | 否 | 可忽略 | 总体 | 只保证长期平均覆盖率;区间滞后于波动跳变 |
| Conformal PID | Angelopoulos, Candès, Tibshirani. NeurIPS 2023. arXiv:2307.16895。官方代码 github.com/aangelopoulos/conformal-time-series | 否 | 可忽略 | 总体 | 同上 |
| EnbPI | Xu, Xie. ICML 2021, PMLR 139:11559-11569 | 需要 bootstrap 集成 | M 倍 | 总体 | — |
| 回归校准 | Kuleshov, Fenner, Ermon. ICML 2018. arXiv:1807.00263 | 否 | 可忽略 | — | 只修正平均校准;制度变化后失效 |
| Student-t 输出 | Detlefsen, Jørgensen, Hauberg. NeurIPS 2019. arXiv:1906.03260 | 是 | 1 倍 | 偶然 | 自由度和尺度难以同时识别 |

### 评估指标及其来源

| 指标 | 来源 |
|---|---|
| PICP、MPIW | Pearce, Zaki, Brintrup, Neely. ICML 2018. arXiv:1802.07167 |
| Interval score、CRPS、对数得分 | Gneiting, Raftery. JASA 102:359-378, 2007 |
| 校准曲线、校准误差 | Kuleshov 等. ICML 2018 |
| ENCE(按 sigma 分箱的校准误差) | Levi, Gispan, Giladi, Fetaya. arXiv:1905.11659(发表会议未核实) |
| PIT 直方图 | Gneiting, Balabdaoui, Raftery. JRSS-B 69(2):243-268, 2007 |
| Risk-coverage 曲线、AURC | Geifman, Uziel, El-Yaniv. ICLR 2019. arXiv:1805.08206 |
| Sparsification error | Ilg 等. ECCV 2018. arXiv:1802.07095 |

### 最重要的评估陷阱(判断)

当 R² 接近 0 时,预测误差约等于收益率本身。此时"预测误差大小"就是"预测波动率",而波动率本来就能从过去 16 天的数据里预测出来。后果:

- sigma 与 |误差| 的相关性、risk-coverage 曲线、sparsification 曲线都会显得很好,但这只说明 sigma 学到了波动率,不说明模型知道自己哪里预测得准。
- 边际 PICP 用无条件分布的固定区间就能达标。
- MPIW 由噪声水平决定,不反映模型质量。

对策:
1. 所有指标都要和**朴素波动率基线**比较(mu = 0,sigma = 过去 16 天已实现波动率)。
2. 用 **IC 对覆盖率的曲线**代替误差对覆盖率的曲线:按不确定性排序,保留最确定的 100% / 80% / 60% ... 样本,看 RankIC 是否上升。
3. 置信区间按**交易日**重抽样,不按"股票-日"重抽样。同一天的 1000 只股票不是 1000 个独立样本。

### 未核实,不得引用

- Springer 2026 "PNNs with t-distributed outputs"(页面跳转登录)。
- Lange 等 1989、Takahashi 等 IJCAI 2018(只在他人参考文献里见到)。
- 以下只核实了 arXiv 页面,发表会议未核实:arXiv 2203.06102、1904.06019、1705.08500、1905.11659、2005.10036、1910.03127、2212.03463、2208.08401、2107.07511、2109.10254。

## 三、综合判断

### 两份调研互相印证的三点

1. MC Dropout 只给认知不确定性,而本任务里绝大部分不确定性是偶然的(噪声)。需要补一个异方差输出。
   金融方向的先例(Chauhan 等 ICML 2020;Yang 等 2025)用的正是这个组合。
2. 不确定性很可能主要反映波动率。Sanderink 报告不确定性与 |score| 相关 0.616;方法调研从原理上给出同样的预期。
   必须加波动率基线,否则结论站不住。
3. FactorVAE 的 `mu - eta*sigma` 是与本任务直接可比的、已发表的用法。

### 选定的实验(按顺序)

| 编号 | 实验 | 依据 | 是否需要 GPU |
|---|---|---|---|
| U1 | 评估框架 + 波动率基线 + IC-覆盖率曲线 | 方法调研的评估部分 | 否 |
| U2 | 检验 sigma 与 \|mu\|、与已实现波动率的相关性 | Sanderink 2026 的反面结论 | 否 |
| U3 | `mu - eta*sigma` 选股(TDrisk),sigma 分别取 MC Dropout、集成、波动率基线 | FactorVAE, AAAI 2022 | 否 |
| U4 | 滚动 conformal + ACI,得到有覆盖率保证的区间 | Gibbs, Candès, NeurIPS 2021 | 否 |
| U5 | 异方差输出头,beta-NLL(beta=0.5),5 个种子 | Seitzer 等 ICLR 2022 | 是 |
| U6 | U5 的 5 个模型组成 Deep Ensemble,分解偶然和认知不确定性 | Lakshminarayanan 等 NIPS 2017 | 复用 U5 |

### 明确不做

- Deep Evidential Regression:三篇独立的已发表批评。
- SWAG:对一小时内能训完的模型,调参成本高于 5 个种子的集成。
- 扩散模型、资产定价类论文的完整复现:无代码或需要基本面数据。
- FactorVAE 模型本身的完整复现:无官方代码,特征需要成交均价。只借用其选股规则。

## 四、精读:Gal & Ghahramani, "Dropout as a Bayesian Approximation"(ICML 2016, arXiv:1506.02142;附录 arXiv:1506.02157)

2026-09-29 精读。标 [原文] 的内容来自论文正文或官方代码,附位置;标 [判断] 的是推论。

### 论文的预测方差有两项,我们只用了一项

- [原文] 似然(式 2):`p(y|x,w) = N(y; ŷ(x,w), τ⁻¹ I)`,τ 是同方差观测噪声的精度。
- [原文] 预测方差(第 4 节,式 6 与式 7 之间):`Var[y*] ≈ τ⁻¹ I + T 次随机前向的样本方差`。
- [原文] 式 7:`τ = p l² / (2 N λ)`,p 是保留概率,l 是先验长度尺度,λ 是权重衰减,N 是样本数。
- [原文] 第 5.3 节和官方代码 `experiment.py`:τ 和 dropout 比率在验证集上按对数似然做网格搜索。
- [判断] 我们没有权重衰减(λ = 0),式 7 给不出有限的 τ;τ 只能当超参数在验证集上选。
- [判断] 全零预测的 MSE 为 0.000378,对应 τ⁻¹ ≈ 3.8e-4,噪声标准差约 0.019,大于 MC 标准差 0.003–0.015。
  这一项的缺失足以解释 90% 区间只覆盖 17%–59%。

### 记号陷阱

- [原文] 论文里的 p 是"保留"概率;代码里的 dropout 是"丢弃"比率。我们的 0.1/0.2/0.3 对应论文的 p = 0.9/0.8/0.7。

### MC 平均是否提高精度

- [原文] 第 5.3 节只有一句话称 MC 估计优于权重平均,没有给出任何对比数字或表格。
- [原文] 给出数字的是"有 dropout 对无 dropout":Boston Housing 上 dropout 概率为 0 时 RMSE 3.07、LL −2.59;表 1 中 Dropout 为 2.97 ± 0.19、−2.46 ± 0.06。
- [原文] 表 1 比较的是 VI、PBP、Dropout 三种方法,不是 MC 对权重平均。
- [判断] 我们观察到的"单次预测与 MC 均值 MSE 几乎相同"与论文不矛盾。不应引用该论文作为"MC 平均提升精度"的依据。

### 不确定性的评估方式

- [原文] 主要指标是测试集预测对数似然(式 8)。论文没有报告区间覆盖率、校准曲线或 sigma 与误差的相关性。
- [原文] 附录 5.1.1:"We can show that the dropout model is not calibrated",建议对不确定性做线性缩放。
- [原文] 附录 E.3:MC dropout 过度自信,变分推断会低估模型不确定性。

### Dropout 的位置

- [原文] 第 3 节:等价关系要求 dropout 加在每个权重层之前。
- [原文] 附录 5.2:只在部分层前加 dropout,相当于把 MAP 估计和贝叶斯估计交错使用。
- [原文] 论文自己的 MNIST 实验(第 5.2 节)也只在最后一个全连接层前加 dropout。
- [判断] 我们只在 Mixer 块内加 dropout,MC 方差只反映这部分权重的不确定性。

### 相关批评与后续(均已核实原文)

- Osband 2016(NeurIPS BDL workshop 摘要):dropout 采样的方差不随数据量增加而收缩,更像是对风险的近似,而不是模型不确定性。
- Gal, Hron, Kendall, Concrete Dropout(NeurIPS 2017, arXiv:1705.07832):要得到校准好的不确定性,必须对 dropout 概率做网格搜索;数据量趋于无穷时 dropout 概率应趋于 0。
- Kendall & Gal(NIPS 2017, arXiv:1703.04977)式 9:总方差 = MC 样本方差(认知)+ 预测噪声方差的均值(偶然)。

### 由此得到的实验(前三项不需要重新训练)

| 编号 | 做法 | 指标 | 什么结果有信息量 |
|---|---|---|---|
| T1 | 事后补上 τ⁻¹:在验证集上按对数似然选 τ | 测试对数似然;用 sqrt(1/τ + MC 方差) 的 90% 覆盖率 | 覆盖率接近 90% 说明校准差主要是缺了这一项 |
| T2 | 对数似然基线:零均值常数方差;方差取 16 日波动率 | 同上 | 波动率基线若优于 T1,说明 MC sigma 没有提供波动率之外的信息 |
| T3 | 对 MC sigma 做乘法缩放(附录 5.1.1) | 同上 | 乘法缩放若与加法 τ⁻¹ 一样好,说明 MC sigma 只是尺度不对的波动率代理 |
| T4 | 数据量实验:用 25% / 50% / 100% 的训练期训练 | 同一测试集上的 MC sigma 均值 | sigma 不随数据量下降,支持 Osband 的解读 |
| T5 | dropout 加在每个权重层前 + 按式 7 的权重衰减 | RankIC、MSE、偏秩相关 | 与 head dropout 的失败经验冲突,需谨慎 |
| T6 | 异方差输出头 | 即 X6 | — |

## 五、精读:单股实验对应的论文与本地 notebook

### 已核实的文献

Chandra R, He Y. Bayesian neural networks for stock price forecasting before and during COVID-19 pandemic. PLoS ONE. 2021;16(7):e0253217.
https://doi.org/10.1371/journal.pone.0253217 (论文的 Data Availability 指向本地克隆的同一个 GitHub 仓库。arXiv 版本未找到。)

### 论文事实 [原文]

- 数据:MMM、600118.SS、CBA.AX、DAI.DE,Yahoo Finance 收盘价,2012-01-01 至 2020-07-01;min-max 缩放到 [0,1](4.1 节)。
- 设定:5 个输入、5 个输出的直接多步预测;单隐层 5-5-5,隐层和输出层均为 sigmoid(3.2、4.2 节)。
- 方法:Langevin 梯度并行回火 MCMC,10 个副本,100,000 个样本(4.2 节)。基线只有 FNN-Adam、FNN-SGD。
- 指标:只有逐步 RMSE,30 次运行的均值和 95% 区间。
- 不确定性只以阴影图展示,没有覆盖率、区间宽度、对数似然或校准曲线。
- 没有朴素基线("明天价格 = 今天价格")。
- 新冠部分:Setup 1 与 Setup 2 的测试期不同,不是受控比较(论文 4.4 节自己承认)。

### 本地 notebook 的事实(读取代码和输出核实)

- 模型:MLP 5→10→5,ReLU,线性输出;dropout 加在每个线性层前,包括 5 个输入上。
- 权重衰减:`reg` 被计算并传入模型,但没有进入损失或优化器,实际无效。
- 未设随机种子;训练中每 20 个 epoch 打印测试 RMSE,dropout 比率和 epoch 数是看着测试集选的,没有验证集。
- MC 采样 10,000 次;不确定性带取 MC 样本的 5%–95% 分位,不含观测噪声项。

### `compare.csv` 的问题

- `*_bayes` 列不是论文表 2,而是仓库里附带的单次运行输出(`code/results/*/ptmcmc_indi_rmse.txt`),数值到小数点后 6 位一致。
- 与论文表 2 不可比:本地数据是 2015–2019 年,论文是 2012–2020 年;min-max 范围不同;本地缩放用到了测试期的最大最小值。
- CBA.AX 上 0.158 对 0.045 的差距来自外推失败:训练期缩放后目标最大 0.759,测试期到 1.000;
  仓库模型输出层是 sigmoid,预测值从未超过 0.726,平均偏差 −0.128 至 −0.166。与后验近似的质量无关。
- MMM 的 0.02702 在任何 notebook 输出中都找不到;notebook 里 p=0.1 的两次运行是 0.0417 和 0.0368。

### 朴素基线(只读计算,缩放后的 RMSE)

| 股票 | 步长 | 朴素基线 | 仓库 Bayes | MC Dropout p=0.1 | 普通 MLP p=0 |
|---|---|---|---|---|---|
| MMM | 1 | 0.02368 | 0.03090 | 0.02702 | 0.02429 |
| MMM | 5 | 0.05349 | 0.05419 | 0.05456 | 0.05399 |
| CBA.AX | 1 | 0.02876 | 0.15817 | 0.04456 | 0.05771 |
| CBA.AX | 5 | 0.06667 | 0.19813 | 0.08354 | 0.09296 |
| DAI.DE | 1 | 0.02283 | 0.04343 | 0.04154 | 0.02349 |
| DAI.DE | 5 | 0.05962 | 0.06114 | 0.06530 | 0.05858 |
| 600118.SS | 1 | 0.00811 | 0.01888 | 0.01112 | 0.00840 |
| 600118.SS | 5 | 0.01740 | 0.01966 | 0.01858 | 0.01782 |

- 仓库 Bayes 运行:20 个"股票 × 步长"格子全部不如朴素基线。
- MC Dropout p=0.1:20 个格子全部不如朴素基线(比值 1.02–1.82)。
- 普通 MLP(p=0):与朴素基线基本持平;DAI.DE 步长 2–5 上好 0.5%–1.7%,小于运行间波动。
- 四只股票上 dropout 都是单调变差:p=0 最好。

### 可以带到横截面项目的想法 [判断]

1. 危机前/中/后对比,但用固定的评估规则和 walk-forward(A 股样本含 2015 年股灾、2020 年初、2024 年初小盘股下跌)。
2. 多步长目标(1、5、10、20 日),看 IC、换手后的收益和 sigma 是否随步长增大。
3. 在小问题上比较 MCMC、MC Dropout、深度集成三种后验(例如冻结 StockMixer 特征后只对最后一层做)。
4. 分布外检验:测试输入超出训练范围时 sigma 是否升高。
