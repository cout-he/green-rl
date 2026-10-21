# GAS-MARL 候选问题的近邻文献核查（阶段笔记）

核查日期：2026-10-21。范围：约五分钟的定向检索；以 `research/PROTOCOL.md` 和 `reports/拆解报告_GAS_MARL.md` 的候选问题为边界。仅使用作者预印本平台、官方 arXiv 和出版商来源作为论据。本笔记不构成全面综述、查新通过或最终方向决定；未运行实验，也未改变训练代码、配置或协议。

## 证据可读性

“全文可读”表示本次确实取得正文并定位到相关段落，不表示已逐式审计。“摘要可读”包含搜索服务返回的原始出版商/预印本页面摘要；网页直开失败时单独注明。“仅元数据”不支持任何方法细节判断。搜索结果的相对发布日期、第三方转载摘要和自动生成总结不用于判断。

| 工作 | 本次实际可读范围 | 对本项目的关联 |
| --- | --- | --- |
| Shan et al., *Job Energy Prediction-Driven Green Scheduling for Heterogeneous HPC Workloads via Attention-Enhanced Dual-Critic Reinforcement Learning* (SSRN, 2026) | 官方 SSRN 摘要及元数据经搜索索引可读；直开页面和平台 Delivery 链接均 403，未读全文 | 直接以 GAS-MARL-ADC 为调度器，是必须优先排查的近邻 |
| Gan et al., *Energy-aware job scheduling for green data centers under uncertainty: A structure-compressed spectral risk control approach* (Applied Energy 420, 128137, 2026) | 官方 ScienceDirect 摘要、highlights 和部分 introduction/conclusion 摘录经索引可读；页面直开 403，未取得完整方法和实验 | 不确定性与风险控制的直接邻近问题 |
| Sahai, *Renewable Energy Prediction using Weather Forecasts for Optimal Scheduling in HPC Systems* (arXiv:1402.6552, 2014) | 官方 arXiv HTML 正文可读，核查 §4–6；PDF 提取失败 | HPC 绿电预测与调度的早期直接关联 |
| *RLScheduler: An Automated HPC Batch Job Scheduler Using Reinforcement Learning* (arXiv:1910.08925v3, 2020 修订版) | 官方 arXiv PDF 14 页可读，重点核查 §V-C、V-E、V-F 和表 VII | 已经研究跨作业轨迹应用及用户公平性 |
| Ding et al., *Data Center Job Scheduling and Energy Management Under Uncertain Environments* (IEEE TIA, 2025) | 本次只确认出版商参考文献条目及 DOI 到 IEEE 文献页的映射；IEEE 页面要求机器人验证，未从原始来源读到摘要或正文 | 待取得原始文本后核查的风险调度近邻 |

## 已核实内容与不能据此推出的结论

### 1. GAS-MARL-ADC（最优先补全文）

来源：[SSRN 原始页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6530554)，[DOI](https://doi.org/10.2139/ssrn.6530554)。页面索引标为 18 页，Posted 6 Apr 2026；其预印本身份不等同于已同行评审。

摘要明确涉及作业能耗预测、稀疏/冷启动用户，以及将预测能耗作为 GAS-MARL-ADC 的先验输入；也明确提到状态注意力、dual-critic 和 Marconi100/ERA5 数据。这里预测的对象首先是**作业能耗**，不能仅凭摘要将它等同于**可再生供给预测误差**。然而它直接扩展 GAS-MARL，且涉及预测与策略稳定性，因此“给编码器加注意力”“改双 critic”“加入预测”均不能作为未经核查的独立创新。

待查：正文是否扰动绿电预测、如何区分预测值与真实供给、是否有偏差/时间错位实验；dual-critic 是何种目标与估计结构；延迟动作和绿色回填是否保留；是否涉及等待分位数与跨负载测试。未读全文前无法判定这些精确重叠。

### 2. SCS-RC（风险控制近邻）

来源：[ScienceDirect 原始页面](https://www.sciencedirect.com/science/article/pii/S0306261926007890)，[摘要页面](https://www.sciencedirect.com/science/article/abs/pii/S0306261926007890)，[DOI](https://doi.org/10.1016/j.apenergy.2026.128137)。出版商索引显示卷 420、1 October 2026、128137；不采用其他站点相互矛盾的日期。

摘要明确提到有依赖的异构任务、任务图压缩、自适应分位数回报分布及谱风险聚合；公开摘录强调电价、温度和到达不确定性。由此可确认“数据中心调度＋分布式回报估计＋尾部风险”已有直接近邻。**回报尾部风险不自动等同于逐作业等待 P95/P99 约束**；目前不能确认它是否检验供给预测误差，亦不能声称它没有做。

待查：随机变量及可观测信息的精确定义；风险施加于收益还是延迟；是否显式处理绿电预测残差；动作是否有主动等待/回填；风险参数的退化关系及评测划分。本次少量标题/作者检索未找到可读作者全文；这不证明公开版本不存在。

### 3. 风能预测与 HPC 调度（已有的问题动机）

来源：[arXiv 摘要与版本](https://arxiv.org/abs/1402.6552)，[官方正文 HTML](https://arxiv.org/html/1402.6552v1)。

正文 §4 给出天气属性与风能的回归关系；§5 提及 greedy、randomized-greedy 与 brute force 的调度比较，并提出按当前天气选择预计算调度的决策树思路。§6 将预测风能与未来绿色调度连接。摘要还提出根据实时天气及预测偏离更新调度。因此“考虑绿电预测不准有价值”本身早有动机。此次读到的内容不足以证明它提供了与 GAS-MARL 一致的误差压力测试、在线学习策略或等待尾部保证。

待查：若正式比较，应取得可复核的调度实验定义；不能把提出的决策树思路写成已验证的稳健 RL 方法。

### 4. RLScheduler（泛化与公平性必须对照）

来源：[arXiv 版本页](https://arxiv.org/abs/1910.08925)，[官方 PDF](https://arxiv.org/pdf/1910.08925)。

正文 §V-E、表 VII（PDF 第 10 页）确实将轨迹 X 训练得到的 RL-X 应用于其他轨迹 Y，并按配对进行随机序列评估；不能把“不同负载训练的策略用于未见负载”作为首次提出的问题。§V-F 改用各用户平均 slowdown 的聚合目标，其中 Maximal 聚合关注最差用户平均值；还说明可通过动作掩码处理用户配额。这和逐作业等待分位数不是同一个量，但意味着“加入公平/避免饥饿”也不足以单独定义新贡献。

待查：后续若保留跨负载候选，必须对照绿电供给分布、主动延迟动作和归一化导致的迁移失效；若保留尾部候选，应明确逐作业尾部、用户聚合公平、硬约束与回报风险的区别。本文没有给本项目的绿电鲁棒性提供现成结论。

### 5. IEEE 2025 不确定环境调度（只列待查）

来源：[DOI](https://doi.org/10.1109/TIA.2025.3548576)，其跳转至 [IEEE 文献页 10916511](https://ieeexplore.ieee.org/document/10916511/)。SCS-RC 出版商参考文献也列出该题目及 IEEE TIA/2025。

由于本次原始页面停在机器人验证，未采用第三方平台的摘要来补齐方法，不在这里声称其具体风险函数、网络或不确定性模型。它是需要补充原始摘要/作者版本的检索线索，不能据此确认或排除候选重叠。

## 候选问题的阶段判断

- **绿电预测误差下 HPC 多动作调度稳健性：继续作为待检验候选。** 最值得核查的差异不是网络名称，而是仅扰动调度器可见供给预测、真实供给积分不变时，作业选择、主动等待、回填接受分别如何失效。现有证据尚不能排除两篇 2026 近邻已覆盖该问题。
- **等待尾部约束：需要进一步收窄。** 逐作业尾部服务约束应与谱风险回报、用户平均 slowdown 公平性区别清楚；当前未证明该约束在近邻中缺失。
- **跨负载泛化：宽泛表述已有明确先例。** 若以后推进，应围绕绿色调度新增信息与供给/负载联合分布变化提出可验证问题；仅换负载或增加零样本测试不能作为充分贡献。

下一步文献动作是补读前两项全文和第五项原始摘要/全文，再形成按信息、动作、约束、误差模型、评价划分逐项对照的表。当前正式基线尚在运行，本笔记不给出“新颖性成立”“创新有效”或最终研究方向结论。
