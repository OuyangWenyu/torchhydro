# 实验审计报告

> 审计时间: 2026-06-25

## 1. 总体结论

**部分符合。** 核心修复（edge direction, route_steps=4）已应用，主要实验运行完毕。存在以下不一致：

| 优先级 | 问题 | 影响 |
|:---:|------|------|
| P1 | all_events_metrics.csv 为修复前旧数据 | 逐事件指标表过期 |
| P1 | 实验4 空间敏感性未完成 | 缺少4张空间敏感性图 |
| P2 | 实验3 单流域结果未用新配置重跑 | 数值偏旧 |
| P2 | Basin LSTM 未与 GNN 模型统一时间范围重跑 | 但不影响（无图） |
| P2 | REPORT 中洪水事件图未分 train/test | 当前全部混在 flood_events/ |

## 2. 数据与变量检查表

| 项目 | 设计要求 | 当前实际 | 一致 |
|------|------|------|:---:|
| 研究区域 | 松辽6 basin | songliao_20800900~21401550 | ✅ |
| 时间分辨率 | 3h | min_time_interval=3, min_time_unit="h" | ✅ |
| 数据时段 | 1980-2024 | train:1980-2014, valid:2015-2018, test:2019-2024 | ✅ |
| 输入变量 | DRP (站点降雨) | station_cols=["DRP"], var_t=["rain"] | ✅ |
| 输出变量 | inflow | var_out=["inflow","flood_event"] | ✅ |
| 无历史径流 | 是 | var_t 不含 inflow, use_basin_features=False | ✅ |
| 无气象预报 | 是 | 用 var_t=["rain"] 真实观测, 非预报 | ✅ |
| Train/Valid/Test | 一致 | 所有模型相同划分 | ✅ |

## 3. 实验一致性表

| 实验 | 设计要求 | 当前实现 | 一致 | 问题 |
|------|------|------|:---:|------|
| 实验1 | 6 basin, DRP only | ✅ 6 basin, DRP only | ✅ | — |
| 实验1 | Basin LSTM | ✅ 存在 | ✅ | 使用旧FloodEventDataset,无图,不受edge fix影响 |
| 实验1 | LSTM-only | ✅ 已重跑 | ✅ | — |
| 实验1 | LSTM+GNN | ✅ 已重跑,edge fix | ✅ | — |
| 实验1 | WDNE steps=4 | ✅ 已重跑,edge fix | ✅ | — |
| 实验2 | 5 basin, +静态属性 | ✅ _static后缀自动切5 basin | ✅ | 21100150已排除 |
| 实验2 | 静态字段 | ele_mt_sav,slp_dg_sav,for_pc_sse,pre_mm_syr,run_mm_syr | ✅ | 不使用area(极端值) |
| 实验2 | 所有模型同条件 | ✅ LSTM-only/LSTM+GNN/WDNE 同 static | ✅ | — |
| 实验3 | 21401550单流域 | ⚠️ 有结果但未用steps=4重跑 | ❌ | P2 |
| 实验4 | 空间敏感性 | ❌ NC文件未生成 | ❌ | P1 |

## 4. 模型配置检查表

| 模型 | 图结构 | 输入 | 静态属性 | route_steps | edge方向 | 一致 |
|------|------|------|:---:|:---:|:---:|:---:|
| Basin LSTM | 无 | DRP | — | — | — | ✅ |
| LSTM-only | mean pool | DRP | — | — | — | ✅ |
| LSTM+GNN | GCN(84次/step) | DRP | — | — | downstream=上→下 | ✅ |
| WDNE | 自适应路由 | DRP | 实验1:无, 实验2:5字段 | 4 | downstream=上→下 | ✅ |
| WDNE attribute_encoder | 启用 | — | 实验2用LazyLinear→Linear | — | — | ✅ |

## 5. 指标与结果文件检查表

| 文件 | 存在 | 最新 | 符合设计 | 路径 |
|------|:---:|:---:|:---:|------|
| metric_inflow.csv (实验1各模型) | ✅ | ✅ | ✅ | results/{model}/songliao_3h_normal/ |
| metric_inflow.csv (实验2各模型) | ✅ | ✅ | ✅ | results/static_5basins_{model}/ |
| metric_inflow.csv (实验3) | ✅ | ❌ | ⚠️ 旧配置 | results/single_basin/ |
| all_events_metrics.csv | ✅ | ❌ | ❌ 旧数据 | results/report/data/ |
| 01_exp1_nse.png | ✅ | ✅ | ✅ | results/report/figures/ |
| 02_exp2_static.png | ✅ | ✅ | ✅ | results/report/figures/ |
| 03_wdne_evolution.png | ✅ | ✅ | ✅ | results/report/figures/ |
| 04_avg_nse.png | ✅ | ✅ | ✅ | results/report/figures/ |
| flood_events/ (23张) | ✅ | ✅ | ⚠️ 未分train/test | results/report/figures/flood_events/ |
| 空间敏感性图 | ❌ | — | ❌ 未生成 | — |
| REPORT.md | ✅ | ✅ | ✅ | results/report/ |
| PPT_OUTLINE.md | ✅ | ✅ | ✅ | results/report/ |

## 6. 必须修正的问题

**P1 (必须修):**
1. 重跑 `all_events_metrics.csv` — 使用最新模型结果的逐事件指标
2. 完成实验4空间敏感性 — 生成 normal/sim1/sim2/sim3 NC文件并画图

**P2 (建议修):**
3. 实验3用 steps=4 重跑
4. flood_events/ 增加 train/test 子目录分层
5. 清理 report/figures/ 中旧图表（exp1_baseline, exp2_static, exp3_single 子目录有残留旧文件）

**当前一致，无需修改:**
- 数据划分、输入输出、模型配置、edge方向、route_steps 全部正确
