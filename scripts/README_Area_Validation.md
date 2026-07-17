# Area Validation Script — 说明

## 这个脚本在做什么？

**不是**用蓝图（blueprint）验证绝对面积真值。

本项目的面积流程是：

1. U-Net 预测房间 **像素 mask**
2. 用户在 Web App 侧边栏设置 **比例尺**（Reference Length in pixels + Actual Length in cm）
3. 系统按 \( \text{面积(m²)} = \text{像素面积} / (\text{pixels/cm})^2 / 10000 \) 换算

因此 **绝对 m² 由用户标定决定**；调比例尺就会改变识别/显示面积。论文在 **§3 Real-World Area Calculation** 和 **§5 Discussion** 中说明这一点。

`batch_area_validation.py` 只做 **内部一致性** 检查：

| 比较对象 | 含义 |
|---------|------|
| 预测 mask 房间总面积 | U-Net 输出 |
| `ground_truth_m2` 在 JSON 里 | 默认来自 **人工标注分割 mask**（同一把比例尺换算），不是蓝图 |
| Per-room 误差 | 预测房间 vs 标注 mask 房间 |

## 快速运行

```bat
run_area_validation.bat
```

## 配置文件

`data/area_validation/ground_truth.json`：

- `ref_pixels` / `ref_length_cm`：标定（默认门宽 90 cm，可改为你 Web 里用的值）
- `ground_truth_m2`：**标注 mask 推导的总面积**（`--fill-gt-from-masks` 自动填写）
- 若你**没有**独立蓝图，**不要**把 `ground_truth_m2` 填成“真实建筑面积”；保持 mask 参考即可

## 输出

- `models/area_validation/results.json` — 汇总与逐户结果
- `Paper.txt` 表 `\ref{tab:area_validation}` — 仅在你运行 `--update-paper` 时更新 **内部分割一致性** 数字

## 与审稿意见的关系

若审稿人要求“多户型面积误差分布”：

- **有蓝图/as-built 面积时**：在 JSON 中为各 plan 填写真实 `ground_truth_m2` 后重跑
- **无蓝图时（当前情况）**：在论文中报告 **标定敏感性 + 预测/标注 mask 一致性 + Web 定性案例**，并在 Limitations 中说明绝对值依赖用户比例尺
