# 领域资料与供数约定

## 数据含义

conditions 为日记录，soil_wetness 为 0 至 1 的观测比例，缺测为 null。visits.status 是 unvisited、not_workable、worked、unknown 之一；worked 对应 work_start 到 work_end 含两端的一段可能作业日期，不表示每天都可作业。known_at 为补录可知时点；conditions.day 之后才发生的值不能进入早于该日的预测。requests.horizon_days 示例为 7；visit_budget 为可安排走访次数，而非独立样本数。

## 现行摘要

`field_window/analysis.py` 生成现行资料摘要。源码和运行结果中的 scope 说明其统计边界，不能把未提供的推断当成已有能力。

## 可作业窗口与排期

`field_window/planning.py` 在摘要之外生成未来一周的窗口概率与排期参考，`python3 -m field_window.runner --plan-output <路径>` 可实际运行。

- worked 只作为“区间内至少一天可作业”的区间证据，不把 work_start、work_end 或 visited_on 当成已作业日；`possible_days` 仍按含两端计算。
- not_workable 只否定其走访日；unknown（已作业但日期不详）与 unvisited（未走访）都不进入逐日训练，也不当作不可作业。
- 训练只使用 `known_at` 不晚于预测切点的记录；条件表中晚于切点当日的值不参与该切点的特征统计。`metadata.cutoff` 不自动成为训练截止点。
- 验证使用前向时间切分，并剔除验证窗口日期、前后禁带以及同地块或同村同场降雨；报告的是不同走访预算下的可信窗口数与事后核对，不报告总体准确率。
- 作物支持不足、地块历史不足、未来一周缺测，或未来一周连续强降雨、日降雨、土壤含水率超出该地块历史观测上限时，不输出精确排期建议，并记录原因。

## 供数接口

`GET /manifest` 返回 generation、tables（每表 count 与 sha256）及 metadata。`GET /tables/<name>?generation=<generation>` 返回同代 rows。供数换代时旧 generation 返回 HTTP 409，不允许将新代记录接入旧分析；不存在的表返回 404。表内容为稳定键唯一的完整同代记录，不是增量事件流。客户端对清单摘要进行核验。摘要使用 UTF-8、排序对象键、紧凑 JSON 的 SHA256，数组顺序保留供数端顺序。

字段与主键声明位于 `field_window/schema.json`，合成资料位于 `fixtures/dataset.json`。`metadata.cutoff` 是供数展示截点，不自动为分析选择训练截止点。服务可将整个 dataset 原子替换为新一代，但不在同一 generation 下更改内容。

## 结果保存

`reports.py` 保存现有摘要及来源表摘要，report_id 是内容摘要，不是随机编号。相同内容生成相同编号，临时文件写入完成后原子替换目标文件；历史内容是否保留由调用者选择不同目标路径。服务只监听本机，不接入生产系统。
