# 地块农事作业窗口的区间预测与可用性判断

本项目组织领域资料并提供当前摘要，通过本地 HTTP 供数服务连接应用。业务代码在 `field_window`，字段与单位约定见 `docs/data-contract.md`。示例均为本地合成记录，不代表真实县域记录或业务结论。

## 环境

Python 3.11 及以上，业务运行仅使用标准库。构建工具及其依赖完整固定在 `requirements-build.txt`；先用 `python3 -m pip install -r requirements-build.txt` 安装。源码和业务测试无需访问外部服务，首次安装构建依赖需要访问 Python 包仓库。

## 当前摘要

在项目根目录运行，会临时启动真实本地 HTTP 服务、核验资料代际、计算现有摘要并完整打印 JSON：

```sh
python3 -m field_window.runner --output /tmp/field_window-summary.json
```

当前能力是源码 `analysis.py` 定义的摘要，不会自动给出其他统计推断。输出中的 scope 说明边界，sources 记录资料依据。

## 未来一周可作业窗口与走访排期

在现有摘要之外，`--plan-output` 会额外生成窗口概率、不确定性、经验边界和不同走访预算下的可信窗口数：

```sh
python3 -m field_window.runner --output /tmp/field_window-summary.json \
  --plan-output /tmp/field_window-plan.json
```

`field_window/planning.py` 的 `plan(tables)` 也可直接调用。新增能力的要点：

- **标签语义**：`worked` 的 `work_start..work_end` 只表示“区间内至少一天可作业”，由区间删失模型学习逐日概率，不把填报日或区间内每一天当作作业日；`unknown` 是“已作业但日期不详”，`unvisited` 是“未走访”，两者都不进入逐日训练。`label_evidence.audit` 记录每类记录的使用与排除原因。
- **未走访 / 无法作业 / 已作业日期不详** 分开处理，不会被压成同一个负类。
- **泄漏隔离**：训练只使用 `known_at` 不晚于切点的记录；切点后的补录、跨越切点的作业区间、验证窗口及其前后禁带、同地块或同村同场降雨都不进入训练。`validation` 用前向时间切分，并报告每个走访预算下实际排入日程的可信窗口数与事后核对结果，而不是单一总体准确率。
- **不确定性**：窗口概率为日级可作业概率，`lower`/`upper` 是系数后验抽样的 10%–90% 区间；`credible` 以保守下界不低于 0.5 判定。
- **停止给出精确建议**：没有可辨识的天气—作业关系时输出 `advice: "baseline"`；作物支持不足、地块历史太少、未来一周缺测或连续强降雨/降雨/含水率超出该地块历史经验时输出 `advice: "withheld"` 并给出原因。


## 分开运行供数与应用

```sh
python3 -m field_window.runner --serve --port 8765
```

另一个终端可运行：

```sh
python3 -m field_window.runner --source-url http://127.0.0.1:8765 --output /tmp/field_window-summary.json
```

## 测试与检查

```sh
python3 -m unittest discover -s tests -t . -v
python3 -m compileall -q field_window tests
python3 -m build --wheel --no-isolation
```

测试覆盖现有领域摘要、非法引用、真实 HTTP 换代拒绝、表清单内容核验及结果保存；`tests/test_planning.py` 覆盖区间标签语义、补录与同场降雨隔离、不同预算的可信窗口阶梯和停止建议。展示现有产品可直接运行上述摘要命令；排期能力用 `--plan-output` 实际运行。不需要回放历史日志。扩展分析时应同步说明新增能力的实际运行入口，保留现行摘要调用。

## 容器运行

Dockerfile 固定官方 Python 3.11.14 slim-bookworm 镜像摘要，不依赖本机私有镜像；首次构建需下载基础镜像和固定构建工具。构建与运行实际摘要：

```sh
docker build -t local/field-window .
docker run --rm local/field-window
```

容器内测试：

```sh
docker run --rm --entrypoint python3 local/field-window -m unittest discover -s tests -t . -v
```
