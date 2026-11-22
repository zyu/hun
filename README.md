# 比分牌鲁棒性数据集工具

macOS 双击「启动工具.command」。也可以使用带 Tkinter 的 Python 3.11 或更新版本，在项目目录运行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

## 在 Windows 打包

在 Windows 上使用带 Tkinter 的 64 位 Python（推荐 3.12），然后双击 `build_windows.bat`。脚本会安装打包依赖，并生成 `dist\ScoreboardDataset\ScoreboardDataset.exe`。分发时复制整个 `dist\ScoreboardDataset` 文件夹，运行 EXE 的电脑无需另装 Python。命令行可加 `--no-pause`。

打包时会收入项目中已有的 `yolo26n.pt` 初始权重。视频取帧可使用放在 EXE 旁的 `ffmpeg.exe`，或使用 PATH 中的 FFmpeg。在 Windows 上，默认输出位于 `%LOCALAPPDATA%\ScoreboardDataset\转换数据集`。

## 生成合成样本

1. 批量选图、递归导入文件夹，或点击「奥运样图」载入项目中的 34 张羽毛球截图。选中图片可查看原图与转换结果。
2. 勾选困难主类，选择生成强度。每张原图按每个主类、每档强度生成一张。「附加难度」可叠加轻度压缩、模糊和低对比效果。
3. 选择输出位置，点击「开始批量生成」。转换结果支持八类：低清压缩、小字细笔画、模糊交错、低对比透明底、遮挡截断、状态过渡、画中画及多语言姓名。

标注采用 YOLO 检测框格式。工具读取图片旁边的同名 TXT，也支持 `images` 和 `labels` 对应的目录布局。局部转换使用类别 0 的比分牌框，多语言转换使用类别 1 的姓名框，局部比分过渡使用类别 2 和 3 的计分框。画中画需要两张来自同一比赛来源、同一数据划分且有比分牌框的截图。

「数据来源」可导入包含 `image_id,sport,split,source_id,match_id,timestamp_ms` 的 CSV。文件名或不含后缀的名称均可用作 `image_id`。合成样本继承原图划分，训练派生仅处理训练集，尚未划分的样本导出到 `unassigned`，缺标注的样本进入 `review`。

多语言姓名使用 `image_id,row,text` 真值 CSV。按原图各姓名框从上到下从 1 编号，`text` 填该选手的真实显示名称，可以选择对应字体。没有姓名真值时，工具对已有文字做窄体变换。状态过渡功能生成保留原比分内容的单帧效果。真实比分更新和换盘画面可从本地比赛视频取帧。

## 归档真实困难帧

使用「视频取帧」按时间区间抽取本地比赛视频的截图（需要 PATH 中有 FFmpeg），或直接导入已采集的截图。补齐 YOLO 标注，再选中同类截图归档。每批可选择一个主类、多个难度标签和文字可读性。完成复核并包含比分牌标注的真实帧按原图 SHA-256 去重计数。默认配额为八类 × 两种球类 × 375 张，共 6,000 张真实困难帧。

输出目录分为 `real` 和 `synthetic`。合成批次的图片和 YOLO 标注按相同的「数据划分／困难主类」子目录保存。每批的 `manifest.json` 记录来源、主类、难度标签、参数和未导出原因。根目录的 `quota_report.json` 与界面「查看配额」分别列出真实进度、缺口、合成数量和待复核数量。合成样本的可读性应在正式使用前抽检。

## YOLO26 比分牌检测

主窗口勾选「显示 YOLO 框」即可检查读取到的原图和转换图标注。选择「YOLO26」可执行三项操作：

- **导出数据集**。根据导入队列的图片及同名 TXT 生成 `data.yaml` 和匹配的 `images/{train,val,test}`、`labels/{train,val,test}`。空 TXT 作为背景样本保留，缺 TXT 或格式错误的图片列入 `dataset_report.json`。类别名按原始编号填写，可修改默认值中的 `class_2`、`class_3`。同一比赛保留在同一划分。勾选增强样本后，仅收入来源匹配且已划入训练集的派生图。
- **开始训练**。选择本工具导出的 `data.yaml` 或已有的 YOLO 数据配置，调整轮数、输入尺寸和设备。默认使用 `yolo26n.pt` 初始化，已有本地权重时可直接选取，也可填 `yolo26n.yaml` 从零初始化。训练完成后，`转换数据集/yolo26/runs/` 下的对应运行目录会保留日志和 `train/weights/best.pt`。当前 34 张奥运截图来自同一场比赛，导出后属于训练集，需补充另一场带标注的比赛作为验证集。
- **批量预标注**。选择比分牌模型权重（例如训练得到的 `best.pt`），对导入队列生成同名 TXT，并写入单独的预标注目录。可通过「载入最近预标注」检查结果，原始 TXT 保持不变。

在项目目录执行 `.venv/bin/python -m pip install -r requirements-yolo26.txt` 即可安装训练和预标注依赖。导出和查看标注只使用基础依赖。
