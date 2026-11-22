from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).resolve()
datas = copy_metadata("ultralytics")
weights = next((path for path in (
    root / "models" / "yolo26n.pt",
    root / "转换数据集" / "yolo26" / "models" / "yolo26n.pt",
) if path.is_file()), None)
if weights:
    datas.append((str(weights), "models"))

ffmpeg = root / "ffmpeg.exe"
binaries = collect_dynamic_libs("torchvision", search_patterns=["*.dll", "*.pyd", "*.so", "*.dylib"])
if ffmpeg.is_file():
    binaries.append((str(ffmpeg), "."))
analysis = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["ultralytics.models.yolo.detect",
                   *collect_submodules("ultralytics.utils.callbacks")],
)
archive = PYZ(analysis.pure)
exe = EXE(
    archive,
    analysis.scripts,
    exclude_binaries=True,
    name="ScoreboardDataset",
    console=False,
    upx=False,
)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, name="ScoreboardDataset", upx=False)
