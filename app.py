"""Tk desktop interface for preparing scoreboard robustness datasets."""

import os
import csv
import importlib.util
import json
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from augment import (
    BatchSettings, CLASSES, IMAGE_EXTENSIONS, SPLITS, SPORTS, EXTRA_EFFECTS, atomic_json,
    find_secondary, is_prediction, load_frame, names_for, open_input_frame, quotas, read_names, read_sources,
    run_batch, source_metadata, transform_sample,
)
from yolo26 import DEFAULT_NAMES, export_dataset, run_folder, run_worker_with_log

PROJECT = Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else PROJECT
SAMPLES = APP_DIR / "奥运会OLY"
USER_DATA = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
OUTPUT_ROOT = USER_DATA / "ScoreboardDataset" if FROZEN and sys.platform == "win32" else APP_DIR
DEFAULT_OUTPUT = OUTPUT_ROOT / "转换数据集"
NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)
LEVELS = ("轻度", "中度", "重度")
READABILITY = {
    "readable": "可读 · 已复核",
    "unreadable": "不可读 · 已完整标注",
    "context_required": "依赖上下文 · 已完整标注",
    "review": "待复核",
}
BG = "#edf1f6"
WHITE = "#ffffff"
INK = "#14253d"
MUTED = "#4d6077"
BLUE = "#086d69"
NAVY = "#14253e"
BORDER = "#dae2ed"
FONT = ("PingFang SC" if sys.platform == "darwin" else "TkDefaultFont")
BOX_COLORS = ("#07e6dc", "#ffcf47", "#ff779d", "#99ff7f")


def preview_image(image, boxes):
    width, height = image.size
    normalized = [(box.cls, (box.rect[0] / width, box.rect[1] / height,
                             box.rect[2] / width, box.rect[3] / height)) for box in boxes]
    image = image.copy()
    image.thumbnail((800, 500))
    return image, normalized


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("比分牌鲁棒性数据集")
        self.geometry("1180x820")
        self.minsize(980, 720)
        self.configure(bg=BG)
        self.paths = set()
        self.source_rows = {}
        self.capture_sources = {}
        self.read_capture_indexes = set()
        self.name_rows = {}
        self.source_csv = None
        self.name_csv = None
        self.font_path = None
        self.output = DEFAULT_OUTPUT
        self.latest = None
        self.yolo_names = tk.StringVar(value=", ".join(DEFAULT_NAMES))
        self.yolo_data = None
        self.yolo_weights = None
        self.show_boxes = tk.BooleanVar(value=True)
        self.default_sport = "badminton"
        self.default_split = "unassigned"
        self.mode = tk.StringVar(value="synthetic")
        self.choice = {kind: tk.BooleanVar(value=True) for kind in CLASSES}
        self.extra_effects = {kind: tk.BooleanVar() for kind in EXTRA_EFFECTS}
        self.level = tk.StringVar(value="中度")
        self.primary = tk.StringVar(value=next(iter(CLASSES.values())))
        self.readability = tk.StringVar(value=READABILITY["review"])
        self.preview_mode = tk.StringVar(value=next(iter(CLASSES.values())))
        self.summary = tk.StringVar(value="选中一张图片查看对照")
        self.import_count = tk.StringVar(value="0 张")
        self.expected = tk.StringVar()
        self.status = tk.StringVar(value="导入截图后即可生成或归档数据集")
        self.destination_text = tk.StringVar(value=str(self.output))
        self.events = queue.Queue()
        self.preview_pool = ThreadPoolExecutor(max_workers=1)
        self.preview_future = None
        self.preview_job = None
        self.preview_token = 0
        self.preview_frames = (None, None)
        self.busy = False
        self.closing = False
        self.stop = threading.Event()
        self.image_refs = {}
        self.controls = []
        self.setup_styles()
        self.build()
        self.change_mode()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(80, self.poll_events)

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 12), background=WHITE, foreground=INK, bordercolor=BORDER)
        style.configure("TButton", padding=(11, 6), background=WHITE, bordercolor=BORDER)
        style.map("TButton", background=[("active", "#e7eef6")])
        style.configure("Primary.TButton", background=BLUE, foreground=WHITE, padding=(15, 8),
                        bordercolor=BLUE, lightcolor=BLUE, darkcolor=BLUE)
        style.map("Primary.TButton", background=[("disabled", "#9babaf"), ("active", "#075c59")],
                  foreground=[("disabled", WHITE)])
        style.configure("TCheckbutton", background=WHITE, padding=(0, 1))
        style.map("TCheckbutton", background=[("active", WHITE)])
        style.configure("TRadiobutton", background=WHITE)
        style.map("TRadiobutton", background=[("active", WHITE)])
        style.configure("TCombobox", padding=3)
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, rowheight=24)
        style.configure("Treeview.Heading", background="#e9eff6", padding=(7, 5))
        style.map("Treeview", background=[("selected", "#d7ece9")], foreground=[("selected", INK)])
        style.configure("Horizontal.TProgressbar", background=BLUE, troughcolor="#dce5ee",
                        bordercolor=BG, lightcolor=BLUE, darkcolor=BLUE)

    def label(self, parent, text="", *, size=12, bold=False, fg=INK, bg=WHITE, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=fg, anchor="w",
                        font=(FONT, size, "bold" if bold else "normal"), **kwargs)

    def button(self, parent, text, command, primary=False, lock=True):
        widget = ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton")
        if lock:
            self.controls.append(widget)
        return widget

    def combo(self, parent, variable, values, callback=None, width=20):
        widget = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly")
        self.controls.append(widget)
        if callback:
            widget.bind("<<ComboboxSelected>>", lambda _: callback())
        return widget

    def build(self):
        head = tk.Frame(self, bg=NAVY, padx=24, pady=14)
        head.pack(fill="x")
        self.label(head, "比分牌鲁棒性数据集", size=23, bold=True, fg=WHITE, bg=NAVY).pack(side="left")
        self.label(head, "羽毛球  /  网球", size=12, fg="#c7d5e6", bg=NAVY).pack(side="right")

        footer = tk.Frame(self, bg=BG, padx=20, pady=11)
        footer.pack(side="bottom", fill="x")
        self.start_button = self.button(footer, "开始批量生成", self.start, primary=True)
        self.start_button.pack(side="right")
        self.cancel_button = self.button(footer, "停止", self.cancel, lock=False)
        self.cancel_button.state(["disabled"])
        self.cancel_button.pack(side="right", padx=6)
        self.button(footer, "打开输出", self.open_output, lock=False).pack(side="right")
        progress_area = tk.Frame(footer, bg=BG)
        progress_area.pack(side="left", fill="both", expand=True, padx=(0, 13))
        self.label(progress_area, textvariable=self.status, bg=BG, fg=MUTED).pack(fill="x")
        self.progress = ttk.Progressbar(progress_area, maximum=1)
        self.progress.pack(fill="x", pady=(5, 0))

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=20, pady=(14, 0))
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        panel = tk.Frame(content, bg=WHITE, padx=15, pady=12, highlightbackground=BORDER, highlightthickness=1)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self.build_settings(panel)

        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=4, minsize=244)
        right.grid_rowconfigure(1, weight=3, minsize=215)
        self.build_previews(right)
        self.build_imports(right)

    def build_settings(self, panel):
        self.label(panel, "处理方式", bold=True, size=15).pack(fill="x")
        modes = tk.Frame(panel, bg=WHITE)
        modes.pack(fill="x", pady=(5, 9))
        for value, name in (("synthetic", "生成合成样本"), ("real", "归档真实帧")):
            button = ttk.Radiobutton(modes, text=name, value=value, variable=self.mode, command=self.change_mode)
            button.pack(side="left", padx=(0, 7))
            self.controls.append(button)

        self.settings_body = tk.Frame(panel, bg=WHITE)
        self.settings_body.pack(fill="both", expand=True)
        self.synthetic_panel = tk.Frame(self.settings_body, bg=WHITE)
        self.real_panel = tk.Frame(self.settings_body, bg=WHITE)
        header = tk.Frame(self.synthetic_panel, bg=WHITE)
        header.pack(fill="x")
        self.label(header, "困难主类", bold=True, size=15).pack(side="left")
        self.button(header, "附加难度", self.show_extra_effects).pack(side="right")
        for kind, name in CLASSES.items():
            box = ttk.Checkbutton(self.synthetic_panel, text=name, variable=self.choice[kind], command=self.update_options)
            box.pack(anchor="w", pady=1)
            self.controls.append(box)
        self.label(self.synthetic_panel, "生成强度", bold=True, size=14).pack(fill="x", pady=(10, 4))
        self.combo(self.synthetic_panel, self.level, (*LEVELS, "全部三档"), self.update_options).pack(fill="x")
        self.label(self.synthetic_panel, textvariable=self.expected, fg=MUTED).pack(fill="x", pady=(7, 0))

        self.label(self.real_panel, "计入配额的主类", bold=True, size=15).pack(fill="x", pady=(1, 4))
        self.combo(self.real_panel, self.primary, tuple(CLASSES.values())).pack(fill="x")
        self.label(self.real_panel, "同时标记其他难度", bold=True, size=14).pack(fill="x", pady=(10, 3))
        self.real_tags = {}
        for kind, name in CLASSES.items():
            variable = tk.BooleanVar()
            box = ttk.Checkbutton(self.real_panel, text=name, variable=variable)
            box.pack(anchor="w")
            self.real_tags[kind] = variable
            self.controls.append(box)
        self.label(self.real_panel, "文字可读性与复核状态", bold=True, size=14).pack(fill="x", pady=(9, 4))
        self.combo(self.real_panel, self.readability, tuple(READABILITY.values())).pack(fill="x")
        self.label(self.real_panel, "选中多张可同批归档。", fg=MUTED).pack(fill="x", pady=(5, 0))

        ttk.Separator(panel).pack(fill="x", pady=(10, 8))
        self.label(panel, "数据集", bold=True, size=14).pack(fill="x")
        self.label(panel, textvariable=self.destination_text, fg=MUTED, wraplength=230,
                   justify="left").pack(fill="x", pady=(3, 6))
        paths = tk.Frame(panel, bg=WHITE)
        paths.pack(fill="x")
        self.button(paths, "数据来源", self.show_configuration).pack(side="left")
        self.button(paths, "输出位置", self.pick_output).pack(side="left", padx=(5, 0))
        tools = tk.Frame(panel, bg=WHITE)
        tools.pack(fill="x", pady=(6, 0))
        self.button(tools, "查看配额", self.show_quota, lock=False).pack(side="left", fill="x", expand=True)
        self.button(tools, "YOLO26", self.show_yolo).pack(side="left", fill="x", expand=True, padx=(5, 0))

    def build_previews(self, right):
        panel = tk.Frame(right, bg=WHITE, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
        panel.grid(row=0, column=0, sticky="nsew", pady=(0, 11))
        panel.grid_columnconfigure((0, 1), weight=1, uniform="preview")
        panel.grid_rowconfigure(2, weight=1)
        self.label(panel, "效果预览", size=15, bold=True).grid(row=0, column=0, sticky="w")
        self.preview_combo = self.combo(panel, self.preview_mode, tuple(CLASSES.values()), self.schedule_preview)
        self.preview_combo.grid(row=0, column=1, sticky="e")
        self.original_caption = self.label(panel, "原图", fg=MUTED)
        self.original_caption.grid(row=1, column=0, sticky="w", pady=(5, 2))
        ttk.Checkbutton(panel, text="显示 YOLO 框", variable=self.show_boxes,
                        command=lambda: [self.draw_preview(i) for i in range(2)]).grid(
                            row=1, column=0, sticky="e", padx=(0, 5))
        self.result_caption = self.label(panel, "转换结果", fg=MUTED)
        self.result_caption.grid(row=1, column=1, sticky="w", pady=(5, 2), padx=(5, 0))
        self.canvases = []
        for index in range(2):
            canvas = tk.Canvas(panel, bg=NAVY, highlightthickness=0, width=220, height=130)
            canvas.grid(row=2, column=index, sticky="nsew", padx=(0, 5) if index == 0 else (5, 0))
            canvas.bind("<Configure>", lambda _, slot=index: self.draw_preview(slot))
            self.canvases.append(canvas)
        self.label(panel, textvariable=self.summary, fg=MUTED, size=11).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    def build_imports(self, right):
        panel = tk.Frame(right, bg=WHITE, padx=12, pady=9, highlightbackground=BORDER, highlightthickness=1)
        panel.grid(row=1, column=0, sticky="nsew")
        top = tk.Frame(panel, bg=WHITE)
        top.pack(fill="x", pady=(0, 6))
        self.label(top, "导入队列", bold=True, size=15).pack(side="left")
        self.label(top, textvariable=self.import_count, fg=MUTED).pack(side="left", padx=6)
        for name, command in (("导入图片", self.pick_files), ("导入文件夹", self.pick_folder),
                              ("视频取帧", self.video_dialog), ("奥运样图", self.add_samples)):
            self.button(top, name, command).pack(side="left", padx=2)
        self.button(top, "移除选中", self.remove_selected).pack(side="right")
        table_container = tk.Frame(panel, bg=WHITE)
        table_container.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_container, columns=("name", "sport", "split"), show="headings", height=4)
        for key, title, width in (("name", "文件名", 425), ("sport", "球种", 64), ("split", "划分", 73)):
            self.table.heading(key, text=title, anchor="w")
            self.table.column(key, width=width, minwidth=60, stretch=key == "name")
        scroll = ttk.Scrollbar(table_container, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", lambda _: self.schedule_preview())
        self.table.bind("<Command-a>" if sys.platform == "darwin" else "<Control-a>",
                        lambda _: self.table.selection_set(self.table.get_children()))

    def source_rows_for(self, paths):
        rows = {}
        for path in paths:
            capture_rows = next((self.capture_sources[parent] for parent in path.parents
                                 if parent in self.capture_sources), {})
            local = capture_rows.get(path.name, capture_rows.get(path.stem, {}))
            manual = self.source_rows.get(str(path), self.source_rows.get(path.name, self.source_rows.get(path.stem, {})))
            rows[str(path)] = {**local, **manual}
        return rows

    def metadata(self, paths):
        paths = tuple(paths)
        rows = self.source_rows_for(paths)
        return {path: source_metadata(path, rows, self.default_sport, self.default_split)
                for path in paths}

    def add_paths(self, paths):
        paths = tuple(Path(value).resolve() for value in paths)
        roots = {parent.parent for path in paths for parent in path.parents if parent.name == "images"}
        for index in sorted({index for root in roots for index in root.glob("*captures.csv")}):
            if index.is_file() and index not in self.read_capture_indexes:
                try:
                    self.capture_sources.setdefault(index.parent, {}).update(read_sources(index))
                    self.read_capture_indexes.add(index)
                except (OSError, ValueError) as exc:
                    messagebox.showerror("读取来源失败", str(exc), parent=self)
        selected = None
        for path, row in self.metadata(paths).items():
            if path in self.paths or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = str(path)
            self.table.insert("", "end", iid=key, values=(path.name, SPORTS[row["sport"]], SPLITS[row["split"]]))
            self.paths.add(path)
            if selected is None:
                selected = key
        if selected is not None and not self.table.selection():
            self.table.selection_set(selected)
            self.table.see(selected)
        self.update_options()

    def add_folder(self, folder):
        def images():
            for directory, children, filenames in os.walk(folder):
                children[:] = sorted(name for name in children if not name.startswith((".", "batch-"))
                                     and (Path(directory) / name).resolve() != self.output.resolve())
                for filename in sorted(filenames):
                    if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                        yield Path(directory) / filename
        count = len(self.paths)
        self.add_paths(images())
        if len(self.paths) == count:
            self.status.set("没有新增图片")

    def pick_files(self):
        extensions = " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS))
        picked = filedialog.askopenfilenames(parent=self, filetypes=[("图片", extensions), ("所有文件", "*")])
        if picked:
            self.add_paths(picked)

    def pick_folder(self):
        picked = filedialog.askdirectory(parent=self)
        if picked:
            self.add_folder(Path(picked))

    def add_samples(self):
        if not SAMPLES.is_dir():
            messagebox.showinfo("奥运样图", "项目目录里没有奥运会样图。", parent=self)
            return
        self.add_folder(SAMPLES / "images")
        self.refresh_rows()

    def remove_selected(self):
        for key in self.table.selection():
            self.paths.remove(Path(key))
            self.table.delete(key)
        self.update_options()

    def refresh_rows(self):
        for key, item in self.metadata(map(Path, self.table.get_children())).items():
            self.table.item(str(key), values=(key.name, SPORTS[item["sport"]], SPLITS[item["split"]]))
        self.schedule_preview()

    def pick_output(self):
        initial = next(path for path in (self.output, *self.output.parents) if path.is_dir())
        picked = filedialog.askdirectory(parent=self, initialdir=initial)
        if picked:
            self.output = Path(picked).resolve()
            self.latest = None
            self.yolo_data = self.yolo_weights = None
            self.destination_text.set(str(self.output))

    def change_mode(self):
        real = self.mode.get() == "real"
        self.synthetic_panel.pack_forget()
        self.real_panel.pack_forget()
        (self.real_panel if real else self.synthetic_panel).pack(fill="both", expand=True)
        self.start_button.configure(text="归档真实困难帧" if real else "开始批量生成")
        self.result_caption.configure(text="真实帧归档预览" if real else "转换结果")
        self.update_options()

    def selected_modes(self):
        return tuple(kind for kind, variable in self.choice.items() if variable.get())

    def selected_levels(self):
        return (LEVELS.index(self.level.get()),) if self.level.get() in LEVELS else (0, 1, 2)

    def update_options(self):
        self.import_count.set(f"{len(self.paths)} 张")
        modes = self.selected_modes()
        count = len(self.paths) * len(modes) * len(self.selected_levels())
        self.expected.set(f"本批最多生成 {count:,} 张")
        values = tuple(CLASSES[kind] for kind in modes)
        self.preview_combo.configure(values=values)
        if self.preview_mode.get() not in values:
            self.preview_mode.set(values[0] if values else "")
        self.sync_controls()
        self.schedule_preview()

    def sync_controls(self):
        self.start_button.state(["!disabled"] if not self.busy and self.paths
                                and (self.mode.get() == "real" or self.selected_modes()) else ["disabled"])
        self.preview_combo.state(["!disabled"] if not self.busy and self.mode.get() == "synthetic"
                                 and self.selected_modes() else ["disabled"])

    def schedule_preview(self):
        self.preview_token += 1
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
        if self.preview_future is not None:
            self.preview_future.cancel()
        self.preview_job = self.after(130, self.start_preview)

    def start_preview(self):
        self.preview_job = None
        selection = self.table.selection()
        if not selection:
            self.preview_frames = (None, None)
            self.summary.set("选中一张图片查看对照")
            for index in range(2):
                self.draw_preview(index)
            return
        path = Path(selection[0])
        kinds = {name: kind for kind, name in CLASSES.items()}
        kind = kinds.get(self.preview_mode.get()) if self.mode.get() == "synthetic" else None
        token = self.preview_token
        level = self.selected_levels()[0]
        metadata = self.metadata(map(Path, self.table.get_children())) if kind == "multiple" else None
        names = names_for(path, self.name_rows)
        font_path = self.font_path
        extras = tuple(key for key, value in self.extra_effects.items() if value.get())
        real = self.mode.get() == "real"

        def work():
            original = result = None
            try:
                candidate = is_prediction(path)
                image, boxes, label = load_frame(path) if candidate else open_input_frame(path)
                original = preview_image(image, boxes)
                note = f"同名 TXT · {len(boxes)} 个框" if boxes else (
                    "同名 TXT 为空" if label else "缺少同名 TXT")
                if kind and not candidate:
                    secondary = find_secondary(path, metadata)[1] if metadata else None
                    converted, output_boxes, params = transform_sample(
                        image, boxes, kind, level, secondary, names, font_path, extras)
                    result = preview_image(converted, output_boxes)
                    description = params.get("variant", CLASSES[kind])
                    info = f"{description}  ·  {LEVELS[level]}  ·  {note}"
                else:
                    result = original if real else None
                    info = f"YOLO26 预标注待复核  ·  {note}" if candidate else note
            except Exception as exc:
                info = (f"{note}  ·  " if original else "") + str(exc)
            self.events.put(("preview", (token, original, result, info)))

        self.summary.set("正在读取预览…")
        self.preview_future = self.preview_pool.submit(work)

    def draw_preview(self, index):
        canvas = self.canvases[index]
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        frame = self.preview_frames[index]
        if frame is None:
            canvas.create_text(width / 2, height / 2, text="暂无预览", fill="#c7d5e6", font=(FONT, 12))
            self.image_refs.pop(index, None)
            return
        source, boxes = frame
        scaled = source.copy()
        scaled.thumbnail((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(scaled, master=self)
        self.image_refs[index] = image
        canvas.create_image(width / 2, height / 2, image=image)
        if self.show_boxes.get():
            w, h = scaled.size
            ox, oy = (width - w) / 2, (height - h) / 2
            for kind, (x1, y1, x2, y2) in boxes:
                color = BOX_COLORS[kind % len(BOX_COLORS)]
                x, y = ox + w * x1, oy + h * y1
                canvas.create_rectangle(x, y, ox + w * x2, oy + h * y2, outline=color, tags="yolo_box")

    def show_configuration(self):
        window = self.dialog("数据来源")
        sport = tk.StringVar(value=SPORTS[self.default_sport])
        split = tk.StringVar(value=SPLITS[self.default_split])
        for row, (title, variable, values) in enumerate((
            ("默认球种", sport, tuple(SPORTS.values())),
            ("未指定划分时", split, tuple(SPLITS.values())),
        )):
            self.label(window, title, bold=True).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Combobox(window, textvariable=variable, values=values, state="readonly").grid(
                row=row, column=1, columnspan=2, sticky="ew", pady=4)
        picked = [self.source_csv, self.name_csv, self.font_path]
        for index, (title, extensions) in enumerate((
            ("来源清单", ("CSV", "*.csv")),
            ("姓名真值", ("CSV", "*.csv")),
            ("姓名字体", ("字体", "*.ttf *.ttc *.otf")),
        )):
            row = index + 2
            value = tk.StringVar(value=Path(picked[index]).name if picked[index] else "未选择")
            self.label(window, title, bold=True).grid(row=row, column=0, sticky="w", pady=4)
            self.label(window, textvariable=value, fg=MUTED, width=31).grid(
                row=row, column=1, sticky="w", padx=6)

            def pick(slot=index, variable=value, file_types=extensions):
                selected = filedialog.askopenfilename(parent=window, filetypes=[file_types])
                if selected:
                    picked[slot] = Path(selected)
                    variable.set(picked[slot].name)

            def clear(slot=index, variable=value):
                picked[slot] = None
                variable.set("未选择")

            self.button(window, "选择", pick, lock=False).grid(row=row, column=2, padx=3)
            self.button(window, "清除", clear, lock=False).grid(row=row, column=3)
        self.label(window, "来源 CSV：image_id, sport, split, source_id, match_id, timestamp_ms\n"
                           "姓名 CSV：image_id, row, text（按姓名框从上到下，从 1 编号）\n"
                           "标注可放在同名 TXT 中，或放在与 images 对应的 labels 目录。",
                   wraplength=515, justify="left", fg=MUTED).grid(
                       row=5, column=0, columnspan=4, sticky="w", pady=(10, 12))

        def save():
            try:
                sources = read_sources(picked[0]) if picked[0] else {}
                names = read_names(picked[1]) if picked[1] else {}
            except (OSError, ValueError) as exc:
                messagebox.showerror("读取失败", str(exc), parent=window)
                return
            self.source_rows, self.name_rows = sources, names
            self.source_csv, self.name_csv, self.font_path = picked
            self.default_sport = next(key for key, name in SPORTS.items() if name == sport.get())
            self.default_split = next(key for key, name in SPLITS.items() if name == split.get())
            self.refresh_rows()
            window.destroy()

        self.button(window, "保存设置", save, primary=True, lock=False).grid(row=6, column=0, columnspan=4, sticky="e")

    def show_extra_effects(self):
        window = self.dialog("附加难度")
        self.label(window, "叠加到每个合成样本", size=15, bold=True).pack(anchor="w")
        for key, variable in self.extra_effects.items():
            ttk.Checkbutton(window, text=CLASSES[key], variable=variable,
                            command=self.schedule_preview).pack(anchor="w", pady=2)
        self.button(window, "完成", window.destroy, lock=False).pack(anchor="e", pady=(8, 0))

    def dialog(self, title):
        window = tk.Toplevel(self, bg=WHITE, padx=18, pady=14)
        window.title(title)
        window.transient(self)
        window.grab_set()
        return window

    def show_quota(self):
        try:
            report = quotas(self.output)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("读取配额失败", str(exc), parent=self)
            return
        window = self.dialog("真实困难帧配额")
        overview = (f"配额完成 {report['credited_real']:,} / {report['target_real']:,}  ·  "
                    f"真实 {report['real']:,} 张  ·  合成 {report['synthetic']:,} 张")
        self.label(window, overview, bold=True, size=16).pack(fill="x", pady=(0, 10))
        columns = (("类别", 152), ("球种", 68), ("目标", 58), ("真实", 58),
                   ("缺口", 58), ("合成", 58), ("真实待复核", 86))
        table = ttk.Treeview(window, columns=tuple(name for name, _ in columns), show="headings", height=16)
        for index, (name, width) in enumerate(columns):
            table.heading(index, text=name, anchor="w")
            table.column(index, width=width, anchor="w")
        for row in report["rows"]:
            table.insert("", "end", values=[
                CLASSES[row["primary_class"]], SPORTS[row["sport"]], row["target_real"], row["real"],
                row["remaining_real"], row["synthetic"], row["pending_real_review"],
            ])
        table.pack(fill="both", expand=True, pady=(0, 8))
        self.button(window, "关闭", window.destroy, lock=False).pack(anchor="e")

    def show_yolo(self):
        window = self.dialog("YOLO26 比分牌检测")
        root = self.output / "yolo26"
        paths = tuple(map(Path, self.table.get_children()))
        if self.yolo_data is None:
            self.yolo_data = next(iter(sorted((root / "datasets").glob("dataset-*/data.yaml"), reverse=True)), None)
        if self.yolo_weights is None:
            self.yolo_weights = next(iter(sorted((root / "runs").glob("run-*/train/weights/best.pt"), reverse=True)), None)
        data = tk.StringVar(value=str(self.yolo_data or ""))
        weights = tk.StringVar(value=str(self.yolo_weights or "yolo26n.pt"))
        epochs = tk.StringVar(value="50")
        size = tk.StringVar(value="640")
        conf = tk.StringVar(value="0.25")
        device = tk.StringVar(value="自动")
        extras = tk.BooleanVar()

        def picker(variable, types):
            selected = filedialog.askopenfilename(parent=window, filetypes=[types])
            if selected:
                variable.set(selected)

        self.label(window, f"当前队列 · {len(paths):,} 张", bold=True, size=16).grid(row=0, columnspan=3, sticky="w")
        for row, (title, variable, types) in enumerate((
            ("类别名（按编号）", self.yolo_names, None),
            ("训练 data.yaml", data, ("数据集配置", "*.yaml *.yml")),
            ("检测模型", weights, ("检测模型", "*.pt *.yaml *.yml")),
        ), 1):
            self.label(window, title).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(window, textvariable=variable, width=42).grid(row=row, column=1, sticky="ew", padx=6)
            if types:
                self.button(window, "选择", lambda var=variable, t=types: picker(var, t), lock=False).grid(
                    row=row, column=2, sticky="e")
        params = tk.Frame(window, bg=WHITE)
        params.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 3))
        for title, variable in (("轮数", epochs), ("输入尺寸", size), ("置信度", conf)):
            self.label(params, title).pack(side="left", padx=(0, 4))
            ttk.Entry(params, textvariable=variable, width=6).pack(side="left", padx=(0, 11))
        ttk.Combobox(params, textvariable=device, width=8, values=("自动", "CPU", "MPS", "CUDA 0"),
                     state="readonly").pack(side="left")
        ttk.Checkbutton(window, text="加入原图属于训练集的已导出增强样本", variable=extras).grid(
            row=5, columnspan=3, sticky="w", pady=(5, 0))
        self.label(window, "空 TXT 作为背景帧。导出按比赛来源分组。预标注结果另存，可载入检查。",
                   fg=MUTED, wraplength=540, justify="left").grid(
                       row=6, columnspan=3, sticky="w", pady=(5, 9))

        def start(action):
            try:
                if action in ("export", "predict") and not paths:
                    raise ValueError("请先在主窗口导入图片。")
                if action == "export":
                    names = tuple(value.strip() for value in self.yolo_names.get().replace("，", ",").split(","))
                    if any(not name for name in names) or len(set(names)) != len(names):
                        raise ValueError("请按编号输入不同的类别名，使用逗号分隔。")
                    include = extras.get()
                else:
                    if importlib.util.find_spec("ultralytics") is None:
                        hint = ("请用 build_windows.bat 重新打包。" if FROZEN else
                                "请在当前 Python 环境安装 requirements-yolo26.txt。")
                        raise ValueError("缺少 YOLO26 依赖。" + hint)
                    checkpoint = weights.get().strip()
                    stock_weights = {f"yolo26{scale}.pt" for scale in "nsmlx"}
                    candidate = Path(checkpoint).expanduser()
                    if candidate.is_file() and candidate.suffix.lower() in (".pt", ".yaml", ".yml"):
                        if action == "predict" and candidate.suffix.lower() != ".pt":
                            raise ValueError("比分牌预标注请选择已训练的 .pt 权重。")
                        checkpoint = str(candidate.resolve())
                    elif checkpoint in stock_weights:
                        if action == "predict":
                            raise ValueError("比分牌预标注请选择已训练的比分牌权重，例如 best.pt。")
                        model_dirs = (root / "models", DEFAULT_OUTPUT / "yolo26" / "models",
                                      APP_DIR / "models", PROJECT / "models")
                        cached = next((path for folder in model_dirs
                                       if (path := folder / checkpoint).is_file()), None)
                        if cached:
                            checkpoint = str(cached)
                    elif action == "train" and checkpoint in {f"yolo26{scale}.yaml" for scale in "nsmlx"}:
                        pass
                    else:
                        raise ValueError("请填写 yolo26n.pt，或选择本地权重。离线训练可用 yolo26n.yaml 从零初始化。")
                    imgsz = int(size.get())
                    if imgsz < 32 or imgsz % 32:
                        raise ValueError("输入尺寸至少为 32，并且是 32 的倍数。")
                    config = {"weights": checkpoint, "imgsz": imgsz, "device":
                              {"自动": "", "CPU": "cpu", "MPS": "mps", "CUDA 0": "0"}[device.get()]}
                    if action == "train":
                        yaml = Path(data.get().strip()).expanduser().resolve()
                        if not yaml.is_file() or yaml.suffix.lower() not in (".yaml", ".yml"):
                            raise ValueError("请先导出或选择训练 data.yaml。")
                        report = yaml.with_name("dataset_report.json")
                        if report.is_file() and not json.loads(report.read_text(encoding="utf-8"))["ready"]:
                            raise ValueError("训练集和验证集都需要带检测框的样本。请导入独立比赛的已标注图片后重新导出。")
                        count = int(epochs.get())
                        if count < 1:
                            raise ValueError("训练轮数必须是正整数。")
                        config.update(data=str(yaml), epochs=count)
                        self.yolo_data = yaml
                    else:
                        confidence = float(conf.get())
                        if not 0 < confidence <= 1:
                            raise ValueError("置信度需要大于 0、不超过 1。")
                        config.update(images=list(map(str, paths)), conf=confidence)
                    if action == "predict":
                        self.yolo_weights = Path(checkpoint)
            except (ValueError, OSError, KeyError) as exc:
                messagebox.showerror("检查 YOLO26 设置", str(exc), parent=window)
                return
            window.destroy()
            if action == "export":
                self.start_yolo_export(paths, names, include)
            else:
                self.start_yolo_worker(action, config)

        buttons = tk.Frame(window, bg=WHITE)
        buttons.grid(row=7, columnspan=3, sticky="ew")
        for label, action in (("导出数据集", "export"), ("开始训练", "train"), ("批量预标注", "predict")):
            self.button(buttons, label, lambda task=action: start(task), lock=False,
                        primary=action == "export").pack(side="left", padx=(0, 4))
        predictions = sorted((root / "predictions").glob("predict-*/preannotation.json"))
        if predictions:
            def load_latest():
                images = sorted((predictions[-1].parent / "images").glob("*.png"))
                window.destroy()
                self.add_paths(images)
                if images:
                    key = str(images[0].resolve())
                    self.table.selection_set(key)
                    self.table.see(key)
            self.button(buttons, "载入最近预标注", load_latest, lock=False).pack(side="right")

    def start_yolo_export(self, paths, names, include):
        metadata = self.metadata(paths)
        folder = self.output
        self.set_busy(True)
        self.progress.configure(mode="determinate", maximum=len(paths), value=0)
        self.status.set("正在核对图片和同名 TXT…")

        def work():
            try:
                def progress(done, total):
                    self.events.put(("progress", (done, total, done, 0, "")))
                result = export_dataset(paths, metadata, folder, names, include, self.stop, progress)
                self.events.put(("yolo_export", result))
            except Exception as exc:
                self.events.put(("failed", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def start_yolo_worker(self, action, config):
        root = self.output / "yolo26" / ("runs" if action == "train" else "predictions")
        run = run_folder(root, "run" if action == "train" else "predict")
        self.latest = run
        self.set_busy(True)
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start()
        self.status.set("YOLO26 正在训练…" if action == "train" else "YOLO26 正在批量预标注…")

        def work():
            try:
                run.mkdir(parents=True)
                config_path = run / "worker.json"
                atomic_json(config_path, {**config, "run": str(run)})
                logfile = run / ("train.log" if action == "train" else "predict.log")
                entry = [sys.executable] if FROZEN else [sys.executable, "-u", PROJECT / "app.py"]
                process = subprocess.Popen(
                    [*entry, "--yolo-worker", action, config_path, logfile],
                    cwd=run, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=NO_CONSOLE, env={**os.environ, "YOLO_AUTOINSTALL": "false"},
                )
                while process.poll() is None:
                    if self.stop.wait(0.25):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                if self.stop.is_set():
                    self.events.put(("yolo_worker", (action, run, None)))
                    return
                if process.returncode:
                    message = f"YOLO26 子进程退出（{process.returncode}）"
                    if logfile.is_file():
                        with logfile.open("rb") as logs:
                            logs.seek(max(0, logfile.stat().st_size - 3500))
                            tail = logs.read().decode("utf-8", "replace").splitlines()[-5:]
                        message = f"{logfile}\n\n" + "\n".join(tail)
                    self.events.put(("yolo_failure", (run, message)))
                    return
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                self.events.put(("yolo_worker", (action, run, result)))
            except Exception as exc:
                self.events.put(("failed", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def video_dialog(self):
        bundled = next((path for folder in (APP_DIR, PROJECT) if (path := folder / "ffmpeg.exe").is_file()),
                       None) if sys.platform == "win32" else None
        ffmpeg = str(bundled) if bundled else shutil.which("ffmpeg")
        if not ffmpeg:
            hint = "请把 ffmpeg.exe 放在程序旁边，或将 FFmpeg 加入 PATH。" if sys.platform == "win32" else (
                "请将 FFmpeg 加入 PATH 后使用视频取帧。")
            messagebox.showerror("缺少 FFmpeg", hint, parent=self)
            return
        file = filedialog.askopenfilename(parent=self, filetypes=[
            ("视频", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.ts"), ("所有文件", "*"),
        ])
        if not file:
            return
        window = self.dialog("视频取帧")
        self.label(window, Path(file).name, bold=True, size=14).grid(row=0, column=0, columnspan=2, sticky="w")
        fields = []
        for row, (title, default) in enumerate((("起点（秒）", "0"), ("取帧时长（秒）", "10"),
                                                     ("间隔（秒）", "0.5")), 1):
            self.label(window, title).grid(row=row, column=0, sticky="w", pady=4)
            field = ttk.Entry(window, width=15)
            field.insert(0, default)
            field.grid(row=row, column=1, sticky="e", padx=(8, 0))
            fields.append(field)
        self.label(window, "采集后可补充 YOLO 标注，再归档真实困难帧。", fg=MUTED).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 8))

        def start():
            try:
                start_time, duration, interval = (float(field.get()) for field in fields)
                if not all(map(math.isfinite, (start_time, duration, interval))) or start_time < 0 or duration <= 0 or interval <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("检查取帧参数", "起点应为非负数，时长和间隔应大于 0。", parent=window)
                return
            window.destroy()
            self.extract_video(ffmpeg, Path(file), start_time, duration, interval)

        self.button(window, "开始取帧", start, primary=True, lock=False).grid(row=5, column=0, columnspan=2, sticky="e")

    def extract_video(self, ffmpeg, video, start, duration, interval):
        def work():
            try:
                root = self.output / "captured"
                root.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix=video.stem[:48] + "-", dir=root))
                image_dir = folder / "images"
                image_dir.mkdir()
                video_id = str(video.resolve())
                prefix = folder.name.replace("%", "_")
                process = subprocess.Popen([
                    ffmpeg, "-nostdin", "-v", "error", "-ss", str(start), "-i", video,
                    "-t", str(duration), "-vf", f"fps={1/interval:.12g}:round=up",
                    "-start_number", "0", image_dir / (prefix + "_%06d.png"),
                ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    creationflags=NO_CONSOLE)
                while True:
                    try:
                        _, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop.is_set():
                            process.terminate()
                if process.returncode and not self.stop.is_set():
                    raise ValueError(stderr.decode("utf-8", "replace").strip() or "视频解码失败")
                entries = {}
                captured = []
                for image in sorted(image_dir.glob("*.png")):
                    try:
                        with Image.open(image) as check:
                            check.verify()
                    except OSError:
                        image.unlink(missing_ok=True)
                        continue
                    timestamp = start + int(image.stem.rsplit("_", 1)[1]) * interval
                    entries[image.name] = {"image_id": image.name, "source_id": video_id,
                                           "timestamp_ms": str(round(timestamp * 1000))}
                    captured.append(image)
                with (folder / "captures.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, ("image_id", "source_id", "timestamp_ms"))
                    writer.writeheader()
                    writer.writerows(entries.values())
                self.events.put(("captured", (captured, self.stop.is_set())))
            except Exception as exc:
                self.events.put(("failed", str(exc)))

        self.set_busy(True)
        self.status.set("正在从视频取帧…")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        threading.Thread(target=work, daemon=True).start()

    def set_busy(self, busy):
        self.busy = busy
        if busy:
            self.stop.clear()
        for widget in self.controls:
            widget.state(["disabled" if busy else "!disabled"])
        self.cancel_button.state(["!disabled" if busy else "disabled"])
        self.sync_controls()

    def start(self):
        real = self.mode.get() == "real"
        if real:
            selected = tuple(map(Path, self.table.selection()))
            if not selected:
                messagebox.showinfo("归档真实困难帧", "请在导入队列里选中要归档的图片。", parent=self)
                return
            kinds = (next(key for key, value in CLASSES.items() if value == self.primary.get()),)
        else:
            selected = tuple(map(Path, self.table.get_children()))
            kinds = self.selected_modes()
        settings = BatchSettings(
            folder=self.output, modes=kinds, levels=(0,) if real else self.selected_levels(),
            default_sport=self.default_sport, default_split=self.default_split,
            source_rows=self.source_rows_for(selected),
            name_rows=dict(self.name_rows), font_path=self.font_path,
            real=real, tags=tuple(kind for kind, var in (
                self.real_tags if real else self.extra_effects).items() if var.get()),
            readability=next(key for key, value in READABILITY.items() if value == self.readability.get()),
        )

        def work():
            try:
                def progress(*values):
                    self.events.put(("progress", values))
                self.events.put(("finished", run_batch(selected, settings, self.stop, progress)))
            except Exception as exc:
                self.events.put(("failed", str(exc)))

        self.progress.configure(mode="determinate", maximum=len(selected) * len(kinds) * len(settings.levels),
                                value=0)
        self.set_busy(True)
        self.status.set(f"正在处理 {len(selected):,} 张图片…")
        threading.Thread(target=work, daemon=True).start()

    def cancel(self):
        self.stop.set()
        self.cancel_button.state(["disabled"])
        self.status.set("正在停止，已完成结果会保留…")

    def poll_events(self):
        try:
            for _ in range(180):
                event, payload = self.events.get_nowait()
                if event == "preview":
                    token, before, after, info = payload
                    if token == self.preview_token:
                        self.preview_frames = before, after
                        self.summary.set(info)
                        for index in range(2):
                            self.draw_preview(index)
                elif event == "progress":
                    done, total, success, errors, _ = payload
                    self.progress.configure(value=done)
                    if not self.stop.is_set():
                        self.status.set(f"处理 {done:,} / {total:,}   ·   成功 {success:,}   ·   未导出 {errors:,}")
                elif event == "finished":
                    self.latest, manifest, _ = payload
                    self.set_busy(False)
                    count, errors = len(manifest["samples"]), len(manifest["errors"])
                    prefix = "已停止" if manifest["cancelled"] else "批次完成"
                    pending = sum(sample["export_split"] == "review" for sample in manifest["samples"])
                    details = f"   ·   待复核 {pending:,}" if pending else ""
                    self.status.set(f"{prefix}   ·   已导出 {count:,}   ·   未导出 {errors:,}{details}")
                    if errors:
                        self.show_batch_errors(manifest["errors"])
                    if self.closing:
                        self.destroy()
                        return
                elif event == "captured":
                    paths, cancelled = payload
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.add_paths(paths)
                    self.set_busy(False)
                    self.status.set(f"{'取帧已停止' if cancelled else '取帧完成'}   ·   已导入 {len(paths):,} 张")
                    if self.closing:
                        self.destroy()
                        return
                elif event in ("yolo_export", "yolo_worker", "yolo_failure"):
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.set_busy(False)
                    if self.closing:
                        self.destroy()
                        return
                    if event == "yolo_export":
                        if payload is None:
                            self.status.set("YOLO26 数据集导出已停止")
                            continue
                        self.latest, report = payload
                        self.yolo_data = self.latest / "data.yaml"
                        counts = report["counts"]
                        state = (f"训练 {counts['train']} · 验证 {counts['val']} · 测试 {counts['test']}"
                                 f" · 空标注 {report['background']} · 跳过 {len(report['skipped'])}")
                        if not report["ready"]:
                            state += " · 补充独立比赛的带框样本后可训练"
                        self.status.set(state)
                    elif event == "yolo_failure":
                        self.latest, message = payload
                        self.status.set("YOLO26 运行失败，请查看日志")
                        messagebox.showerror("YOLO26 运行失败", message, parent=self)
                    else:
                        action, self.latest, result = payload
                        if result is None:
                            self.status.set("YOLO26 已停止 · 日志已保留")
                        elif action == "train":
                            self.yolo_weights = Path(result["weights"])
                            self.status.set("YOLO26 训练完成 · 已保存 best.pt")
                        else:
                            self.status.set(f"预标注待复核 {result['written']:,} 张 · 未导出 {result['errors']:,} 张")
                elif event == "failed":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.set_busy(False)
                    self.status.set("处理失败")
                    if self.closing:
                        self.destroy()
                        return
                    messagebox.showerror("处理失败", payload, parent=self)
        except queue.Empty:
            pass
        self.after(80, self.poll_events)

    def show_batch_errors(self, errors):
        if self.closing:
            return
        window = self.dialog("未导出的项目")
        self.label(window, f"{len(errors):,} 项未导出，完整记录已写入本批 manifest.json。",
                   bold=True).pack(fill="x", pady=(0, 8))
        display = tk.Text(window, height=12, width=85, wrap="word", bg=WHITE, fg=INK, font=(FONT, 12))
        display.pack(fill="both", expand=True)
        for entry in errors[:60]:
            display.insert("end", f"{Path(entry['source_image']).name}  ·  {CLASSES[entry['primary_class']]}\n"
                           f"{entry['error']}\n\n")
        display.configure(state="disabled")
        self.button(window, "关闭", window.destroy, lock=False).pack(anchor="e", pady=(6, 0))

    def open_output(self):
        path = self.latest or self.output
        if not path.exists():
            messagebox.showinfo("打开输出", "生成、取帧或归档后会创建输出目录。", parent=self)
            return
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])

    def close(self):
        self.preview_pool.shutdown(wait=False, cancel_futures=True)
        if self.busy:
            self.closing = True
            self.cancel()
        else:
            self.destroy()


if __name__ == "__main__":
    freeze_support()
    if sys.argv[1:2] == ["--yolo-worker"]:
        raise SystemExit(run_worker_with_log(*sys.argv[2:]))
    App().mainloop()
