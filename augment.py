"""Deterministic scoreboard transformations and dataset export."""

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat

CLASSES = {
    "compression": "低清与压缩",
    "small_text": "小字与细笔画",
    "blur": "模糊与交错",
    "contrast": "低对比与透明底",
    "occlusion": "遮挡与边缘截断",
    "transition": "动画与状态转换",
    "multiple": "多牌与画中画",
    "names": "多语言与姓名复杂",
}
SPORTS = {"badminton": "羽毛球", "tennis": "网球"}
CLASSIFIED_INDEX = "分类清单.json"
EXTRA_EFFECTS = ("compression", "blur", "contrast")
SPLITS = {"unassigned": "待分配", "train": "训练集", "val": "验证集", "test": "测试集"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TARGET = 375
RESAMPLE = Image.Resampling
FONT_PATHS = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
)


@dataclass(frozen=True)
class Box:
    cls: int
    rect: tuple

    def remap(self, old, new):
        left, top, right, bottom = self.rect
        sx = (new[2] - new[0]) / (old[2] - old[0])
        sy = (new[3] - new[1]) / (old[3] - old[1])
        return Box(self.cls, (
            new[0] + (left - old[0]) * sx,
            new[1] + (top - old[1]) * sy,
            new[0] + (right - old[0]) * sx,
            new[1] + (bottom - old[1]) * sy,
        ))


def intersection(first, second):
    return (max(first[0], second[0]), max(first[1], second[1]),
            min(first[2], second[2]), min(first[3], second[3]))


def area(rect):
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def pixel_rect(rect):
    return math.floor(rect[0]), math.floor(rect[1]), math.ceil(rect[2]), math.ceil(rect[3])


def find_label(path):
    choices = [path.with_suffix(".txt")]
    for parent in path.parents:
        if parent.name.lower() == "images":
            relative = path.relative_to(parent).with_suffix(".txt")
            choices.append(parent.parent / "labels" / relative)
            break
    return next((candidate for candidate in choices if candidate.is_file()), None)


def load_frame(path):
    with Image.open(path) as image:
        label = find_label(path)
        if image.getexif().get(274, 1) != 1 and label:
            raise ValueError("带标注的图片含 EXIF 旋转，请先校准图像和标注坐标")
        frame = ImageOps.exif_transpose(image).convert("RGB")
    boxes = []
    if label:
        width, height = frame.size
        for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                kind, x, y, w, h = line.split()
                kind = int(kind)
                x, y, w, h = map(float, (x, y, w, h))
                if kind < 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError()
                rect = ((x - w / 2) * width, (y - h / 2) * height,
                        (x + w / 2) * width, (y + h / 2) * height)
                rect = intersection(rect, (0, 0, width, height))
                if not area(rect):
                    raise ValueError()
                boxes.append(Box(kind, rect))
            except ValueError as exc:
                raise ValueError(f"{label.name} 第 {number} 行不是有效的 YOLO 检测框") from exc
    return frame, boxes, label


def csv_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} 需要包含列 {', '.join(sorted(required))}")
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items() if key}


def read_sources(path):
    return {row["image_id"]: row for row in csv_rows(path, {"image_id"}) if row["image_id"]}


def read_names(path):
    names = {}
    for row in csv_rows(path, {"image_id", "row", "text"}):
        try:
            index = int(row["row"])
        except ValueError as exc:
            raise ValueError(f"{path.name} 的 row 必须从 1 开始编号") from exc
        key = row["image_id"], index
        if not key[0] or index < 1 or not row["text"] or key in names:
            raise ValueError(f"{path.name} 含空字段、重复姓名行或无效的 row")
        names[key] = row["text"]
    return names


def source_metadata(path, rows, default_sport, default_split):
    individual = rows.get(str(path), rows.get(path.name, rows.get(path.stem, {})))
    raw_sport = individual.get("sport", "") or path.name.split("_", 1)[0]
    if raw_sport.lower() in SPORTS:
        sport = raw_sport.lower()
    else:
        sport = next((key for key, name in SPORTS.items() if name == raw_sport), default_sport)
    raw_split = individual.get("split", "").lower()
    split_aliases = {value: key for key, value in SPLITS.items()}
    split_aliases.update({"training": "train", "valid": "val", "validation": "val"})
    split = raw_split if raw_split in SPLITS else split_aliases.get(raw_split)
    if split is None:
        parent_split = next((parent.name.lower() for parent in path.parents[:2]
                             if parent.name.lower() in ("train", "val", "test")), None)
        for parent in path.parents:
            if parent.name.lower() == "images":
                components = path.relative_to(parent).parts[:-1]
                split = next((name.lower() for name in components if name.lower() in ("train", "val", "test")),
                             parent_split)
                break
        split = split or parent_split or default_split
    return {
        "source_id": individual.get("source_id") or str(path.parent),
        "match_id": individual.get("match_id") or None,
        "timestamp_ms": individual.get("timestamp_ms") or None,
        "sport": sport,
        "split": split,
    }


def compatible_pair(first, second):
    return (first["sport"] == second["sport"]
            and first["split"] == second["split"]
            and first["source_id"] == second["source_id"]
            and first["match_id"] == second["match_id"])


def scoreboard(boxes):
    board = max((box for box in boxes if box.cls == 0), key=lambda box: area(box.rect), default=None)
    if board is None:
        raise ValueError("此项需要 YOLO 类别 0 的比分牌框")
    return board


def board_members(boxes, board):
    return [box for box in boxes if box == board or
            area(intersection(box.rect, board.rect)) >= area(box.rect) * 0.75]


def nearby_pixels(image, rect):
    left, top, right, bottom = rect
    width, height = image.size
    strip = image.crop((left, bottom, right, min(height, bottom + bottom - top)))
    if not strip.height:
        strip = image.crop((left, max(0, top - (bottom - top)), right, top))
    if not strip.height:
        return Image.new("RGB", (right - left, bottom - top), (57, 65, 70))
    return strip.resize((right - left, bottom - top), RESAMPLE.BILINEAR)


def resize_board(image, boxes, level, transition=False):
    board = scoreboard(boxes)
    members = board_members(boxes, board)
    rect = pixel_rect((min(box.rect[0] for box in members), min(box.rect[1] for box in members),
                       max(box.rect[2] for box in members), max(box.rect[3] for box in members)))
    left, top, right, bottom = rect
    width, height = right - left, bottom - top
    if transition:
        scale_x, scale_y, offset_x = ((0.99, 0.84, 0), (0.91, 0.70, 0.07), (0.69, 0.89, 0.25))[level]
        variant = ("局部跳动", "收起过渡", "换盘位置过渡")[level]
    else:
        scale_x, scale_y, offset_x = ((0.83, 0.86, 0.04), (0.65, 0.70, 0.04),
                                      (0.49, 0.56, 0.04))[level]
        variant = "缩小比分牌与细笔画"
    new_width, new_height = max(1, round(width * scale_x)), max(1, round(height * scale_y))
    x = min(right - new_width, left + round(width * offset_x))
    y = top + (height - new_height) // 2
    target = (x, y, x + new_width, y + new_height)
    if any(box not in members and area(intersection(box.rect, rect)) for box in boxes):
        raise ValueError("比分牌与其他标注相交，请复核原图")
    result = image.copy()
    result.paste(nearby_pixels(image, rect).filter(ImageFilter.GaussianBlur(2)), (left, top))
    result.paste(image.crop(rect).resize((new_width, new_height), RESAMPLE.BILINEAR), (x, y))
    return result, [box.remap(rect, target) if box in members else box for box in boxes], {
        "variant": variant, "scoreboard_scale": [scale_x, scale_y],
    }


def transition_frame(image, boxes, level):
    if level:
        return resize_board(image, boxes, level, transition=True)
    board = scoreboard(boxes)
    scores = [box for box in board_members(boxes, board) if box.cls in (2, 3)]
    for box in scores:
        rect = pixel_rect(box.rect)
        if any(other != box and other.cls != 0 and area(intersection(rect, other.rect)) for other in boxes):
            continue
        x, y, right, bottom = rect
        patch = image.crop(rect)
        height = max(1, round(patch.height * 0.8))
        if height == patch.height:
            continue
        target = (x, bottom - height, right, bottom)
        result = image.copy()
        result.paste(nearby_pixels(image, rect).filter(ImageFilter.GaussianBlur(2)), (x, y))
        result.paste(patch.resize((patch.width, height), RESAMPLE.BILINEAR), target[:2])
        return result, [item.remap(rect, target) if item == box else item for item in boxes], {
            "variant": "计分单元局部跳动", "cell": target,
        }
    return resize_board(image, boxes, level, transition=True)


def occlude(image, boxes, level):
    board = scoreboard(boxes)
    left, top, right, bottom = pixel_rect(board.rect)
    margin = (0.06, 0.11, 0.18)[level]
    dx = max(1, round((right - left) * margin))
    dy = max(1, round((bottom - top) * margin))
    vertical = [(left + dx, 0, image.width, image.height), (0, 0, right - dx, image.height)]
    horizontal = [(0, top + dy, image.width, image.height), (0, 0, image.width, bottom - dy)]
    if level == 2:
        safe_crops = [rect for rect in vertical + horizontal
                      if area(rect) >= image.width * image.height * 0.60 and
                      all(area(intersection(box.rect, rect)) >= area(box.rect) * 0.999
                          for box in boxes if box != board)]
        if safe_crops:
            crop = max(safe_crops, key=area)
            cropped = image.crop(crop)
            output_rect = (0, 0, cropped.width, cropped.height)
            return cropped, [
                Box(box.cls, intersection(box.rect, crop)).remap(crop, output_rect) for box in boxes
            ], {"variant": "边缘截断", "crop": crop}
    covers = [
        ((left, max(0, top - dy), right, top + dy), (left, top + dy, right, bottom)),
        ((left, bottom - dy, right, min(image.height, bottom + dy)), (left, top, right, bottom - dy)),
        ((max(0, left - dx), top, left + dx, bottom), (left + dx, top, right, bottom)),
        ((right - dx, top, min(image.width, right + dx), bottom), (left, top, right - dx, bottom)),
    ]
    safe_covers = [(cover, visible) for cover, visible in covers if area(visible) and
                   all(not area(intersection(box.rect, cover)) for box in boxes if box != board)]
    if not safe_covers:
        raise ValueError("没有可避开姓名和数字框的遮挡位置")
    cover, visible = safe_covers[0]
    result = image.copy()
    overlay = Image.new("RGB", (cover[2] - cover[0], cover[3] - cover[1]), (22, 29, 44))
    draw = ImageDraw.Draw(overlay)
    draw.line((0, 0, overlay.width - 1, 0), fill=(131, 162, 204), width=2)
    result.paste(overlay, cover[:2])
    updated = [Box(box.cls, intersection(board.rect, visible)) if box == board else box for box in boxes]
    return result, updated, {"variant": "字幕条或角标遮挡", "overlay": cover}


def composite(image, boxes, secondary, second_boxes, level):
    if not any(box.cls == 0 for box in boxes) or not any(box.cls == 0 for box in second_boxes):
        raise ValueError("画中画需要两张都有比分牌标注的截图")
    width, height = image.size

    def fitted(frame, labels, window):
        placement = ImageOps.contain(frame, (window[2] - window[0], window[3] - window[1]), RESAMPLE.LANCZOS)
        x = (window[0] + window[2] - placement.width) // 2
        y = (window[1] + window[3] - placement.height) // 2
        target = (x, y, x + placement.width, y + placement.height)
        return placement, [box.remap((0, 0, frame.width, frame.height), target) for box in labels], target

    if level < 2:
        inset_w, inset_h = max(1, width * (3 + level) // 10), max(1, height * (3 + level) // 10)
        gap = max(1, width // 45)
        windows = [(width - gap - inset_w, height - gap - inset_h), (gap, height - gap - inset_h),
                   (width - gap - inset_w, gap), (gap, gap)]
        for x, y in windows:
            window = (x, y, x + inset_w, y + inset_h)
            if x < 0 or y < 0 or any(area(intersection(box.rect, window)) for box in boxes):
                continue
            inset, mapped, target = fitted(secondary, second_boxes, window)
            result = image.copy()
            ImageDraw.Draw(result).rectangle((x, y, x + inset_w - 1, y + inset_h - 1), fill=(17, 22, 33))
            result.paste(inset, target[:2])
            return result, boxes + mapped, {"variant": "画中画", "inset": target}
    if width < 4:
        raise ValueError("图片太窄，无法安全生成双窗口")
    result = Image.new("RGB", image.size, (17, 22, 33))
    first, labels, target = fitted(image, boxes, (0, 0, width // 2, height))
    second, mapped, other_target = fitted(secondary, second_boxes, (width // 2 + 1, 0, width, height))
    result.paste(first, target[:2])
    result.paste(second, other_target[:2])
    return result, labels + mapped, {"variant": "双窗口", "windows": [target, other_target]}


def name_font(text, font_path):
    options = (font_path,) if font_path else FONT_PATHS
    characters = {char for char in text if unicodedata.category(char)[0] not in ("C", "Z")}
    for option in options:
        if not Path(option).is_file():
            continue
        font = ImageFont.truetype(option, size=40)
        missing = bytes(font.getmask(chr(0x10FFFF)))
        if all(bytes(font.getmask(char)) != missing for char in characters):
            return font
    raise ValueError("姓名含当前字体不支持的字符，请在数据来源里选择覆盖该语言的字体")


def complex_names(image, boxes, level, names, font_path):
    name_boxes = sorted((box for box in boxes if box.cls == 1), key=lambda box: (box.rect[1], box.rect[0]))
    if not name_boxes:
        raise ValueError("此项需要 YOLO 类别 1 的姓名框")
    if any(index > len(name_boxes) for index in names):
        raise ValueError("姓名真值 CSV 的行号超过当前图片的姓名框数量")
    result = image.copy()
    changed = {}
    rendered_names = []
    for index, box in enumerate(name_boxes, 1):
        rect = pixel_rect(box.rect)
        left, top, right, bottom = rect
        patch = image.crop(rect)
        if any(other != box and other.cls != 0 and area(intersection(other.rect, rect)) for other in boxes):
            raise ValueError("姓名框与其他文字框相交，请复核原标注")
        background = tuple(round(value) for value in ImageStat.Stat(
            patch.resize((1, 1), RESAMPLE.BOX)
        ).mean)
        result.paste(background, rect)
        text = names.get(index)
        if text:
            font = name_font(text, font_path)
            mask_rect = font.getbbox(text)
            if mask_rect[2] <= mask_rect[0] or mask_rect[3] <= mask_rect[1]:
                raise ValueError("姓名真值无法渲染")
            mask = Image.new("L", (mask_rect[2] - mask_rect[0], mask_rect[3] - mask_rect[1]))
            ImageDraw.Draw(mask).text((-mask_rect[0], -mask_rect[1]), text, font=font, fill=255)
            scale = min(patch.width / mask.width, patch.height / mask.height)
            size = (max(1, round(mask.width * scale)), max(1, round(mask.height * scale)))
            mask = mask.resize(size, RESAMPLE.LANCZOS)
            color = (250, 250, 250) if max(background) < 160 else (19, 23, 30)
            y = top + (patch.height - size[1]) // 2
            result.paste(color, (left, y, left + size[0], y + size[1]), mask)
            rendered_names.append({"row": index, "text": text})
        else:
            narrow = (0.86, 0.74, 0.66)[level]
            size = (max(1, int(patch.width * narrow)), max(1, int(patch.height * (0.95 - level * 0.05))))
            y = top + min(patch.height - size[1], (index - 1) % 3)
            result.paste(patch.resize(size, RESAMPLE.BILINEAR), (left, y))
        changed[box] = Box(box.cls, (left, y, left + size[0], y + size[1]))
    return result, [changed.get(box, box) for box in boxes], {
        "variant": "按真值渲染姓名" if rendered_names else "保留文字的窄体排版",
        "ocr_names": rendered_names,
    }


def transform(image, boxes, kind, level, secondary=None, names=None, font_path=None):
    image = image.copy()
    if kind == "small_text":
        return resize_board(image, boxes, level)
    if kind == "transition":
        return transition_frame(image, boxes, level)
    if kind == "occlusion":
        return occlude(image, boxes, level)
    if kind == "names":
        return complex_names(image, boxes, level, names or {}, font_path)
    if kind == "multiple":
        if secondary is None:
            raise ValueError("请导入同来源、同数据划分的另一张带比分牌标注的截图")
        return composite(image, boxes, *secondary, level)
    if kind == "compression":
        target_height = (480, 360, 360)[level]
        quality = (52, 30, 17)[level]
        size = (max(1, round(image.width * min(1, target_height / image.height))),
                min(image.height, target_height))
        buffer = BytesIO()
        image.resize(size, RESAMPLE.BILINEAR).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.resize(image.size, RESAMPLE.BILINEAR)
        return image, boxes, {"source_height": size[1], "jpeg_quality": quality}
    if kind == "blur":
        size = (3, 5, 5)[level]
        kernel = [0] * (size * size)
        kernel[size * (size // 2):size * (size // 2 + 1)] = [1] * size
        image = image.filter(ImageFilter.Kernel((size, size), kernel))
        image = image.filter(ImageFilter.GaussianBlur((0.45, 0.95, 1.45)[level]))
        if level:
            shifted = Image.new("RGB", image.size)
            offset = level
            if image.width > offset:
                shifted.paste(image.crop((0, 0, image.width - offset, image.height)), (offset, 0))
            stripe = Image.new("L", (1, image.height))
            stripe.putdata([90 if row % 2 else 0 for row in range(image.height)])
            stripe = stripe.resize(image.size, RESAMPLE.NEAREST)
            ImageDraw.Draw(stripe).rectangle((0, 0, offset - 1, image.height - 1), fill=0)
            image.paste(shifted, (0, 0), stripe)
        return image, boxes, {"motion_kernel": size, "interlace_shift": level}
    contrast, brightness = ((0.78, 1.06), (0.56, 0.80), (0.45, 1.17))[level]
    image = ImageEnhance.Brightness(ImageEnhance.Contrast(image).enhance(contrast)).enhance(brightness)
    boards = [box for box in boxes if box.cls == 0]
    alpha = (0.14, 0.24, 0.32)[level] if boards else 0
    for board in boards:
        rect = pixel_rect(board.rect)
        backdrop = nearby_pixels(image, rect).filter(ImageFilter.GaussianBlur(2))
        image.paste(Image.blend(image.crop(rect), backdrop, alpha), rect[:2])
    return image, boxes, {"contrast": contrast, "brightness": brightness, "background_blend": alpha}


def transform_sample(image, boxes, kind, level, secondary, names, font_path, extras):
    result, output_boxes, params = transform(image, boxes, kind, level, secondary, names, font_path)
    additional = {}
    for extra in extras:
        if extra != kind and extra in EXTRA_EFFECTS:
            result, output_boxes, details = transform(result, output_boxes, extra, 0)
            additional[extra] = details
    if additional:
        params["additional_effects"] = additional
    return result, output_boxes, params


@dataclass(frozen=True)
class BatchSettings:
    folder: Path
    modes: tuple
    levels: tuple
    default_sport: str
    default_split: str
    source_rows: dict
    name_rows: dict
    font_path: str
    real: bool = False
    tags: tuple = ()
    readability: str = "readable"


def is_prediction(path):
    return any(parent.name.startswith("predict-") and
               ((parent / "preannotation.json").is_file() or (parent / "worker.json").is_file())
               for parent in path.parents)


def is_batch_export(path):
    return any((parent.name.startswith("batch-") and (parent / "manifest.json").is_file())
               or (parent / CLASSIFIED_INDEX).is_file()
               for parent in path.parents)


def open_input_frame(path):
    if is_prediction(path):
        raise ValueError("请导入使用人工复核标注的原始截图，当前图片来自模型预标注")
    if is_batch_export(path):
        raise ValueError("请导入原始截图或视频采集帧，当前文件来自已导出的批次")
    return load_frame(path)


def names_for(path, rows):
    matches = {index: value for (name, index), value in rows.items() if name == path.stem}
    matches.update({index: value for (name, index), value in rows.items() if name == path.name})
    return matches


def find_secondary(path, metadata):
    for other, other_metadata in metadata.items():
        if other == path or not compatible_pair(metadata[path], other_metadata):
            continue
        try:
            frame, boxes, _ = open_input_frame(other)
            if any(box.cls == 0 for box in boxes):
                return other, (frame, boxes)
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    return None, None


def format_labels(boxes, size):
    width, height = size
    lines = []
    for box in boxes:
        left, top, right, bottom = intersection(box.rect, (0, 0, width, height))
        if right <= left or bottom <= top:
            raise ValueError("变换后有标注框不可见，请将该帧转入复核")
        coords = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
        lines.append(f"{box.cls} " + " ".join(f"{value:.9f}" for value in coords) + "\n")
    return "".join(lines)


def file_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def atomic_json(file, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=file.parent, encoding="utf-8",
                                     prefix=".json-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, file)
    finally:
        temp.unlink(missing_ok=True)


def quotas(folder):
    real, synthetic, review = Counter(), Counter(), Counter()
    seen = set()
    for kind, records in (("real", real), ("synthetic", synthetic)):
        for manifest in sorted((folder / kind).glob("batch-*/manifest.json")):
            with manifest.open(encoding="utf-8") as stream:
                entries = json.load(stream)["samples"]
            for entry in entries:
                key = entry["sport"], entry["primary_class"]
                if kind == "real":
                    if entry["review_status"] != "confirmed":
                        review[key] += 1
                        continue
                    digest = entry["source_sha256"]
                    if digest in seen:
                        continue
                    seen.add(digest)
                records[key] += 1
    rows = [
        {"sport": sport, "primary_class": kind, "target_real": TARGET, "real": real[sport, kind],
         "remaining_real": max(0, TARGET - real[sport, kind]),
         "synthetic": synthetic[sport, kind], "pending_real_review": review[sport, kind]}
        for kind in CLASSES for sport in SPORTS
    ]
    return {"target_real": TARGET * len(rows), "real": sum(real.values()),
            "credited_real": sum(min(row["target_real"], row["real"]) for row in rows),
            "synthetic": sum(synthetic.values()), "rows": rows}


def run_batch(paths, settings, stop, progress):
    metadata = {path: source_metadata(path, settings.source_rows,
                                      settings.default_sport, settings.default_split) for path in paths}
    jobs = [(kind, level) for kind in settings.modes for level in settings.levels]
    total = len(paths) * len(jobs)
    parent = settings.folder / ("real" if settings.real else "synthetic")
    parent.mkdir(parents=True, exist_ok=True)
    batch_id = "batch-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stage = Path(tempfile.mkdtemp(dir=parent, prefix=".working-"))
    samples, errors = [], []
    done = 0
    try:
        for source_index, (path, item) in enumerate(metadata.items(), 1):
            if stop.is_set():
                break
            try:
                if not settings.real and item["split"] in ("val", "test"):
                    raise ValueError("验证集或测试集的图片不能派生训练样本")
                frame, boxes, original_label = open_input_frame(path)
                digest = file_digest(path)
                input_error = None
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                input_error = str(exc)

            paired_path, paired = None, None
            if not input_error and not settings.real and "multiple" in settings.modes:
                paired_path, paired = find_secondary(path, metadata)
            source_names = names_for(path, settings.name_rows)
            for kind, level in jobs:
                if stop.is_set():
                    break
                try:
                    if input_error:
                        raise ValueError(input_error)
                    if settings.real:
                        result, output_boxes, params = frame, boxes, {}
                    else:
                        result, output_boxes, params = transform_sample(
                            frame, boxes, kind, level, paired, source_names, settings.font_path, settings.tags)
                    known_labels = original_label is not None
                    confirmed = (settings.real and known_labels and
                                 settings.readability != "review" and any(box.cls == 0 for box in boxes))
                    needs_review = (not known_labels or
                                    (settings.real and not confirmed))
                    split = "review" if needs_review else item["split"]
                    filename = f"{source_index:05d}_{path.stem[:100]}"
                    if not settings.real:
                        filename += f"_{kind}_{level + 1}"
                    suffix = path.suffix.lower() if settings.real else ".png"
                    subfolder = Path(split) if settings.real else Path(split, CLASSES[kind])
                    image_path = Path("images", subfolder, filename + suffix)
                    label_path = Path("labels", subfolder, filename + ".txt") if known_labels else None
                    label_text = format_labels(output_boxes, result.size) if label_path else None
                    (stage / image_path.parent).mkdir(parents=True, exist_ok=True)
                    if label_path:
                        (stage / label_path.parent).mkdir(parents=True, exist_ok=True)
                    try:
                        if settings.real:
                            shutil.copyfile(path, stage / image_path)
                        else:
                            result.save(stage / image_path)
                        if label_path:
                            (stage / label_path).write_text(label_text, encoding="utf-8")
                    except OSError:
                        (stage / image_path).unlink(missing_ok=True)
                        if label_path:
                            (stage / label_path).unlink(missing_ok=True)
                        raise
                    record = {
                        **item,
                        "source_image": str(path),
                        "source_label": str(original_label) if original_label else None,
                        "source_sha256": digest,
                        "image": image_path.as_posix(),
                        "label": label_path.as_posix() if label_path else None,
                        "size": list(result.size),
                        "primary_class": kind,
                        "difficulty_tags": list(dict.fromkeys((kind, *settings.tags))),
                        "is_synthetic": not settings.real,
                        "export_split": split,
                        "readability": settings.readability if settings.real else "unverified",
                        "review_status": "confirmed" if confirmed else "pending",
                        "transform_params": {} if settings.real else {"level": level + 1, **params},
                    }
                    if kind == "multiple" and not settings.real:
                        record["secondary_source"] = {
                            **metadata[paired_path], "source_image": str(paired_path)
                        }
                    samples.append(record)
                except (OSError, ValueError) as exc:
                    errors.append({"source_image": str(path), "primary_class": kind,
                                   "level": None if settings.real else level + 1, "error": str(exc)})
                done += 1
                progress(done, total, len(samples), len(errors), path.name)

        manifest = {
            "batch_id": batch_id, "kind": "real" if settings.real else "synthetic",
            "cancelled": stop.is_set(), "attempted": done,
            "planned": total, "samples": samples, "errors": errors,
        }
        atomic_json(stage / "manifest.json", manifest)
        destination = parent / batch_id
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    report = quotas(settings.folder)
    atomic_json(settings.folder / "quota_report.json", report)
    return destination, manifest, report
