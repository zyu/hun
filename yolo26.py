"""Package paired scoreboard labels and run Ultralytics detection in a worker process."""

import json
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from augment import atomic_json, file_digest, find_label, format_labels, is_batch_export, is_prediction, load_frame

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_NAMES = ("scoreboard", "player_name", "class_2", "class_3")


def run_folder(root, prefix):
    return root / (prefix + datetime.now().strftime("-%Y%m%d-%H%M%S-%f"))


def group_for(item):
    return item["match_id"] or item["source_id"]


def split_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[group_for(record["metadata"])].append(record)

    assignments = {}
    undecided = []
    for group, items in sorted(groups.items()):
        existing = {item["metadata"]["split"] for item in items} - {"unassigned"}
        if len(existing) > 1:
            raise ValueError(f"同一比赛来源有多个数据划分：{group}")
        if existing:
            assignments[group] = existing.pop()
        else:
            undecided.append(group)

    for split in ("train", "val"):
        if split not in assignments.values() and undecided:
            positive_groups = [key for key in undecided if any(item["count"] for item in groups[key])]
            group = (max if split == "train" else min)(positive_groups or undecided,
                                                       key=lambda key: len(groups[key]))
            assignments[group] = split
            undecided.remove(group)

    goal = max(1, round(len(records) * 0.2))
    val_count = sum(len(groups[key]) for key, split in assignments.items() if split == "val")
    for group in sorted(undecided, key=lambda key: len(groups[key])):
        split = "val" if 0 < val_count and val_count + len(groups[group]) <= goal else "train"
        assignments[group] = split
        if split == "val":
            val_count += len(groups[group])
    return assignments


def training_variants(folder, originals):
    train = {str(item["path"]): item["digest"] for item in originals if item["split"] == "train"}
    for manifest in sorted((folder / "synthetic").glob("batch-*/manifest.json")):
        entries = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
        for item in entries:
            if (item["split"] != "train" or item["export_split"] != "train" or not item["label"]
                    or train.get(item["source_image"]) != item["source_sha256"]):
                continue
            extra = item.get("secondary_source")
            if extra and extra["source_image"] not in train:
                continue
            yield manifest.parent / item["image"], item["source_image"]


def export_dataset(paths, metadata, folder, names, include_synthetic, stop, progress):
    records, skipped = [], []
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("请按类别编号填写互不重复的类别名")
    for number, path in enumerate(paths, 1):
        if stop.is_set():
            return None
        try:
            if is_prediction(path):
                raise ValueError("模型预标注待复核，请导入人工确认后的原始样本")
            if is_batch_export(path):
                raise ValueError("请导入原始截图。已导出的增强图可通过相应选项加入训练集")
            if not find_label(path):
                raise ValueError("缺少同名 TXT")
            image, boxes, _ = load_frame(path)
            if any(box.cls >= len(names) for box in boxes):
                raise ValueError(f"发现超出类别列表的编号，请核对 {path.name} 的类别定义")
            records.append({"path": path, "digest": file_digest(path), "metadata": metadata[path],
                            "labels": format_labels(boxes, image.size), "count": len(boxes)})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            skipped.append({"image": str(path), "reason": str(exc)})
        progress(number, len(paths))
    if not records:
        first_reason = f"（{skipped[0]['reason']}）" if skipped else ""
        raise ValueError(f"队列里没有可导出的 YOLO 图片/标注配对{first_reason}")
    assignments = split_groups(records)
    seen = {}
    for item in records:
        item["split"] = assignments[group_for(item["metadata"])]
        previous = seen.setdefault(item["digest"], item["split"])
        if previous != item["split"]:
            raise ValueError(f"相同图片分到了多个数据划分：{item['path'].name}")

    root = folder / "yolo26" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    destination = run_folder(root, "dataset")
    counts = dict.fromkeys(SPLIT_NAMES, 0)
    positives = dict.fromkeys(SPLIT_NAMES, 0)
    background, augmented = 0, 0
    manifest = []
    with TemporaryDirectory(dir=root, prefix=".building-") as temp:
        stage = Path(temp) / destination.name
        for split in SPLIT_NAMES:
            (stage / "images" / split).mkdir(parents=True)
            (stage / "labels" / split).mkdir(parents=True)

        def write(path, split, labels, count, source_path, synthetic):
            name = f"{len(manifest) + 1:06d}_{path.stem[:100]}"
            target = Path("images", split, name + path.suffix.lower())
            shutil.copyfile(path, stage / target)
            (stage / "labels" / split / (name + ".txt")).write_text(labels, encoding="utf-8")
            manifest.append({"source_image": str(source_path), "image": target.as_posix(),
                             "split": split, "boxes": count, "synthetic": synthetic})
            counts[split] += 1
            positives[split] += bool(count)

        for item in records:
            if stop.is_set():
                return None
            write(item["path"], item["split"], item["labels"], item["count"], item["path"], False)
            background += item["count"] == 0

        if include_synthetic:
            for variant, source in training_variants(folder, records):
                if stop.is_set():
                    return None
                try:
                    image, boxes, label = load_frame(variant)
                    if not label or any(box.cls >= len(names) for box in boxes):
                        raise ValueError("增强样本类别或同名 TXT 无效")
                    write(variant, "train", format_labels(boxes, image.size), len(boxes), source, True)
                    augmented += 1
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    skipped.append({"image": str(variant), "reason": str(exc)})

        report = {"counts": counts, "positives": positives, "background": background,
                  "augmented": augmented, "skipped": skipped, "groups": assignments,
                  "ready": bool(positives["train"] and positives["val"]), "samples": manifest}
        yaml = [
            f"path: {json.dumps(str(destination), ensure_ascii=False)}",
            "train: images/train", "val: images/val",
        ]
        if counts["test"]:
            yaml.append("test: images/test")
        yaml.append("names:")
        yaml.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
        (stage / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
        atomic_json(stage / "dataset_report.json", report)
        stage.rename(destination)
    return destination, report


def model_worker(action, config):
    from ultralytics import YOLO

    run = Path(config["run"])
    model = YOLO(config["weights"])
    if model.task != "detect":
        raise ValueError("请选择 YOLO 检测模型的权重")
    options = {"device": config["device"]} if config["device"] else {}
    if action == "train":
        model.train(data=config["data"], epochs=config["epochs"], imgsz=config["imgsz"], workers=0,
                    fliplr=0, flipud=0, mosaic=0, mixup=0, cutmix=0, copy_paste=0,
                    degrees=0, translate=0, scale=0, shear=0, perspective=0,
                    project=str(run), name="train", exist_ok=True, **options)
        weights = run / "train" / "weights" / "best.pt"
        if not weights.is_file():
            raise RuntimeError("训练结束，但未找到最佳权重，请查看训练日志")
        atomic_json(run / "result.json", {"weights": str(weights)})
        return

    (run / "images").mkdir()
    (run / "labels").mkdir()
    samples, errors, used = [], [], set()
    for number, source in enumerate(config["images"], 1):
        path = Path(source)
        try:
            with Image.open(path) as image:
                frame = ImageOps.exif_transpose(image).convert("RGB")
            result = model.predict(frame, imgsz=config["imgsz"], conf=config["conf"], verbose=False, **options)[0]
            names = result.names
            detections = [
                {"class_id": int(kind), "name": names[int(kind)], "xywhn": coords, "confidence": score}
                for coords, kind, score in zip(result.boxes.xywhn.cpu().tolist(),
                                               result.boxes.cls.cpu().tolist(),
                                               result.boxes.conf.cpu().tolist())
            ]
            stem = path.stem
            while stem.casefold() in used:
                stem += f"_{number:06d}"
            target = run / "images" / (stem + ".png")
            frame.save(target)
            label = run / "labels" / (stem + ".txt")
            label.write_text("".join(
                f"{item['class_id']} " + " ".join(f"{value:.9f}" for value in item["xywhn"]) + "\n"
                for item in detections
            ), encoding="utf-8")
            samples.append({"source_image": source, "image": target.name, "label": label.name,
                            "detections": detections, "review_status": "pending"})
            used.add(stem.casefold())
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({"image": source, "reason": str(exc)})
        print(f"PROGRESS {number}/{len(config['images'])}", flush=True)
    summary = {"weights": config["weights"], "samples": samples, "errors": errors}
    atomic_json(run / "preannotation.json", summary)
    atomic_json(run / "result.json", {"written": len(samples), "errors": len(errors)})


def run_worker_with_log(action, config_path, log_path):
    with Path(log_path).open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                config = json.loads(Path(config_path).read_text(encoding="utf-8"))
                model_worker(action, config)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    command, config_file = sys.argv[1:]
    model_worker(command, json.loads(Path(config_file).read_text(encoding="utf-8")))
