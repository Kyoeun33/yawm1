"""Run the trained yawn classifier over Simuletic_DMS_Dataset/images.

Writes compact prediction metadata and review thumbnails under reports/.work.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from yawn_common import (
    DEFAULT_FACE_MODEL,
    DEFAULT_WEIGHTS,
    CropConfig,
    MouthExtractor,
    choose_device,
    load_checkpoint,
    make_transform,
    project_path,
    verify_face_model,
)


@torch.inference_mode()
def main() -> None:
    project = Path(__file__).resolve().parent
    source = project.parent / "Simuletic_DMS_Dataset" / "images"
    output = project / "reports" / ".simuletic_dms_work"
    output.mkdir(parents=True, exist_ok=True)
    originals = output / "originals"
    crops = output / "mouth_crops"
    originals.mkdir(exist_ok=True)
    crops.mkdir(exist_ok=True)

    device = choose_device("cpu")
    weights = project_path(DEFAULT_WEIGHTS)
    model, checkpoint = load_checkpoint(weights, device)
    threshold = float(checkpoint["threshold"])
    face_path = verify_face_model(project_path(DEFAULT_FACE_MODEL), checkpoint)
    transform = make_transform()
    config = CropConfig(**checkpoint["crop_config"])
    paths = sorted(
        (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
        key=lambda p: p.name.casefold(),
    )
    rows = []
    with MouthExtractor(face_path, config) as extractor:
        for index, path in enumerate(paths, start=1):
            row = {
                "file": path.name,
                "source": str(path.resolve()),
                "status": "",
                "prediction": "판정 불가",
                "p_yawn": None,
                "threshold": threshold,
                "original_thumb": None,
                "mouth_thumb": None,
                "face_count": 0,
            }
            try:
                with Image.open(path) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
                    result = extractor.extract(original)
                row["status"] = result["status"]
                row["face_count"] = result["face_count"]

                orig_thumb_path = originals / f"{index:04d}.jpg"
                ImageOps.fit(original, (150, 100), method=Image.Resampling.LANCZOS).save(orig_thumb_path, quality=82)
                row["original_thumb"] = str(orig_thumb_path.resolve())

                if result["status"] == "ok":
                    x = transform(result["crop"]).unsqueeze(0).to(device)
                    probability = float(model(x).softmax(1)[0, 1])
                    row["p_yawn"] = probability
                    row["prediction"] = "하품" if probability >= threshold else "하품 아님"
                    crop_thumb_path = crops / f"{index:04d}.jpg"
                    result["crop"].resize((100, 100), Image.Resampling.LANCZOS).save(crop_thumb_path, quality=85)
                    row["mouth_thumb"] = str(crop_thumb_path.resolve())
                else:
                    row["mouth_thumb"] = None
            except Exception as exc:
                row["status"] = f"오류: {type(exc).__name__}: {exc}"
            rows.append(row)
            if index % 20 == 0 or index == len(paths):
                print(f"{index}/{len(paths)} 처리 완료")

    metadata = {
        "input_dir": str(source.resolve()),
        "weights": str(weights.resolve()),
        "threshold": threshold,
        "count": len(rows),
        "rows": rows,
    }
    (output / "predictions.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = [row for row in rows if row["p_yawn"] is not None]
    print(
        f"끝: 전체 {len(rows)}장, 판정 가능 {len(valid)}장, "
        f"하품 {sum(r['prediction'] == '하품' for r in valid)}장, "
        f"하품 아님 {sum(r['prediction'] == '하품 아님' for r in valid)}장, "
        f"판정 불가 {len(rows) - len(valid)}장"
    )


if __name__ == "__main__":
    main()
