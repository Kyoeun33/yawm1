"""2단계: 정제 -> 그룹 단위 분할 -> 전처리 저장

    python 01-02_prepare.py --input data/raw --output data/processed --mode group
    python 01-02_prepare.py --input data/raw --output data/processed_random --mode random   # 누수 비교용

흐름
  data/raw (폴더 이름이 라벨: Closed/Open/no_yawn/yawn, 기존 train/test 구분은 무시)
    1) 읽기 검사 + 해시 계산
    2) 완전 중복 제거 (같은 이미지가 다른 라벨이면 둘 다 제거)
    3) 거의 같은 이미지를 그룹으로 묶음
    4) 그룹 단위로 train/val/test 분할 (mode=group) / 이미지 단위 (mode=random, 누수 있는 방식)
    5) 224x224 패딩 전처리 후 저장 + manifest.csv + 라벨 확인용 샘플 시트
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import random
import shutil

from PIL import Image, ImageDraw, UnidentifiedImageError

from common import CLASS_MAP, EXT, md5_file, dhash, near_groups, to_square

TASKS = ("eyes", "yawn")
SPLITS = ("train", "val", "test")


def find_class(path, root):
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in CLASS_MAP:
            return CLASS_MAP[part.lower()]
    return None


def assign_split(items, mode, rng, ratios=(0.70, 0.15)):
    """mode=group: 같은 그룹은 한 split에만 / mode=random: 이미지마다 무작위"""
    if mode == "random":
        units = [[it] for it in items]
    else:
        by_group = defaultdict(list)
        for it in items:
            by_group[it["group"]].append(it)
        units = list(by_group.values())
    rng.shuffle(units)

    n = len(items)
    t1, t2 = n * ratios[0], n * (ratios[0] + ratios[1])
    seen = 0
    for unit in units:
        s = "train" if seen < t1 else "val" if seen < t2 else "test"
        for it in unit:
            it["split"] = s
        seen += len(unit)


def make_sheet(out, per=3):
    """라벨이 맞는지 눈으로 확인하는 샘플 시트"""
    samples = []
    for task in TASKS:
        for split in SPLITS:
            for ld in sorted((out / task / split).glob("*")):
                if ld.is_dir():
                    samples += [(p, f"{task}/{split}/{ld.name}") for p in sorted(ld.glob("*.jpg"))[:per]]
    if not samples:
        return None
    cols, tw, th, lh = 6, 150, 150, 18
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + lh)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, cap) in enumerate(samples):
        with Image.open(p) as im:
            im.thumbnail((tw - 6, th - 6))
            x, y = (i % cols) * tw + 3, (i // cols) * (th + lh)
            sheet.paste(im, (x, y))
        d.text((x, y + th), cap, fill="black")
    path = out / "label_review_samples.jpg"
    sheet.save(path, quality=90)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--mode", choices=["group", "random"], default="group")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--near", type=int, default=5, help="유사 이미지 기준(해밍거리, 최대 5)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root, out = Path(a.input).resolve(), Path(a.output).resolve()
    if not root.exists():
        raise SystemExit(f"[오류] 입력 폴더가 없습니다: {root}")
    rng = random.Random(a.seed)

    # 1) 읽기 검사 + 해시
    items, bad = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        cls = find_class(p, root)
        if cls is None:
            continue
        try:
            with Image.open(p) as im:
                im.load()
                h = dhash(im)
            items.append({"path": p, "task": cls[0], "label": cls[1], "md5": md5_file(p), "dhash": h})
        except (OSError, ValueError, UnidentifiedImageError) as e:
            bad.append(f"{p} ({e})")
    if not items:
        raise SystemExit("[오류] Closed/Open/no_yawn/yawn 폴더에서 이미지를 찾지 못했습니다.")

    # 2) 완전 중복 제거
    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it)
    kept, dup, conflict = [], 0, 0
    for g in by_md5.values():
        if len({(x["task"], x["label"]) for x in g}) > 1:
            conflict += len(g)  # 같은 이미지인데 라벨이 다름 -> 믿을 수 없으니 제거
            continue
        kept.append(g[0])
        dup += len(g) - 1

    # 3~5) 작업별 그룹화 -> 분할 -> 저장
    for t in TASKS:
        shutil.rmtree(out / t, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(Counter)
    rows, biggest = [], {}
    for task in TASKS:
        ti = [x for x in kept if x["task"] == task]
        if not ti:
            continue
        gids = near_groups([x["dhash"] for x in ti], a.near)
        for x, g in zip(ti, gids):
            x["group"] = g
        biggest[task] = max(Counter(gids).values())
        assign_split(ti, a.mode, rng)

        for x in ti:
            d = out / task / x["split"] / x["label"]
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{x['path'].stem}_{x['md5'][:8]}.jpg"
            with Image.open(x["path"]) as im:
                to_square(im, a.imgsz).save(dest, "JPEG", quality=95)
            counts[(task, x["split"])][x["label"]] += 1
            rows.append([task, x["split"], x["label"], x["group"], str(x["path"]), str(dest)])

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task", "split", "label", "group", "source", "output"])
        w.writerows(rows)
    sheet = make_sheet(out)

    # 결과 요약
    print("\n========== 전처리 결과 ==========")
    print(f"분할 방식: {a.mode}  (group=누수 방지, random=누수 가능)")
    print(f"읽은 이미지 {len(items)}장 / 읽기 오류 {len(bad)}장 / 완전 중복 제거 {dup}장 / 라벨 충돌 제거 {conflict}장")
    for task in TASKS:
        if task in biggest:
            print(f"[{task}] 가장 큰 유사 그룹: {biggest[task]}장")
    for task in TASKS:
        print(f"\n[{task}]")
        for s in SPLITS:
            c = counts[(task, s)]
            print(f"  {s:5s}: {dict(c)} (총 {sum(c.values())}장)")
            if task in biggest and not c:
                print(f"  [경고] {task}/{s}가 비어 있습니다. 데이터가 너무 적거나 그룹이 한쪽에 몰렸습니다.")
    print(f"\n출력: {out}")
    if sheet:
        print(f"라벨 확인용 샘플: {sheet}  <- 꼭 열어서 폴더 라벨과 맞는지 눈으로 확인")
    for line in bad[:10]:
        print(" 읽기 오류:", line)


if __name__ == "__main__":
    main()
