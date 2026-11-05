"""1단계: 데이터 감사 (audit)
- 폴더 구조 / 클래스별 개수 / 손상 파일 / 이미지 크기·형식
- 완전 중복(MD5) + 거의 같은 이미지(dHash) 검사
- train/val/test 사이 누수(leakage) 추정

사용법:
    pip install pillow
    python 01-01_audit_data.py --data data
결과:
    audit_out/audit_report.csv      (이미지별 정보)
    audit_out/near_dup_groups.csv   (유사 이미지 그룹)
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib

from PIL import Image, UnidentifiedImageError

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
HASH_BITS = 64
CHUNK_BOUNDS = [0, 11, 22, 33, 44, 54, 64]  # 6조각: 해밍거리<=5이면 최소 한 조각은 동일


def md5_file(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(im, size=8):
    g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            i = r * (size + 1) + c
            bits = (bits << 1) | int(px[i] > px[i + 1])
    return bits


def detect_split(path, root):
    try:
        parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    except ValueError:
        parts = [p.lower() for p in path.parts[:-1]]
    for p in parts:
        if p in SPLIT_MAP:
            return SPLIT_MAP[p]
    return "none"


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--near", type=int, default=5, help="유사 이미지 해밍거리 기준 (기본 5, 최대 5)")
    args = ap.parse_args()

    root = Path(args.data).resolve()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    if not root.exists():
        raise SystemExit(f"[오류] 폴더가 없습니다: {root}")

    rows, bad = [], []
    sizes, formats, modes = Counter(), Counter(), Counter()
    folder_counts = Counter()
    total_bytes = 0

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
    print(f"이미지 파일 {len(files)}장 발견. 검사 중...")

    for i, p in enumerate(files, 1):
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                sizes[im.size] += 1
                formats[(im.format or "?").upper()] += 1
                modes[im.mode] += 1
                h = dhash(im)
                width, height = im.size
            total_bytes += p.stat().st_size
            rel = p.relative_to(root)
            rows.append({
                "path": str(rel),
                "split": detect_split(p, root),
                "label": p.parent.name,
                "w": width,
                "h": height,
                "md5": md5_file(p),
                "dhash": h,
            })
            folder_counts[str(rel.parent)] += 1
        except (OSError, ValueError, UnidentifiedImageError) as e:
            bad.append(f"{p} ({e})")
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}")

    n = len(rows)
    print("\n========== 폴더별 개수 ==========")
    for k, v in sorted(folder_counts.items()):
        print(f"{k}: {v}장")

    print("\n========== 기본 정보 ==========")
    print(f"정상 {n}장 / 손상 {len(bad)}장 / 용량 {total_bytes / 1024**2:.1f} MB")
    print(f"형식: {dict(formats)}")
    print(f"색상 모드: {dict(modes)}")
    print("이미지 크기 상위 5개:", sizes.most_common(5), f"(총 {len(sizes)}종류)")
    for b in bad[:10]:
        print(" 손상:", b)

    # ---- 완전 중복
    by_md5 = defaultdict(list)
    for idx, r in enumerate(rows):
        by_md5[r["md5"]].append(idx)
    exact = [v for v in by_md5.values() if len(v) > 1]
    conflict = [v for v in exact if len({rows[i]["label"] for i in v}) > 1]
    print("\n========== 완전 중복 (MD5) ==========")
    print(f"중복 그룹 {len(exact)}개, 중복으로 낭비된 이미지 {sum(len(v) - 1 for v in exact)}장")
    print(f"라벨이 서로 다른 중복 그룹: {len(conflict)}개  <- 있으면 반드시 확인")
    for v in conflict[:5]:
        print("  ", [rows[i]["path"] for i in v])

    # ---- 유사 이미지
    thr = min(args.near, 5)
    dsu = DSU(n)
    buckets = defaultdict(list)
    for idx, r in enumerate(rows):
        h = r["dhash"]
        for c in range(len(CHUNK_BOUNDS) - 1):
            lo, hi = CHUNK_BOUNDS[c], CHUNK_BOUNDS[c + 1]
            buckets[(c, (h >> lo) & ((1 << (hi - lo)) - 1))].append(idx)
    for members in buckets.values():
        if len(members) < 2 or len(members) > 300:  # 너무 큰 버킷(단색 이미지 등)은 건너뜀
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if bin(rows[i]["dhash"] ^ rows[j]["dhash"]).count("1") <= thr:
                    dsu.union(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[dsu.find(idx)].append(idx)
    near = [v for v in groups.values() if len(v) > 1]

    print(f"\n========== 유사 이미지 (dHash, 해밍거리<={thr}) ==========")
    print(f"유사 그룹 {len(near)}개, 그룹에 속한 이미지 {sum(len(v) for v in near)}장 ({sum(len(v) for v in near) / max(n, 1) * 100:.1f}%)")

    # ---- 누수 추정
    split_counts = Counter(r["split"] for r in rows)
    print("\n========== 분할(split) 현황 ==========")
    print(dict(split_counts))
    leak_test = leak_val = 0
    cross = 0
    for v in near:
        sp = {rows[i]["split"] for i in v}
        if len(sp - {"none"}) > 1:
            cross += 1
        if "train" in sp:
            leak_test += sum(rows[i]["split"] == "test" for i in v)
            leak_val += sum(rows[i]["split"] == "val" for i in v)
    print(f"train과 test/val에 걸쳐 있는 유사 그룹: {cross}개")
    if split_counts.get("test"):
        print(f"test 이미지 중 train에 거의 같은 이미지가 있는 비율: "
              f"{leak_test}/{split_counts['test']} = {leak_test / split_counts['test'] * 100:.1f}%")
    if split_counts.get("val"):
        print(f"val 이미지 중 train에 거의 같은 이미지가 있는 비율: "
              f"{leak_val}/{split_counts['val']} = {leak_val / split_counts['val'] * 100:.1f}%")
    if not split_counts.get("test") and not split_counts.get("val"):
        print("(원본에 train/val/test 폴더가 없어 누수 비율은 계산하지 않음. 그룹 단위 분할이 필요함)")

    # ---- 저장
    with (out / "audit_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "label", "w", "h", "md5", "dhash_hex"])
        for r in rows:
            w.writerow([r["path"], r["split"], r["label"], r["w"], r["h"], r["md5"], f"{r['dhash']:016x}"])
    with (out / "near_dup_groups.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "size", "splits", "labels", "paths"])
        for gid, v in enumerate(sorted(near, key=len, reverse=True)):
            w.writerow([
                gid, len(v),
                "|".join(sorted({rows[i]["split"] for i in v})),
                "|".join(sorted({rows[i]["label"] for i in v})),
                " ; ".join(rows[i]["path"] for i in v[:10]),
            ])
    print(f"\n저장 완료: {out / 'audit_report.csv'}, {out / 'near_dup_groups.csv'}")


if __name__ == "__main__":
    main()
