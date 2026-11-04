"""공통 함수 모음 (02_prepare, 04_evaluate, 05_app에서 같이 사용)

핵심: 학습할 때와 예측할 때 전처리(to_square)를 똑같이 쓰기 위해 한 곳에 둔다.
"""
from pathlib import Path
from collections import defaultdict
import hashlib

from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMGSZ = 224

# 폴더 이름(소문자) -> (작업, 라벨)
CLASS_MAP = {
    "closed": ("eyes", "Closed"),
    "open": ("eyes", "Open"),
    "no_yawn": ("yawn", "no_yawn"),
    "yawn": ("yawn", "yawn"),
}

_BOUNDS = [0, 11, 22, 33, 44, 54, 64]  # 64비트 해시를 6조각으로 (해밍거리<=5면 한 조각은 반드시 같음)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_square(im: Image.Image, size: int = IMGSZ) -> Image.Image:
    """학습·추론 공통 전처리: EXIF 회전 -> RGB -> 비율 유지 + 패딩"""
    im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.pad(im, (size, size), method=Image.Resampling.LANCZOS, color=(24, 24, 24))


def dhash(im: Image.Image, size: int = 8) -> int:
    """64비트 차이 해시: 비슷한 이미지는 비슷한 값이 나온다."""
    g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            i = r * (size + 1) + c
            bits = (bits << 1) | int(px[i] > px[i + 1])
    return bits


def near_groups(hashes, thr: int = 5):
    """거의 같은 이미지끼리 같은 그룹 번호를 돌려준다. (분할 때 한 그룹은 한 곳에만 들어가야 누수가 없다)"""
    thr = min(thr, 5)
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets = defaultdict(list)
    for idx, h in enumerate(hashes):
        for c in range(len(_BOUNDS) - 1):
            lo, hi = _BOUNDS[c], _BOUNDS[c + 1]
            buckets[(c, (h >> lo) & ((1 << (hi - lo)) - 1))].append(idx)

    for members in buckets.values():
        if not 1 < len(members) <= 300:  # 단색 이미지처럼 너무 큰 버킷은 건너뜀
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if bin(hashes[i] ^ hashes[j]).count("1") <= thr:
                    parent[find(j)] = find(i)

    return [find(i) for i in range(n)]
