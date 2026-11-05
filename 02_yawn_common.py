"""하품 전용 공통 처리: 입 추출 규칙과 모델 입력을 학습·평가·캠에서 공유한다."""
from pathlib import Path
from dataclasses import dataclass, asdict
import csv
import hashlib
import json
import math
import os
import urllib.request

import numpy as np
from PIL import Image
from common import EXT, to_square

ROOT = Path(__file__).resolve().parent
LABELS = ('no_yawn', 'yawn')  # 순서 고정: yawn 확률은 항상 index 1
SPLITS = ('train', 'val', 'test')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
DEFAULT_FACE_MODEL = 'models/face_landmarker.task'
DEFAULT_WEIGHTS = 'runs/02_yawn_mobilenet_v3_small/weights/02_best.pt'
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
        191, 80, 81, 82, 13, 312, 311, 310, 415,
        185, 40, 39, 37, 0, 267, 269, 270, 409)


def project_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def ensure_face_model(path):
    """최초 실행 때 공식 모델 다운로드. 중간 실패 파일은 모델로 사용하지 않는다."""
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.download')
    print('MediaPipe 얼굴 모델 다운로드 중...')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=90) as r, tmp.open('wb') as f:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError('모델 다운로드 크기가 비정상입니다.')
        os.replace(tmp, path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'모델 다운로드 실패. {MODEL_URL} 파일을 {path}에 직접 저장하세요.') from exc
    return path


@dataclass(frozen=True)
class CropConfig:
    # 기존 02_prepare의 정사각 패딩 크기. 새 전처리도 이 이미지에서 시작한다.
    face_size: int = 224
    output_size: int = 224
    width_scale: float = 1.8
    height_scale: float = 1.35
    y_offset: float = 0.08
    min_mouth_pixels: float = 6.0
    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    algorithm: str = 'mouth_square_v1'

    def __post_init__(self):
        if self.algorithm != 'mouth_square_v1':
            raise ValueError('지원하지 않는 입 추출 규칙입니다.')
        if self.face_size < 32 or self.output_size != 224:
            raise ValueError('face_size >= 32, output_size = 224 이어야 합니다.')
        if self.width_scale < 1 or self.height_scale < 1:
            raise ValueError('입 영역 배율은 1 이상이어야 합니다.')


def mouth_box(landmarks, size, config):
    """입 벌림 높이만으로 확대하지 않도록 입 너비를 기준으로 여유 영역을 잡는다."""
    w, h = size
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LIPS])
    if not np.isfinite(pts).all():
        raise ValueError('invalid_landmarks')
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mw, mh = hi - lo
    if mw < config.min_mouth_pixels or mh < 1:
        raise ValueError('mouth_too_small')
    if lo[0] < 0 or lo[1] < 0 or hi[0] >= w or hi[1] >= h:
        raise ValueError('mouth_outside_image')
    side = int(math.ceil(max(mw * config.width_scale, mh * config.height_scale)))
    cx, cy = (lo + hi) / 2
    cy += mw * config.y_offset
    x, y = int(round(cx - side / 2)), int(round(cy - side / 2))
    return x, y, x + side, y + side


def crop_with_padding(im, box, output_size):
    x1, y1, x2, y2 = box
    # 화면 밖 여유 영역에는 기존 패딩과 같은 회색을 채운다.
    out = Image.new('RGB', (x2 - x1, y2 - y1), (24, 24, 24))
    clipped = (max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2))
    out.paste(im.crop(clipped), (clipped[0] - x1, clipped[1] - y1))
    return out.resize((output_size, output_size), Image.Resampling.LANCZOS)


class MouthExtractor:
    def __init__(self, model_path, config):
        import mediapipe as mp
        self.mp, self.config = mp, config
        model_path = ensure_face_model(model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            # Windows 네이티브 MediaPipe는 한글이 포함된 경로를 열지 못할 수 있다.
            # 파일은 Python에서 읽고 모델 바이트를 전달해 경로 인코딩을 피한다.
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,  # 여러 얼굴이면 임의로 승객을 고르지 않고 판단 불가 처리
            min_face_detection_confidence=config.detection_confidence,
            min_face_presence_confidence=config.presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(self, image, prepared=False):
        if prepared:
            if image.size != (self.config.face_size, self.config.face_size):
                raise ValueError('기존 전처리 이미지 크기와 --face-size가 다릅니다.')
            face = image.convert('RGB')
        else:
            face = to_square(image, self.config.face_size)
        arr = np.ascontiguousarray(np.asarray(face))
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=arr))
        count = len(result.face_landmarks)
        if count != 1:
            return {'status': 'no_face' if count == 0 else 'multiple_faces', 'face': face,
                    'crop': None, 'box': None, 'face_count': count}
        try:
            box = mouth_box(result.face_landmarks[0], face.size, self.config)
        except ValueError as exc:
            return {'status': str(exc), 'face': face, 'crop': None, 'box': None, 'face_count': count}
        return {'status': 'ok', 'face': face, 'crop': crop_with_padding(face, box, self.config.output_size),
                'box': box, 'face_count': count}

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def image_samples(folder, require_both=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'이미지 폴더가 없습니다: {folder}')
    unknown = [p.name for p in folder.iterdir() if p.is_dir() and p.name not in LABELS]
    if unknown:
        raise ValueError(f'라벨 폴더는 {LABELS}만 허용합니다: {unknown}')
    samples = []
    for i, label in enumerate(LABELS):
        paths = sorted(p for p in (folder / label).rglob('*') if p.is_file() and p.suffix.lower() in EXT)
        if require_both and not paths:
            raise ValueError(f'{folder / label}에 이미지가 없습니다.')
        samples += [(p, i) for p in paths]
    if not samples:
        raise ValueError(f'이미지가 없습니다: {folder}')
    return samples


def make_transform(training=False):
    from torchvision import transforms as T
    # 입이 잘리지 않도록 ImageNet의 중앙 자르기는 사용하지 않는다.
    steps = [T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)]
    if training:
        steps += [T.RandomHorizontalFlip(), T.ColorJitter(brightness=.15, contrast=.15, saturation=.1)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class MouthDataset:
    def __init__(self, samples, training=False):
        self.samples = samples
        self.transform = make_transform(training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        p, y = self.samples[index]
        with Image.open(p) as im:
            x = self.transform(im.convert('RGB'))
        return x, y, str(p)


def make_model(pretrained=False):
    from torch import nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


def choose_device(value=None):
    import torch
    value = value or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if value.isdigit():
        value = f'cuda:{value}'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다. --device cpu 또는 CUDA 지원 PyTorch 설치를 확인하세요.')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


def load_checkpoint(path, device):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'학습 가중치가 없습니다: {path}. 03_train.py를 먼저 실행하세요.')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('architecture') != 'mobilenet_v3_small' or ck.get('class_names') != list(LABELS):
        raise ValueError('이 프로젝트의 MobileNetV3-Small best.pt가 필요합니다. 기존 YOLO 파일은 호환되지 않습니다.')
    model = make_model(False)
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval(), ck


def verify_face_model(path, checkpoint):
    path = ensure_face_model(path)
    if sha256(path) != checkpoint['face_model_sha256']:
        raise ValueError('입 추출에 사용한 MediaPipe 모델과 파일이 다릅니다.')
    return path


def binary_metrics(actual, predicted):
    cm = [[0, 0], [0, 0]]
    for a, p in zip(actual, predicted):
        cm[int(a)][int(p)] += 1
    n = sum(map(sum, cm))
    per = {}
    for i, label in enumerate(LABELS):
        tp, fp, fn = cm[i][i], cm[1-i][i], cm[i][1-i]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = dict(precision=precision, recall=recall, f1=f1, support=sum(cm[i]))
    return dict(n=n, accuracy=(cm[0][0]+cm[1][1])/n if n else None,
                macro_f1=sum(p['f1'] for p in per.values())/2 if n else None,
                per_class=per, confusion_matrix=cm)


class YawnTimer:
    """확률 초과가 hold초 연속 지속되면 후보 1회. 음성 프레임 또는 검출 실패시 초기화."""
    def __init__(self, threshold=.5, hold=1.5, max_gap=1.0):
        self.threshold, self.hold, self.max_gap = threshold, hold, max_gap
        self.started = self.last = None
        self.active = False
        self.events = 0

    def update(self, probability, now):
        gap = self.last is not None and now - self.last > self.max_gap
        self.last = now
        if gap or probability is None or probability < self.threshold:
            self.started, self.active = None, False
        if probability is None:
            return 'UNKNOWN', 0.0
        if probability < self.threshold:
            return 'NO YAWN', 0.0
        if self.started is None:
            self.started = now
        duration = now - self.started
        if duration >= self.hold:
            if not self.active:
                self.events += 1
            self.active = True
            return 'YAWN CANDIDATE', duration
        return 'CHECKING', duration
