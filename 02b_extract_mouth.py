"""기존 02_prepare 결과에서 입만 추가 추출한다. 원본/분할/기존 전처리는 수정하지 않는다.

python 02b_extract_mouth.py
결과: data/mouth/yawn/<split>/<label>/*.png, mouth_manifest.csv, crop_config.json
"""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import csv
import importlib.metadata

from PIL import Image, ImageDraw
from yawn_common import (LABELS, SPLITS, CropConfig, MouthExtractor, project_path,
                         image_samples, save_json, save_csv, sha256, DEFAULT_FACE_MODEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='data/processed', help='02_prepare 결과의 루트')
    ap.add_argument('--output', default='data/mouth', help='새 출력 폴더; 기존 분할은 유지')
    ap.add_argument('--face-model', default=DEFAULT_FACE_MODEL)
    ap.add_argument('--face-size', type=int, default=224, help='02_prepare --imgsz와 일치')
    ap.add_argument('--width-scale', type=float, default=1.8)
    a = ap.parse_args()
    src, out = project_path(a.input), project_path(a.output)
    if src == out or src in out.parents or out in src.parents:
        raise ValueError('입력과 출력은 서로 분리된 폴더를 사용하세요.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out}은 비어 있지 않습니다. --output으로 새 폴더를 지정하세요.')

    # 먼저 기존 폴더 목록을 읽는다. 무작위 분할이나 데이터 이동은 하지 않는다.
    samples = [(split, p, y) for split in SPLITS for p, y in image_samples(src/'yawn'/split)]
    config = CropConfig(face_size=a.face_size, width_scale=a.width_scale)
    for _, p, _ in samples:
        with Image.open(p) as im:
            if im.size != (config.face_size, config.face_size):
                raise ValueError(f'{p.name}: {im.size}. --face-size를 기존 전처리 크기에 맞추세요.')

    # 기존 manifest의 사람 식별 대신 유사 이미지 group 정보를 그대로 연결한다.
    lineage = {}
    original_manifest = src/'manifest.csv'
    if original_manifest.exists():
        with original_manifest.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['task'] == 'yawn':
                    name = row['output'].replace('\\', '/').split('/')[-1]
                    lineage[(row['split'], row['label'], name)] = row

    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for label in LABELS:
            (out/'yawn'/split/label).mkdir(parents=True, exist_ok=True)
    fields = ['split', 'label', 'group', 'original_source', 'input', 'input_sha256', 'output',
              'status', 'face_count', 'x1', 'y1', 'x2', 'y2', 'error']
    rows, preview = [], []
    face_model = project_path(a.face_model)
    # 중간에 중단돼도 기록한 행은 남긴다. complete 플래그는 끝에만 저장한다.
    with MouthExtractor(face_model, config) as extractor, (out/'mouth_manifest.csv').open(
            'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (split, p, y) in enumerate(samples, 1):
            label = LABELS[y]
            old = lineage.get((split, label, p.name), {})
            row = dict.fromkeys(fields, '')
            row.update(split=split, label=label, group=old.get('group', ''),
                       original_source=old.get('source', ''), input=str(p), input_sha256=sha256(p))
            try:
                with Image.open(p) as im:
                    result = extractor.extract(im, prepared=True)
                row.update(status=result['status'], face_count=result['face_count'])
                if result['status'] == 'ok':
                    # 확장자·하위 폴더가 다른 동명 파일의 덮어쓰기를 방지한다.
                    import hashlib
                    token = hashlib.sha256(str(p.relative_to(src)).encode()).hexdigest()[:12]
                    dest = out/'yawn'/split/label/f'{p.stem}_{token}.png'
                    result['crop'].save(dest)
                    row['output'] = dest.relative_to(out).as_posix()
                    row.update(dict(zip(('x1','y1','x2','y2'), result['box'])))
                key = (split, label, result['status'])
                if sum(x[0] == key for x in preview) < 3:
                    preview.append((key, result['face'].copy(), result['crop'], result['box']))
            except (OSError, ValueError) as exc:
                row.update(status='read_or_crop_error', error=str(exc))
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if i % 100 == 0 or i == len(samples):
                print(f'입 추출 {i}/{len(samples)}')

    summary = []
    for split in SPLITS:
        for label in LABELS:
            sub = [r for r in rows if r['split'] == split and r['label'] == label]
            counts = Counter(r['status'] for r in sub)
            total, ok = len(sub), counts['ok']
            summary.append(dict(split=split, label=label, total=total, ok=ok, failed=total-ok,
                                coverage=ok/total if total else 0, status_counts=dict(counts)))
            print(f'{split}/{label}: {ok}/{total} 성공, 실패 {total-ok}')
    save_json(out/'extraction_summary.json', summary)
    save_csv(out/'extraction_summary.csv', [{k:v for k,v in r.items() if k!='status_counts'} for r in summary],
             ['split','label','total','ok','failed','coverage'])
    save_json(out/'crop_config.json', dict(crop=asdict(config), face_model_sha256=sha256(face_model),
              mediapipe_version=importlib.metadata.version('mediapipe'),
              source_manifest_sha256=sha256(original_manifest) if original_manifest.exists() else None,
              manifest_sha256=sha256(out/'mouth_manifest.csv'), complete=True))
    # 원본+입 쌍을 눈으로 확인: 실패 이미지도 포함한다.
    if preview:
        sheet = Image.new('RGB', (640, 185*len(preview)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (key, face, crop, box) in enumerate(preview):
            annotated = face.copy()
            if box:
                ImageDraw.Draw(annotated).rectangle(box, outline='lime', width=2)
            sheet.paste(annotated.resize((160,160)), (0,185*i))
            if crop is not None:
                sheet.paste(crop.resize((160,160)), (165,185*i))
            draw.text((332,185*i+25), '/'.join(key), fill='black')
        sheet.save(out/'mouth_review.jpg')
    print(f'\n저장 완료: {out}\n먼저 mouth_review.jpg와 extraction_summary.csv를 확인하세요.')
    if any(r['ok']==0 for r in summary):
        print('[주의] 일부 split/label의 성공 이미지가 0장입니다. 학습 전에 확인하세요.')


if __name__ == '__main__':
    main()
