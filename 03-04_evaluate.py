"""입 이미지 분류 성능과 입 검출 성공률을 함께 평가한다.

python 03-04_evaluate.py
python 03-04_evaluate.py --extra data/webcam_test/yawn
--extra는 잘라내기 전 원본 이미지의 no_yawn/, yawn/ 폴더이다.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from importlib import import_module
_model_tools = import_module("03_yawn_common")
LABELS, SPLITS, DEFAULT_WEIGHTS, DEFAULT_FACE_MODEL, project_path, read_json, save_json, save_csv, sha256, image_samples, MouthDataset, make_transform, load_checkpoint, choose_device, binary_metrics, MouthExtractor, CropConfig, verify_face_model = (_model_tools.LABELS, _model_tools.SPLITS, _model_tools.DEFAULT_WEIGHTS, _model_tools.DEFAULT_FACE_MODEL, _model_tools.project_path, _model_tools.read_json, _model_tools.save_json, _model_tools.save_csv, _model_tools.sha256, _model_tools.image_samples, _model_tools.MouthDataset, _model_tools.make_transform, _model_tools.load_checkpoint, _model_tools.choose_device, _model_tools.binary_metrics, _model_tools.MouthExtractor, _model_tools.CropConfig, _model_tools.verify_face_model)


@torch.inference_mode()
def classify_crops(model, samples, device, batch, threshold):
    rows = []
    loader = DataLoader(MouthDataset(samples), batch_size=batch, shuffle=False, num_workers=0)
    for x, y, paths in loader:
        probabilities = model(x.to(device)).softmax(1)[:,1].cpu().tolist()
        for path, actual, p in zip(paths, y.tolist(), probabilities):
            rows.append(dict(input=path, actual=LABELS[actual], predicted=LABELS[int(p>=threshold)],
                             p_yawn=p, status='ok'))
    return rows


@torch.inference_mode()
def classify_raw(model, folder, device, extractor, threshold):
    transform = make_transform()
    rows = []
    for path, actual in image_samples(folder):
        row = dict(input=str(path), actual=LABELS[actual], predicted='unknown', p_yawn='', status='')
        try:
            with Image.open(path) as im:
                result = extractor.extract(im)
            row['status'] = result['status']
            if result['status']=='ok':
                x = transform(result['crop']).unsqueeze(0).to(device)
                p = float(model(x).softmax(1)[0,1])
                row.update(predicted=LABELS[int(p>=threshold)], p_yawn=p)
        except (OSError, ValueError) as exc:
            row.update(status=f'read_or_crop_error: {exc}')
        rows.append(row)
    return rows


def report(title, rows, out):
    ok = [r for r in rows if r['status']=='ok']
    metrics = binary_metrics([LABELS.index(r['actual']) for r in ok],
                             [LABELS.index(r['predicted']) for r in ok])
    total, n = len(rows), len(ok)
    metrics.update(total_input=total, detected=n, unknown=total-n,
                   detection_coverage=n/total if total else 0,
                   correct_over_all_inputs=sum(r['actual']==r['predicted'] for r in ok)/total if total else None,
                   failure_counts=dict(Counter(r['status'] for r in rows if r['status']!='ok')))
    metrics['coverage_by_class'] = {}
    for label in LABELS:
        sub = [r for r in rows if r['actual']==label]
        classified = sum(r['status']=='ok' for r in sub)
        metrics['coverage_by_class'][label] = dict(total=len(sub), detected=classified,
            coverage=classified/len(sub) if sub else None,
            correctly_classified_over_all=sum(r['predicted']==label for r in sub)/len(sub) if sub else None)
    save_json(out/f'03_{title}_metrics.json', metrics)
    save_csv(out/f'03_{title}_predictions.csv', rows, ['input','actual','predicted','p_yawn','status'])
    errors = [r for r in rows if r['predicted']!=r['actual']]
    save_csv(out/f'03_{title}_errors.csv', errors, ['input','actual','predicted','p_yawn','status'])
    cm = metrics['confusion_matrix']
    save_csv(out/f'03_{title}_confusion.csv', [dict(actual=label, **dict(zip(LABELS,cm[i])))
             for i,label in enumerate(LABELS)], ['actual',*LABELS])
    accuracy = metrics['accuracy']
    acc_text = f'{accuracy:.2%}' if accuracy is not None else '평가 불가'
    print(f'\n[{title}] 전체 {total} / 판단 가능 {n} / 판단 불가 {total-n}')
    print(f'입 검출 성공률 {metrics["detection_coverage"]:.2%} / 성공 이미지 내 정확도 {acc_text}')
    print('yawn 지표:', metrics['per_class']['yawn'])
    print('혼동행렬 (행=정답, 열=예측; no_yawn, yawn):', cm)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=['yawn'], default='yawn')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--data', default='data/mouth_wide_fixed')
    ap.add_argument('--extra', help='원본 사진 테스트: <folder>/no_yawn, <folder>/yawn')
    ap.add_argument('--face-model', default=DEFAULT_FACE_MODEL)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--threshold', type=float, default=None, help='기본: 학습 설정. 조정은 val로만 수행')
    a = ap.parse_args()
    device = choose_device(a.device)
    weights, root = project_path(a.weights), project_path(a.data)
    model, ck = load_checkpoint(weights, device)
    threshold = ck['threshold'] if a.threshold is None else a.threshold
    if not 0 < threshold < 1 or a.batch < 1:
        raise ValueError('threshold는 0~1 사이, batch는 1 이상이어야 합니다.')
    meta = read_json(root/'crop_config.json')
    if meta['crop'] != ck['crop_config'] or meta['face_model_sha256'] != ck['face_model_sha256']:
        raise ValueError('학습 때와 다른 입 추출 설정입니다.')
    if sha256(root/'mouth_manifest.csv') != ck['dataset_manifest_sha256']:
        raise ValueError('학습 때 사용한 데이터 분할과 다릅니다. 새 데이터는 --extra로 평가하세요.')
    with (root/'mouth_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest = list(csv.DictReader(f))
    out = weights.parent.parent/'03_evaluation'
    out.mkdir(parents=True,exist_ok=True)
    results = dict(threshold=threshold, weights=str(weights), class_order=list(LABELS))
    for split in SPLITS:
        records = [r for r in manifest if r['split']==split]
        successful = [r for r in records if r['status']=='ok']
        samples = [(root/r['output'], LABELS.index(r['label'])) for r in successful]
        for p,_ in samples:
            if not p.is_file():
                raise FileNotFoundError(f'입 추출 이미지가 없습니다: {p}')
        rows = classify_crops(model,samples,device,a.batch,threshold) if samples else []
        rows += [dict(input=r['input'], actual=r['label'], predicted='unknown', p_yawn='', status=r['status'])
                 for r in records if r['status']!='ok']
        results[split] = report(split,rows,out)
    if a.extra:
        face_model = verify_face_model(project_path(a.face_model),ck)
        with MouthExtractor(face_model,CropConfig(**ck['crop_config'])) as extractor:
            rows = classify_raw(model,project_path(a.extra),device,extractor,threshold)
        results['extra'] = report('extra',rows,out)
    save_json(out/'03_summary.json',results)
    print(f'\n저장: {out}\n검출 실패는 no_yawn으로 바꾸지 않았습니다. 정확도와 검출 성공률을 함께 보세요.')
    print('기존 그룹 분할은 운전자별 분할을 보장하지 않습니다. 새로운 사람의 영상도 확인하세요.')


if __name__ == '__main__':
    main()
