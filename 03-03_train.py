"""MediaPipe로 추출한 입 이미지로 ResNet-18 하품 이진 분류를 전이학습한다.

python 03-03_train.py --task yawn
결과: runs/03_yawn_resnet18/weights/03_best.pt, 03_results.csv, 03_learning_curves.png
"""
import argparse
import csv
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from importlib import import_module
_model_tools = import_module("03_yawn_common")
LABELS, project_path, read_json, save_json, sha256, image_samples, MouthDataset, make_model, choose_device, binary_metrics = (_model_tools.LABELS, _model_tools.project_path, _model_tools.read_json, _model_tools.save_json, _model_tools.sha256, _model_tools.image_samples, _model_tools.MouthDataset, _model_tools.make_model, _model_tools.choose_device, _model_tools.binary_metrics)


def epoch_pass(model, loader, device, criterion, optimizer=None, freeze_features=False, threshold=.5):
    training = optimizer is not None
    model.train(training)
    if training and freeze_features:
        model.eval()  # warmup 중 ResNet의 BatchNorm 통계 보존
        model.fc.train()
    losses, actual, predicted = 0., [], []
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            losses += loss.item()*len(y)
            actual.extend(y.detach().cpu().tolist())
            predicted.extend((logits.detach().softmax(1)[:,1] >= threshold).cpu().int().tolist())
    return losses/len(actual), binary_metrics(actual, predicted)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=['yawn'], default='yawn')
    ap.add_argument('--data', default='data/mouth_wide_fixed')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4, help='특징 추출부 학습률; 분류부는 10배')
    ap.add_argument('--warmup', type=int, default=2, help='분류부만 먼저 학습할 epoch 수')
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--device', default=None)
    ap.add_argument('--workers', type=int, default=0, help='Windows에서는 0 권장')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', default='03_yawn_resnet18')
    ap.add_argument('--threshold', type=float, default=.5)
    a = ap.parse_args()
    if a.epochs < 1 or a.batch < 1 or a.patience < 1 or a.warmup < 0 or a.lr <= 0 or not 0 < a.threshold < 1:
        raise ValueError('epochs/batch/patience/lr/threshold 값을 확인하세요.')
    if not a.name or any(c in a.name for c in '/\\') or a.name in ('.','..'):
        raise ValueError('--name에는 폴더 이름 하나만 지정하세요.')
    root, run = project_path(a.data), project_path('runs')/a.name
    metadata = read_json(root/'crop_config.json')
    if not metadata.get('complete') or sha256(root/'mouth_manifest.csv') != metadata['manifest_sha256']:
        raise ValueError('입 추출이 완료되지 않았거나 처리 내역이 변경됐습니다. 02b를 다시 실행하세요.')
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f'{run}에 결과가 있습니다. --name 03_yawn_resnet18_v2처럼 새 이름을 지정하세요.')
    # test는 학습·조기 종료·모델 선택에 사용하지 않는다.
    tr = image_samples(root/'yawn/train', require_both=True)
    va = image_samples(root/'yawn/val', require_both=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(a.device)
    generator = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(MouthDataset(tr, True), batch_size=a.batch, shuffle=True,
                              num_workers=a.workers, generator=generator)
    val_loader = DataLoader(MouthDataset(va), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    print(f'장치: {device} / train {len(tr)}장 / val {len(va)}장 / 클래스: {LABELS}')
    print('ImageNet 사전학습 ResNet-18 로딩 (최초 실행 시 다운로드)')
    model = make_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': a.lr},
        {'params': model.fc.parameters(), 'lr': a.lr*10}], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    (run/'weights').mkdir(parents=True, exist_ok=True)
    save_json(run/'03_train_config.json', dict(**vars(a), device_used=str(device), data_metadata=metadata))
    fields = ['epoch','train_loss','val_loss','train_accuracy','val_accuracy','val_macro_f1',
              'val_yawn_precision','val_yawn_recall','val_yawn_f1','seconds']
    best, bad, history = float('inf'), 0, []
    with (run/'03_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1,a.epochs+1):
            start = time.perf_counter()
            freeze = epoch <= a.warmup
            for name, p in model.named_parameters():
                if not name.startswith('fc.'):
                    p.requires_grad_(not freeze)
            tl, tm = epoch_pass(model, train_loader, device, criterion, optimizer, freeze, a.threshold)
            vl, vm = epoch_pass(model, val_loader, device, criterion, threshold=a.threshold)
            yc = vm['per_class']['yawn']
            row = dict(epoch=epoch, train_loss=tl, val_loss=vl, train_accuracy=tm['accuracy'],
                       val_accuracy=vm['accuracy'], val_macro_f1=vm['macro_f1'], val_yawn_precision=yc['precision'],
                       val_yawn_recall=yc['recall'], val_yawn_f1=yc['f1'], seconds=time.perf_counter()-start)
            history.append(row)
            writer.writerow(row)
            f.flush()
            checkpoint = dict(format_version=1, architecture='resnet18',
                model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                class_names=list(LABELS), positive_index=1, epoch=epoch, threshold=a.threshold,
                crop_config=metadata['crop'], face_model_sha256=metadata['face_model_sha256'],
                dataset_manifest_sha256=metadata['manifest_sha256'], val_loss=vl, val_metrics=vm,
                torch_version=str(torch.__version__))
            torch.save(checkpoint, run/'weights/03_last.pt')
            if vl < best:
                best, bad = vl, 0
                torch.save(checkpoint, run/'weights/03_best.pt')
            else:
                bad += 1
            print(f'{epoch:02d}/{a.epochs} train loss={tl:.4f} val loss={vl:.4f} '
                  f'val acc={vm["accuracy"]:.3f} yawn F1={yc["f1"]:.3f}', flush=True)
            if bad >= a.patience:
                print('검증 loss 개선이 없어 조기 종료합니다.')
                break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for key in ('train_loss','val_loss'):
        axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    for key in ('train_accuracy','val_accuracy','val_yawn_f1'):
        axes[1].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    for ax in axes:
        ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'03_learning_curves.png',dpi=160); plt.close(fig)
    print(f'\n완료: {run / "weights/03_best.pt"}\n다음: python 03-04_evaluate.py --weights runs/{a.name}/weights/03_best.pt')


if __name__ == '__main__':
    main()
