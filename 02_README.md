# 두 번째 실험: MobileNetV3-Small 하품 분류

기존 프로젝트의 `yawn_project` 폴더에 아래 **4개 파일**을 그대로 넣어 실행하세요. 기존 MediaPipe 입 추출 결과(`data/mouth_wide_fixed`)를 재사용하며, `01`, `02`, `02b`를 다시 실행하지 않습니다.

- `02-03_train.py`: ImageNet 사전학습 MobileNetV3-Small로 새 하품 분류기 학습
- `02-04_evaluate.py`: 새 체크포인트 평가
- `02-05_app.py`: 기존과 동일한 OpenCV 웹캠 확인. Q 또는 ESC로 종료
- `02_yawn_common.py`: 위 3개 파일이 함께 쓰는 전처리·모델 로딩 함수

아래 기존 파일도 같은 폴더에 있어야 합니다: `common.py`, `data/mouth_wide_fixed/crop_config.json`, `data/mouth_wide_fixed/mouth_manifest.csv`, `models/face_landmarker.task` 및 기존 입 이미지 폴더. Python 패키지는 원래 프로젝트의 `requirements.txt`를 이용하세요. `02_yawn_common.py`는 숫자로 시작하는 파일 이름이므로 일반 `from ... import` 대신 스크립트 내부에서 `import_module`로 로딩합니다.

## 실행 순서

```powershell
python 02-03_train.py
python 02-04_evaluate.py
python 02-05_app.py
```

처음 학습할 때 MobileNetV3-Small의 ImageNet 사전학습 가중치를 자동 다운로드합니다. **기존 EfficientNet-B0 best.pt는 다른 구조라 그대로 사용할 수 없으며**, 새 모델을 위 순서대로 학습해야 합니다. 학습이 완료되면 `runs/02_yawn_mobilenet_v3_small/weights/02_best.pt`가 만들어집니다. 평가와 앱은 이 파일을 기본으로 읽습니다.

기본 데이터가 다른 위치라면 학습과 평가에 같은 경로를 지정하세요. 데이터 내부의 추출 설정·manifest가 학습 때와 일치해야 합니다.

```powershell
python 02-03_train.py --data data/mouth_wide_fixed
python 02-04_evaluate.py --data data/mouth_wide_fixed
```

모델을 다시 학습하려면 기존 결과가 덮어써지지 않도록 새 이름을 지정하고, 평가·웹캠에서 해당 가중치를 함께 지정합니다.

```powershell
python 02-03_train.py --name 02_yawn_mobile_v2
python 02-04_evaluate.py --weights runs/02_yawn_mobile_v2/weights/02_best.pt
python 02-05_app.py --weights runs/02_yawn_mobile_v2/weights/02_best.pt
```

캠 장치가 0이 아니면 `python 02-05_app.py --camera 1`로 실행하세요. `NO YAWN`, `CHECKING`, `YAWN CANDIDATE`, `UNKNOWN` 상태와 실시간 하품 점수·입 이미지를 기존 앱처럼 표시합니다. 기본적으로 하품 확률이 0.5 이상으로 1.5초 연속 유지될 때 후보를 표시합니다. 본인의 카메라에서 실제로 판정되는지는 직접 확인해야 합니다.

## 결과 위치

- 학습: `runs/02_yawn_mobilenet_v3_small/weights/02_best.pt`, `02_last.pt`, `02_results.csv`, `02_learning_curves.png`, `02_train_config.json`
- 평가: 같은 run 안 `02_evaluation/02_train_metrics.json`, `02_val_metrics.json`, `02_test_metrics.json` 및 예측 기록
- 사진 한 장 시험: `python 02-05_app.py --image 사진.jpg` → `reports/02_image_check/02_prediction.json`, `02_preview.jpg`, `02_mouth.png`

`02-04_evaluate.py`는 원래 평가처럼 분할별 지표를 저장하지만, 실시간 카메라 확인을 위해 추가 지표 수집 절차를 요구하지 않습니다. 기존 `01-03_train.py`, `01-04_evaluate.py`, `01-05_app.py`, `yawn_common.py`와 기존 학습 결과는 수정하지 않습니다. 다른 모델끼리는 가중치 파일을 교차 사용하지 않습니다.
