# 세 번째 실험: MediaPipe + ResNet-18

기존 `yawn_project` 폴더에 압축 안의 파일 5개를 복사하세요. 원래 EfficientNet-B0와 두 번째 MobileNetV3-Small 파일은 그대로 둡니다. 이 실험은 기존 MediaPipe 입 추출 결과 `data/mouth_wide_fixed`를 재사용합니다. `01`, `02`, `02b`를 다시 실행할 필요는 없습니다.

## 필요한 파일

- `03-03_train.py`: ResNet-18 전이학습
- `03-04_evaluate.py`: 새 가중치 평가
- `03-05_app.py`: OpenCV 로컬 웹캠 실행, Q 또는 ESC로 종료
- `03_yawn_common.py`: 입 추출·ResNet 모델 생성·체크포인트 로딩. Windows 한글 경로용 MediaPipe 모델 바이트 전달 포함
- `03_README.md`: 이 안내

기존 프로젝트의 `common.py`, `models/face_landmarker.task`, `data/mouth_wide_fixed/`도 필요합니다. 설치된 Python 환경에서 원래 `requirements.txt`를 사용하세요. 처음 학습할 때 ResNet-18 ImageNet 가중치를 자동으로 다운로드합니다.

## 실행

```powershell
python 03-03_train.py
python 03-04_evaluate.py
python 03-05_app.py
```

기본 데이터 경로는 `data/mouth_wide_fixed`입니다. 다르면 학습과 평가에 같은 `--data`를 지정합니다. 이 실험의 학습 결과는 `runs/03_yawn_resnet18/weights/03_best.pt`에 저장됩니다. 새 평가와 앱은 이 경로를 기본으로 사용하므로 수동 지정이 필요 없습니다.

사진으로 캠 없이 먼저 확인하려면 다음을 실행합니다.

```powershell
python 03-05_app.py --image "C:\photos\example.jpg"
```

사진 결과는 `reports/03_image_check/`에 저장됩니다. 캠 장치 번호가 다르면 `python 03-05_app.py --camera 1`을 사용합니다. 화면에는 학습 때와 같은 입 추출 영역·모델의 하품 점수와 `NO YAWN`, `CHECKING`, `YAWN CANDIDATE`, `UNKNOWN` 상태가 표시됩니다. 기본 기준은 하품 확률 0.5 이상이 1.5초 지속되는 것입니다.

기존 파일과 겹치지 않도록 새 파일과 결과 이름에는 03을 붙였습니다. 가중치도 모델 구조가 다르므로 이전 `best.pt`나 `02_best.pt`를 ResNet-18 앱에 넣을 수 없습니다. 데이터·분할이 같아도 ResNet-18 학습은 새로 해야 합니다.

기본 실행의 결과가 이미 있는 경우 기존 결과를 덮어쓰지 않습니다. 새 실험 이름을 사용하고 평가·앱에도 가중치를 명시하세요.

```powershell
python 03-03_train.py --name 03_yawn_resnet18_v2
python 03-04_evaluate.py --weights runs/03_yawn_resnet18_v2/weights/03_best.pt
python 03-05_app.py --weights runs/03_yawn_resnet18_v2/weights/03_best.pt
```

평가는 기존과 동일한 분할별 결과를 `runs/03_yawn_resnet18/03_evaluation/`에 저장합니다. `03_results.csv`, `03_learning_curves.png`, `03_train_config.json`도 이 run 폴더에 생성됩니다. 실제 데이터 정확도와 PC의 웹캠 속도는 사용자가 해당 환경에서 확인해야 합니다.
