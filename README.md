# 하품 분류: MediaPipe + EfficientNet-B0

하품 담당 부분만 구현한 프로젝트입니다. 기존 `yawn / no_yawn` 라벨로 학습하며, 눈 감김 모델과 별도로 실행합니다.

## 바뀐 부분과 실행 순서

| 파일 | 역할 |
|---|---|
| `01_audit_data.py` | 기존 파일 그대로: 이미지 검사 |
| `02_prepare.py` | 기존 파일 그대로: 중복 정리·그룹 분할·224×224 패딩 |
| `common.py` | 기존 전처리 공통 함수 유지 |
| `02b_extract_mouth.py` | **추가**: 기존 분할에서 입 주변 추출·PNG/CSV 저장 |
| `03_train.py` | ImageNet 사전학습 EfficientNet-B0를 하품 이진 분류로 전이학습 |
| `04_evaluate.py` | train/val/test 및 별도 원본 사진 평가, 실패율·오분류 저장 |
| `05_app.py` / `app.py` | Streamlit 없이 OpenCV 창에서 로컬 웹캠 실행 |
| `yawn_common.py` | 새 입 추출·모델 입력·체크포인트·시간 판정 공통 코드 |

`02_prepare.py`의 데이터 삭제·분할 동작도 원래 그대로입니다. 이미 `data/processed`가 있다면 **01·02는 다시 실행할 필요가 없습니다.** 새 단계는 기존 파일을 읽기만 합니다.

## 1. 설치

Python 3.11 또는 3.12의 새 가상환경에서 실행하는 것을 권장합니다. 아래 명령은 이 README가 있는 프로젝트 폴더에서 실행합니다. PyCharm에서도 해당 가상환경을 인터프리터로 선택하세요.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell에서 가상환경 활성화가 막히면 활성화 대신 `.venv\Scripts\python.exe`를 `python` 자리에 사용하거나 PyCharm 인터프리터 설정에서 선택하세요.

PyTorch 설치는 환경에 따라 CPU용일 수 있습니다. GPU로 학습하려면 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 Windows / Pip / Python / 사용 가능한 CUDA 빌드를 선택하여 해당 환경에 설치하세요. RTX 5060은 오래된 PyTorch/CUDA 빌드에서 지원되지 않을 수 있으므로 현재 공식 빌드를 사용하세요.

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`False`여도 CPU 실행은 가능합니다. GPU 지정은 `--device 0`, CPU 지정은 `--device cpu`입니다. `opencv-python`, `opencv-contrib-python`, `opencv-python-headless`를 함께 설치하면 충돌할 수 있으므로 새 환경에 위 requirements만 설치하는 편이 간단합니다. 데스크톱 창이 필요하므로 headless OpenCV는 사용하지 않습니다.

## 2. 기존 데이터 연결

이 폴더 안에 기존 `data/processed`를 복사하거나, `--input`에 기존 경로를 지정합니다.

```text
data/processed/
    manifest.csv
    yawn/
        train/no_yawn/...
        train/yawn/...
        val/no_yawn/...
        val/yawn/...
        test/no_yawn/...
        test/yawn/...
```

모든 상대 경로는 새 02b·03·04·05 파일이 있는 프로젝트 폴더 기준입니다. 기존 01·02는 원래처럼 터미널의 현재 폴더를 기준으로 합니다.

## 3. 입 주변 추출

```powershell
python 02b_extract_mouth.py
```

기존 데이터가 다른 폴더에 있다면:

```powershell
python 02b_extract_mouth.py --input "C:\project\vision\data\processed"
```

이 단계는 **기존 train/val/test와 라벨을 그대로 이어받습니다.** 다시 섞거나 분할하지 않습니다. MediaPipe는 사전학습된 Face Landmarker를 사용하며, 너희 하품 라벨로 재학습하지 않습니다. 최초 실행 때 공식 얼굴 모델을 `models/face_landmarker.task`로 다운로드합니다.

입술 특징점을 기준으로 입 너비의 1.8배 영역을 잡고, 입이 크게 열렸을 때에는 높이도 고려하여 사각형을 만듭니다. 턱 쪽 여유를 조금 포함하며, 입술만 꽉 차게 자르지 않습니다. ROI를 정사각형 224×224 PNG로 저장합니다. 입력 이미지 크기는 기존 전처리의 224×224를 그대로 사용합니다. 이전에 02의 `--imgsz`를 바꿨다면 02b의 `--face-size`도 같은 값으로 지정하세요.

| 생성물 | 확인할 내용 |
|---|---|
| `data/mouth/yawn/.../*.png` | 실제 학습용 입 이미지 |
| `mouth_manifest.csv` | 기존 group/출처, split, label, 입 좌표, 성공·실패 이유, 저장 경로 |
| `extraction_summary.csv` / `.json` | split·라벨별 입력 수, 성공 수, 실패 수, 성공률 |
| `mouth_review.jpg` | 원본에 표시한 추출 영역과 잘린 입의 비교. 실패 예시도 포함 |
| `crop_config.json` | 이후 학습·캠에서 재사용할 설정과 모델 해시 |

**학습 전에 mouth_review.jpg를 확인하세요.** 하품 이미지에서 실패가 유독 많은지도 확인합니다. 실패 사진은 `no_yawn`으로 바꾸지 않으며, 원본은 그대로 남습니다. 얼굴 없음, 여러 얼굴, 입이 너무 작음 등은 판단 불가입니다. 여러 얼굴 중 운전자를 고르는 기능은 넣지 않았으므로 한 명의 얼굴이 보이는 촬영 조건을 사용합니다.

이미 결과가 있는 폴더는 덮어쓰지 않습니다. 다른 설정은 새 출력 경로로 비교하세요.

```powershell
python 02b_extract_mouth.py --width-scale 2.0 --output data/mouth_wide
```

입 이미지가 이미 224×224로 줄어든 원본에서 생성되므로, 작게 나온 얼굴의 세부 정보가 복원되지는 않습니다. 이번 구현은 요청대로 기존 전처리를 유지합니다.

## 4. 학습

```powershell
python 03_train.py --task yawn
```

기본값: 25 epoch, batch 16, 첫 2 epoch는 분류부만 학습, 이후 전체 미세조정. 특징 추출부 학습률 0.0001, 분류부 0.001, 검증 loss가 7 epoch 동안 개선되지 않으면 조기 종료합니다. **가장 낮은 val loss의 가중치**를 `best.pt`로 저장합니다. test는 학습이나 모델 선택에 사용하지 않습니다.

첫 실행 때 ImageNet 사전학습 가중치를 다운로드합니다. ImageNet의 RGB 정규화 값을 사용하지만 입이 잘리지 않도록 기본 중앙 자르기 대신 전체 ROI를 유지합니다. 좌우 반전과 약한 색·밝기 증강은 train에만 적용합니다. 출력 순서는 `no_yawn=0`, `yawn=1`로 고정합니다.

```powershell
python 03_train.py --device 0 --epochs 30 --batch 16
# GPU 메모리가 부족하면 --batch 8 또는 --batch 4
```

기본 결과는 `runs/yawn_efficientnet_b0/`에 저장합니다.

- `weights/best.pt`: 평가·웹캠에 사용할 가중치, 클래스 순서, 입 추출 설정
- `weights/last.pt`: 마지막 epoch 가중치 (옵티마이저 재개용 파일은 아님)
- `results.csv`, `learning_curves.png`: loss·정확도·하품 F1
- `train_config.json`: 실행 설정

기존 결과를 덮어쓰지 않습니다. 재실험은 새 이름을 쓰세요.

```powershell
python 03_train.py --data data/mouth_wide --name yawn_b0_wide
python 04_evaluate.py --data data/mouth_wide --weights runs/yawn_b0_wide/weights/best.pt
python 05_app.py --weights runs/yawn_b0_wide/weights/best.pt
```

**예전 YOLO best.pt는 호환되지 않습니다.** 새 데이터로 EfficientNet-B0를 학습해야 합니다. 학습 후 다른 컴퓨터에서 사용할 때에는 다시 학습하지 않고 새 `best.pt`, 동일한 `face_landmarker.task`, 실행 코드를 옮기면 됩니다.

## 5. 평가

```powershell
python 04_evaluate.py
```

`runs/yawn_efficientnet_b0/evaluation/`에 다음을 저장합니다.

- split별 `*_metrics.json`: 입 검출 성공률, 분류 정확도, 클래스별 precision/recall/F1, macro F1
- `*_confusion.csv`: 행=정답, 열=예측, 순서 no_yawn → yawn
- `*_predictions.csv`: 이미지별 예측과 yawn 점수
- `*_errors.csv`: 오분류 및 판단 불가 이미지
- `summary.json`: 전체 결과

분류 정확도는 **입 추출에 성공한 사진 기준**입니다. 별도로 전체 입력 대비 올바른 판정 비율(`correct_over_all_inputs`)과 라벨별 검출 성공률을 기록합니다. 판단 불가는 혼동행렬의 no_yawn에 섞지 않습니다. 성공 이미지가 0개면 정확도를 계산하지 않습니다.

새로 찍은 원본 사진을 테스트하려면 `data/webcam_test/yawn/no_yawn/`, `data/webcam_test/yawn/yawn/`에 넣습니다. `--extra`에서는 웹캠과 동일하게 패딩→입 추출→분류를 적용합니다.

```powershell
python 04_evaluate.py --extra data/webcam_test/yawn
```

임계값은 기본 0.5이며 `--threshold`로 바꿀 수 있습니다. 결정에는 val만 쓰고 최종 test를 보고 반복 조정하지 마세요. 기존 유사 이미지 group 분할은 **사람별 분할을 보장하지 않습니다.** 새로운 사람·다른 각도·조명·말하기·웃기 영상에서 추가 확인해야 합니다.

## 6. 내 컴퓨터의 캠 켜기

```powershell
python 05_app.py
# 같은 실행
python app.py
```

카메라 영상, 입 영역, 모델 입력 입 이미지, 하품 점수, 지속 시간을 표시합니다. 화면 글자는 OpenCV 기본 글꼴 호환 때문에 영어입니다. Q / ESC 또는 창의 닫기 버튼으로 종료합니다. 영상은 저장하지 않습니다.

| 표시 | 의미 |
|---|---|
| `NO YAWN` | 해당 프레임의 하품 점수가 기준 미만 |
| `CHECKING` | 하품 점수가 기준 이상이지만 지속 시간 미달 |
| `YAWN CANDIDATE` | 기본 1.5초 이상 연속으로 하품 예측이 유지됨 |
| `UNKNOWN` | 얼굴·입 검출 실패. 이전 하품 지속 시간 초기화 |
| `Candidate events` | 위 조건을 새로 충족한 횟수. 정확한 실제 하품 횟수로 검증된 값은 아님 |

1.5초는 실험용 설정값이며, 하품의 보편적 기준이 아닙니다. 확률이 한 프레임이라도 기준 미만이면 지속 시간을 초기화하는 단순 규칙입니다. 말하기·웃기와 하품의 시간 패턴을 학습하는 영상 모델은 아닙니다. 눈 감김이나 최종 졸음 점수는 계산하지 않습니다.

```powershell
python 05_app.py --camera 1
python 05_app.py --threshold 0.6 --hold-seconds 2
```

캠이 안 켜지면 카메라 번호 0/1, Windows 카메라 권한, Zoom 등 다른 앱의 카메라 사용 여부를 확인합니다. 추론이 느려 프레임 간격이 1초를 넘으면 기본적으로 연속성을 초기화합니다. 필요한 경우 `--max-gap 2`로 조절하되 긴 간격에서는 지속 시간의 정확도가 낮아집니다.

캠 없이 원본 사진 한 장을 검사할 수도 있습니다.

```powershell
python 05_app.py --image "C:\photos\example.jpg"
```

`reports/image_check/`에 JSON·입 이미지·미리보기 그림이 저장됩니다. 사진 한 장은 시간 조건을 적용하지 않습니다. 앱은 `best.pt`에 저장된 입 추출 규칙을 읽고, 학습과 동일한 MediaPipe 모델인지 확인합니다.

## 검증과 자료

`VALIDATION.md`에 이 패키지에서 실제로 실행한 검사와 미검증 범위를 기록합니다. 선택적으로 `python -m pip install pytest` 후 `python -m pytest tests -q`로 핵심 회귀 검사를 실행할 수 있습니다.

- [MediaPipe Face Landmarker Python 공식 문서](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [TorchVision EfficientNet-B0 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
#   y a w m 1  
 # yawm1
