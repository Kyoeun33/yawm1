"""OpenCV 웹캠 데모. 실행하면 기본 카메라가 켜진다. Q/ESC 또는 창 닫기로 종료.

python 01-05_app.py
python 01-05_app.py --camera 1
python 01-05_app.py --image example.jpg  # 사진 1장 검사(캠 없이 결과 저장)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from yawn_common import (DEFAULT_WEIGHTS, DEFAULT_FACE_MODEL, project_path, choose_device,
    load_checkpoint, verify_face_model, CropConfig, MouthExtractor, make_transform, YawnTimer, save_json)


@torch.inference_mode()
def predict(image, extractor, model, transform, device):
    result = extractor.extract(image)
    probability = None
    if result['status']=='ok':
        x = transform(result['crop']).unsqueeze(0).to(device)
        probability = float(model(x).softmax(1)[0,1])
    return result, probability


def render(result, probability, state, duration, events, fps):
    face = cv2.cvtColor(np.asarray(result['face']), cv2.COLOR_RGB2BGR).copy()
    color = (0,180,255) if state=='YAWN CANDIDATE' else (80,220,80)
    if result['box']:
        x1,y1,x2,y2 = result['box']
        cv2.rectangle(face,(x1,y1),(x2,y2),color,1)
    canvas = np.full((520,880,3),24,dtype=np.uint8)
    canvas[20:500,0:480] = cv2.resize(face,(480,480))
    if result['crop'] is not None:
        canvas[15:239,545:769] = cv2.cvtColor(np.asarray(result['crop']),cv2.COLOR_RGB2BGR)
    text = [state, f'p(yawn): {probability:.3f}' if probability is not None else 'p(yawn): unavailable',
            f'Positive duration: {duration:.1f}s', f'Candidate events: {events}',
            f'Status: {result["status"]}', f'FPS: {fps:.1f}', 'Q / ESC: quit']
    for i,line in enumerate(text):
        cv2.putText(canvas,line,(495,270+32*i),cv2.FONT_HERSHEY_SIMPLEX,.57,color if i==0 else (230,230,230),1,cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--face-model', default=DEFAULT_FACE_MODEL)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--hold-seconds', type=float, default=1.5, help='하품 예측이 연속 유지되어야 하는 시간(실험용)')
    ap.add_argument('--max-gap', type=float, default=1.0, help='이보다 긴 프레임 간격이면 지속 시간 초기화')
    ap.add_argument('--image', help='웹캠 대신 원본 사진 한 장 평가')
    ap.add_argument('--output', default='reports/image_check')
    a = ap.parse_args()
    device = choose_device(a.device)
    model, ck = load_checkpoint(project_path(a.weights),device)
    threshold = ck['threshold'] if a.threshold is None else a.threshold
    if not 0 < threshold < 1 or a.hold_seconds < 0 or a.max_gap <= 0:
        raise ValueError('threshold/hold-seconds/max-gap 값을 확인하세요.')
    face_model = verify_face_model(project_path(a.face_model),ck)
    transform = make_transform()
    with MouthExtractor(face_model,CropConfig(**ck['crop_config'])) as extractor:
        if a.image:
            with Image.open(project_path(a.image)) as im:
                result,p = predict(im,extractor,model,transform,device)
            out = project_path(a.output)
            out.mkdir(parents=True,exist_ok=True)
            state = 'UNKNOWN' if p is None else ('YAWN FRAME' if p>=threshold else 'NO YAWN')
            data = dict(status=result['status'], p_yawn=p, prediction=state, threshold=threshold, box=result['box'])
            save_json(out/'prediction.json',data)
            # PIL 저장으로 Windows 한글 경로도 지원
            Image.fromarray(cv2.cvtColor(render(result,p,state,0,0,0),cv2.COLOR_BGR2RGB)).save(out/'preview.jpg')
            if result['crop'] is not None:
                result['crop'].save(out/'mouth.png')
            else:
                (out/'mouth.png').unlink(missing_ok=True)
            print(data, '\n저장:',out)
            return
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError('카메라를 열지 못했습니다. --camera 1, 카메라 권한, 다른 앱의 사용 여부를 확인하세요.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        timer = YawnTimer(threshold,a.hold_seconds,a.max_gap)
        window = 'Yawn - MediaPipe + EfficientNet-B0'
        print('웹캠 실행 중. Q/ESC 또는 창 닫기. 영상은 저장하지 않습니다.')
        try:
            cv2.namedWindow(window,cv2.WINDOW_NORMAL)
            while True:
                start = time.monotonic()
                ok,frame = cap.read()
                if not ok:
                    print('카메라 프레임을 읽지 못해 종료합니다.')
                    break
                rgb = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                result,p = predict(rgb,extractor,model,transform,device)
                now = time.monotonic()
                state,duration = timer.update(p,now)
                fps = 1/max(now-start,1e-6)
                cv2.imshow(window,render(result,p,state,duration,timer.events,fps))
                if cv2.waitKey(1)&0xFF in (ord('q'),ord('Q'),27):
                    break
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
