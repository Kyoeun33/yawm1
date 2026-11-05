"""이미지를 업로드해 하품 여부를 확인하는 간단한 Streamlit 웹 앱.

실행: python -m streamlit run upload_app.py
"""
import streamlit as st
import torch
from PIL import Image, ImageOps

from yawn_common import (
    DEFAULT_FACE_MODEL,
    DEFAULT_WEIGHTS,
    CropConfig,
    MouthExtractor,
    choose_device,
    load_checkpoint,
    make_transform,
    project_path,
    verify_face_model,
)


st.set_page_config(page_title="하품 이미지 판별", page_icon="😮", layout="centered")
st.title("하품 이미지 판별")
st.write("이미지를 올리면 얼굴의 입 주변을 분석해 하품 여부를 판정합니다.")

uploaded = st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png", "bmp", "webp"])


@st.cache_resource
def load_predictor():
    device = choose_device("cpu")
    model, checkpoint = load_checkpoint(project_path(DEFAULT_WEIGHTS), device)
    face_model = verify_face_model(project_path(DEFAULT_FACE_MODEL), checkpoint)
    extractor = MouthExtractor(face_model, CropConfig(**checkpoint["crop_config"]))
    return model, checkpoint, extractor, make_transform(), device


if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        with st.spinner("이미지를 분석하고 있습니다..."):
            model, checkpoint, extractor, transform, device = load_predictor()
            result = extractor.extract(image)

            if result["status"] != "ok":
                st.warning("얼굴 하나와 입 영역을 찾지 못했습니다. 정면 얼굴이 선명하게 나온 이미지를 사용해 주세요.")
                st.caption(f"분석 상태: {result['status']}")
            else:
                x = transform(result["crop"]).unsqueeze(0).to(device)
                probability = float(model(x).softmax(1)[0, 1])
                threshold = float(checkpoint.get("threshold", 0.5))
                if probability >= threshold:
                    st.success("하품으로 판단했습니다.")
                else:
                    st.info("하품이 아닌 것으로 판단했습니다.")
                st.metric("하품 확률", f"{probability:.1%}")
                st.caption(f"판정 기준: {threshold:.1%}")
                st.image(result["crop"], caption="모델이 분석한 입 주변", width=224)
    except Exception as exc:
        st.error(f"분석 중 문제가 발생했습니다: {exc}")
