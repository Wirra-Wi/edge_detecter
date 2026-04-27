import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("邊緣偵測器")

# 側邊參數設定欄
st.sidebar.header("參數設定")

# 選擇模式
mode = st.sidebar.selectbox("選擇模式", ["自然影像", "醫學影像", "自訂參數"])

# 定義預設參數
presets = {
    "自然影像": {
        "Canny": {"blur_kernel": 3, "minVal": 70, "maxVal": 200},
        "LoG": {"blur_kernel": 3, "thresh": 10},
        "Sobel": {"blur_kernel": 3, "kernel_size": 3, "thresh": 100}
    },
    "醫學影像": {
        "Canny": {"blur_kernel": 5, "minVal": 50, "maxVal": 150},
        "LoG": {"blur_kernel": 5, "thresh": 5},
        "Sobel": {"blur_kernel": 3, "kernel_size": 3, "thresh": 70}
    }
}

# 上傳圖片
uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # 轉換為灰階
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # 影像去噪
    st.sidebar.subheader("影像去噪")
    denoise = st.sidebar.checkbox("啟用去噪", value=True)
    if denoise:
        denoise_h = st.sidebar.slider("去噪強度 h", 1, 20, 15)
        gray = cv2.fastNlMeansDenoising(gray, None, denoise_h, 7, 21)
        st.sidebar.caption("已對影像進行非局部均值去噪")

    st.image(image, caption="原始圖片", use_container_width=True)

    # 選擇邊緣偵測方法
    method = st.selectbox("選擇邊緣偵測演算法", ["Canny", "LoG", "Sobel"])

    if mode == "自訂參數":
        if method == "Canny":
            blur_kernel = st.sidebar.slider("高斯模糊 kernel 大小", 1, 15, 3, step=2)
            minVal = st.sidebar.slider("Canny minVal", 0, 255, 70)
            maxVal = st.sidebar.slider("Canny maxVal", 0, 255, 200)
        elif method == "LoG":
            blur_kernel = st.sidebar.slider("高斯模糊 kernel 大小", 1, 15, 5, step=2)
            thresh = st.sidebar.slider("閾值", 0, 100, 5)
        elif method == "Sobel":
            blur_kernel = st.sidebar.slider("Sobel Gaussian blur kernel", 1, 15, 3, step=2)
            kernel_size = st.sidebar.selectbox("Sobel kernel size", [1, 3, 5, 7], index=1)
            thresh = st.sidebar.slider("閾值", 0, 255, 70)
    else:
        # 使用預設參數
        params = presets[mode][method]
        if method == "Canny":
            blur_kernel = params["blur_kernel"]
            minVal = params["minVal"]
            maxVal = params["maxVal"]
        elif method == "LoG":
            blur_kernel = params["blur_kernel"]
            thresh = params["thresh"]
        elif method == "Sobel":
            blur_kernel = params["blur_kernel"]
            kernel_size = params["kernel_size"]
            thresh = params["thresh"]

    if st.button("執行"):
        if method == "Canny":
            # Canny 邊緣偵測
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            edges = cv2.Canny(blurred, minVal, maxVal)
            result = edges

        elif method == "LoG":
            # 高斯模糊 + 拉普拉斯
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            log = cv2.Laplacian(blurred, cv2.CV_64F)
            log_abs = np.absolute(log)
            _, result = cv2.threshold(log_abs, thresh, 255, cv2.THRESH_BINARY) 
            result = np.uint8(result)

        elif method == "Sobel":
            # Sobel 邊緣偵測，先進行高斯模糊
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)
            dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)
            mag = np.sqrt(dx**2 + dy**2)
            mag = np.uint8(np.clip(mag, 0, 255))
            _, result = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
        else:
            st.error("請選擇一個有效的邊緣偵測方法。")
            result = None

        # 顯示結果
        st.image(result, caption=f"{method} 結果", use_container_width=True, clamp=True)

        # 下載按鈕
        result_pil = Image.fromarray(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="下載處理後圖片",
            data=byte_im,
            file_name=f"edge_detected_{method}.png",
            mime="image/png"
        )