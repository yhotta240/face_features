import cv2
import dlib
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize import (
    image_path,
    isPlot,
    plt_axis,
    isOutputImage,
)

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出器とランドマーク予測器の読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)  # ダウンロードが必要

# 顔を検出
faces = detector(gray_image)

if len(faces) == 0:
    print("顔が検出できませんでした")
else:
    for face in faces:
        landmarks = predictor(gray_image, face)

        # ランドマークを色分けして描画
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            if 0 <= n <= 16:  # 輪郭
                color = (255, 0, 0)
            elif 17 <= n <= 26:  # 眉
                color = (0, 255, 0)
            elif 27 <= n <= 35:  # 鼻
                color = (0, 0, 255)
            elif 36 <= n <= 47:  # 目
                color = (255, 255, 0)
            elif 48 <= n <= 67:  # 口
                color = (255, 0, 255)

            cv2.circle(image, (x, y), 2, color, -1)
            cv2.putText(
                image, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
            )  # 番号を描画

    # 左目の幅を測定
    left_eye_width = landmarks.part(39).x - landmarks.part(36).x
    # 右目の幅を測定
    right_eye_width = landmarks.part(45).x - landmarks.part(42).x
    print(f"左目の幅: {left_eye_width} ピクセル")
    print(f"右目の幅: {right_eye_width} ピクセル")

    # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_dlib_sample.png"), image)

    # 結果の表示
    if isPlot:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Face and Dlib Detection")
        plt.axis(plt_axis)
        plt.show()
