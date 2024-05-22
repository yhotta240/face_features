import cv2
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

# 顔検出器の読み込み
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 顔の検出
faces = face_cascade.detectMultiScale(
    image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

if len(faces) == 0:
    print("顔が検出できませんでした")
else:
    for x, y, w, h in faces:
        # 検出された顔領域に四角形を描画
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 顔の領域を切り取り
        roi_color = image[y : y + h, x : x + w]

        # 顔の特徴点を検出
        landmark_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        landmarks = landmark_cascade.detectMultiScale(roi_color)

        for lx, ly, lw, lh in landmarks:
            # 特徴点を描画（目は緑色）
            cv2.rectangle(roi_color, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)

        # 鼻の特徴点を検出
        nose_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
        )
        noses = nose_cascade.detectMultiScale(roi_color)

        for nx, ny, nw, nh in noses:
            # 特徴点を描画（鼻は赤色）
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)

        # 画像を保存
        if isOutputImage:
            today_date = datetime.now().strftime("%Y-%m-%d")
            output_folder = os.path.join("output", today_date)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, "face_landmark_detection.png"), image)

        # 結果の表示
        # 画像をBGRからRGBに変換して表示
        if isPlot:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Face Landmark Detection")
            plt.axis(plt_axis)
            plt.show()
