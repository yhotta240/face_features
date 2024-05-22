import cv2
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize import (
    image_path,
    isResize,
    width_size,
    hight_size,
    isPlot,
    plt_axis,
    isOutputImage,
)

# 画像の読み込みとグレースケール化
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出器の読み込み
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 顔の検出
faces = face_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

if len(faces) == 0:
    print("顔が検出できませんでした")
else:
    # 顔が検出された場合の処理
    for x, y, w, h in faces:
        face_roi = gray_image[y : y + h, x : x + w]  # 顔の部分を切り出す
        # Cannyエッジ検出
        if isResize:
            face_resized = cv2.resize(face_roi, (width_size, hight_size))  # リサイズ
            edges = cv2.Canny(
                face_resized, 100, 200
            )  # 100から200の間の勾配をエッジとして検出
        else:
            edges = cv2.Canny(
                face_roi, 100, 200
            )  # 100から200の間の勾配をエッジとして検出

    # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_canny_edge_sample.png"), edges)

    # 結果の表示
    if isPlot:
        plt.imshow(edges, cmap="gray")
        plt.title("Face and Canny Edge Detection")
        plt.axis(plt_axis)
        plt.show()
