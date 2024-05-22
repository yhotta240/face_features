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
    for x, y, w, h in faces:
        face = gray_image[y : y + h, x : x + w]  # 顔の部分を切り出す
        if isResize:
            resized_face = cv2.resize(face, (hight_size, width_size))  # 256pxにリサイズする
        else:
            resized_face = face
            
        # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_haar-like_sample.png"), resized_face)

    # 結果の表示
    if isPlot:
        plt.imshow(resized_face, cmap="gray")
        plt.title("Face and Haar-like Detection")
        plt.axis(plt_axis)
        plt.show()

