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
        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_detection.png"), gray_image)

    # 結果の表示
    # OpenCVでの色空間はBGRなので、RGBに変換して表示
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if isPlot:
        plt.imshow(gray_image, cmap="gray")
        plt.title("Face and Grayscale Detection")
        plt.axis(plt_axis)
        plt.show()
