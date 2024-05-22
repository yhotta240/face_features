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
    # 顔領域ごとに処理
    for (x, y, w, h) in faces:
        # 顔を四角で囲む
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 顔領域を切り出し
        face_region = gray_image[y:y+h, x:x+w]
        
        if isResize:
            # リサイズする場合
            face_region = cv2.resize(face_region, (width_size, hight_size))

        # Harrisコーナー検出
        blockSize = 2
        ksize = 3
        k = 0.06
        dst = cv2.cornerHarris(face_region, blockSize, ksize, k)
        
        # 結果を膨張させることでコーナーを強調
        dst = cv2.dilate(dst, None)
        
        # コーナーが検出されたピクセルをマーク
        face_region[dst > 0.01 * dst.max()] = 255
        
        # # 元の画像にコーナーを描画
        # image[y:y+h, x:x+w] = cv2.cvtColor(face_region, cv2.COLOR_GRAY2BGR)

    # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_corner_sample.png"), face_region)

    # 結果の表示
    if isPlot:
        plt.imshow(face_region, cmap='gray')
        plt.title('Face and Corner Detection')
        plt.axis(plt_axis)
        plt.show()



