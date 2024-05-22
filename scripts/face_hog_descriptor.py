import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
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
image = cv2.imread(image_path)

# グレースケールに変換
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
    # HOG特徴量の計算と保存
    for x, y, w, h in faces:
        roi = gray_image[y : y + h, x : x + w]  # 顔の部分を切り出す
        if isResize:
            roi_resized = cv2.resize(roi, (width_size, hight_size))  # リサイズ
        else:
            roi_resized = roi
        
        # HOG特徴量を計算
        hog_features, hog_image = hog(
            roi_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )

        # HOG画像の正規化
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 画像を保存
    if isOutputImage:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join("output", today_date)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, "face_hog_image.png"), hog_image_rescaled * 255)

    # 結果の表示
    if isPlot:
        plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        plt.title("Face and HOG Detection")
        plt.axis(plt_axis)
        plt.show()


