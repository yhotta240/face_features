import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize import image_path

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

# HOG特徴量の計算と表示
for x, y, w, h in faces:
    roi = gray_image[y : y + h, x : x + w]  # 顔の部分を切り出す
    roi_resized = cv2.resize(roi, (128, 128))  # 128x128にリサイズ

    # HOG特徴量を計算
    hog_features, hog_image = hog(
        roi_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )

    # HOG特徴量を棒グラフで表示
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(hog_features)), hog_features, color="skyblue")
    plt.xlabel("Cell")
    plt.ylabel("HOG Value")
    plt.title("Histogram of Oriented Gradients")
    plt.show()


# # CSVファイルに保存
# with open('hog_features.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for (x, y, w, h) in faces:
#         roi = gray_image[y:y+h, x:x+w]  # 顔の部分を切り出す
#         roi = cv2.resize(roi, (128, 128))  # 128x128にリサイズ
#         hog_features = hog.compute(roi)  # HOG特徴量を計算
#         writer.writerow(hog_features.flatten().tolist())
