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

# 画像を読み込む
image = cv2.imread(image_path)

# グレースケールに変換する
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 画像を保存
if isOutputImage:
    today_date = datetime.now().strftime("%Y-%m-%d")
    output_folder = os.path.join("output", today_date)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(os.path.join(output_folder, "grayscale.png"), gray_image)

# 結果の表示
if isPlot:
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale Detection")
    plt.axis(plt_axis)
    plt.show()