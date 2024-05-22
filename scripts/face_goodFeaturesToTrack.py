import numpy as np
import cv2
from matplotlib import pyplot as plt
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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100000000, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# 画像を保存
if isOutputImage:
    today_date = datetime.now().strftime("%Y-%m-%d")
    output_folder = os.path.join("output", today_date)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(os.path.join(output_folder, "face_goodFeaturesToTrack.png"), image)

# 結果の表示
if isPlot:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Face and GoodFeaturesToTrack Detection")
    plt.axis(plt_axis)
    plt.show()