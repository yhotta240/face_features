import cv2
import dlib
import csv  # import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize import (
    image_path,
)

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出器とランドマーク予測器の読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# 顔を検出
faces = detector(gray_image)

# CSVファイルに書き込むデータのリストを作成
data = []

# 顔領域ごとに処理
for face in faces:
    landmarks = predictor(gray_image, face)

    # 特徴の座標を取得
    right_eye_width = landmarks.part(45).x - landmarks.part(42).x
    right_eye_height = landmarks.part(47).y - landmarks.part(43).y

    left_eye_width = landmarks.part(39).x - landmarks.part(36).x
    left_eye_height = landmarks.part(41).y - landmarks.part(37).y

    mouth_width = landmarks.part(54).x - landmarks.part(48).x
    mouth_height = landmarks.part(66).y - landmarks.part(62).y

    nose_width = landmarks.part(35).x - landmarks.part(31).x
    nose_height = landmarks.part(33).y - landmarks.part(27).y

    right_eyebrow_width = landmarks.part(21).x - landmarks.part(17).x
    right_eyebrow_height = landmarks.part(21).y - min(
        landmarks.part(18).y,
        landmarks.part(19).y,
        landmarks.part(20).y,
        landmarks.part(21).y,
    )

    left_eyebrow_width = landmarks.part(26).x - landmarks.part(22).x
    left_eyebrow_height = landmarks.part(26).y - min(
        landmarks.part(23).y,
        landmarks.part(24).y,
        landmarks.part(25).y,
        landmarks.part(26).y,
    )

    # データをリストに追加
    data.append(["右目", right_eye_width, right_eye_height])
    data.append(["左目", left_eye_width, left_eye_height])
    data.append(["口", mouth_width, mouth_height])
    data.append(["鼻", nose_width, nose_height])
    data.append(["右眉", right_eyebrow_width, right_eyebrow_height])
    data.append(["左眉", left_eyebrow_width, left_eyebrow_height])


# 出力フォルダーを作成
today_date = datetime.now().strftime("%Y-%m-%d")
output_folder = os.path.join("output", today_date)
os.makedirs(output_folder, exist_ok=True)

# CSVファイルに書き込み
output_csv_path = os.path.join(output_folder, "face_features.csv")
with open(output_csv_path, "w", newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["特徴", "幅", "高さ"])
    writer.writerows(data)

print(f"顔の特徴の幅と高さを {output_csv_path} に保存しました。")
