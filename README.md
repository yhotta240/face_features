# 顔検出＆特徴量抽出 Python プログラム

このリポジトリは、顔検出および特徴量抽出のための様々な Python スクリプトを含んでいます。各スクリプトは異なるアルゴリズムや手法を使用しており、簡単に実行できるようになっています。

## リポジトリのクローン

まず、リポジトリをクローンします：

```bash
git clone https://github.com/yhotta240/face_features.git
cd face_features
```

## セットアップ

### 依存関係のインストール

必要なパッケージをインストールするために、以下のコマンドを実行してください：

```bash
pip install -r requirements.txt

```

## 画像のアップロード
imagesフォルダに画像をアップロードします。

## 設定の変更

`initialize.py` には、ユーザーが変更できる設定が含まれています。以下の設定を必要に応じて変更してください：

```python
# initialize.py

# 入力画像のパス
image_path = "images/sample.png"

# 画像のリサイズ設定
isResize = True
width_size = 256
hight_size = 256

# プロット設定
isPlot = True
plt_axis = "off"

# 出力画像の保存設定
isOutputImage = True

```
## 実行方法

以下のコマンドを実行して run.py を開始します：

```python
python run.py

```

リポジトリには以下のスクリプトが含まれています。run.py を実行して、実行するスクリプトを選択できます。
1. `face_canny_edge.py` - Canny エッジ検出 <br>
2. `face_corner.py` - Harris コーナー検出 <br>
3. `face_detection.py` - 顔検出 <br>
4. `face_dlib.py` - dlib を使用した顔検出 <br>
5. `face_dlib_features_csv.py` - dlib を使用した顔特徴点抽出と CSV 出力 <br>
6. `face_goodFeaturesToTrack.py` - Shi-Tomasi 角点検出 <br>
7. `face_haar-like.py` - Haar-like 特徴量を使用した顔検出 <br>
8. `face_hog_descriptor.py` - HOG 特徴量抽出 <br>
9. `face_landmark_detection.py` - 顔ランドマーク検出 <br>
10. `grayscale.py` - グレースケール変換 <br>
11. `hog_descriptor.py` - HOG 特徴量抽出 <br>

実行後、番号を入力して対応するスクリプトを選択します。

例：

```markdown
コードをコピーする
1. face_canny_edge.py
2. face_corner.py
3. face_detection.py
...
選択してください: 1
```
選択したスクリプトが実行されます。

## プロジェクト構成
```
face_features/
│
├── images/
│   └── sample.png         # サンプル画像（変更可）
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat # 顔ランドマーク検出用モデル
│
├── scripts/
│   ├── face_canny_edge.py     # Canny エッジ検出スクリプト
│   ├── face_corner.py         # Harris コーナー検出スクリプト
│   ├── face_detection.py      # 顔検出スクリプト
│   ├── face_dlib.py           # dlib を使用した顔検出スクリプト
│   ├── face_dlib_features_csv.py # dlib を使用した顔特徴点抽出と CSV 出力スクリプト
│   ├── face_goodFeaturesToTrack.py # Shi-Tomasi 角点検出スクリプト
│   ├── face_haar-like.py      # Haar-like 特徴量を使用した顔検出スクリプト
│   ├── face_hog_descriptor.py # HOG 特徴量抽出スクリプト
│   ├── face_landmark_detection.py # 顔ランドマーク検出スクリプト
│   ├── grayscale.py           # グレースケール変換スクリプト
│   └── hog_descriptor.py      # HOG 特徴量抽出スクリプト
│
├── face_features.ipynb    # Colab を使用して作成された Jupyter ノートブック
│
├── initialize.py          # 設定ファイル　
│
├── requirements.txt       # 依存関係ファイル
│
└── run.py                 # スクリプト選択および実行ファイル

```
この README は、リポジトリの概要、セットアップ方法、設定の変更方法、スクリプトの実行方法、およびプロジェクト構成について説明しています。必要に応じて内容を追加や修正してください。
