# SAM3 Damage Detector - 既存Python環境でのセットアップ

Windows 11 + 既存Python環境での簡単セットアップガイド

## 📋 前提条件

- ✅ Python 3.10以上（確認済み: Python 3.12.10）
- ✅ NVIDIA GPU（推奨、CPUでも動作可能）
- ✅ 16GB以上のRAM

---

## 🚀 セットアップ手順（5分）

### Step 1: 必要なパッケージのインストール

```powershell
# PyTorch（CUDA 11.8対応）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 画像処理・可視化ライブラリ
pip install opencv-python matplotlib numpy Pillow tqdm

# Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**または、一括インストール:**

```powershell
pip install -r requirements.txt
```

### Step 2: SAMモデルのダウンロード

```powershell
# modelsディレクトリに移動
cd models

# SAM ViT-Hモデルをダウンロード（約2.4GB）
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "sam_vit_h_4b8939.pth"

# プロジェクトルートに戻る
cd ..
```

### Step 3: 動作確認

```powershell
# インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 単一画像でテスト
python main.py --mode single --image "data/1_Test_images-kensg/kensg-rebarexposureRb_001.png"
```

---

## ✅ セットアップ完了！

以下のコマンドですぐに使えます：

```powershell
# 単一画像の損傷検出
python main.py --mode single --image "data/1_Test_images-kensg/kensg-rebarexposureRb_001.png"

# バッチ処理（最初の10枚）
python main.py --mode batch --max_images 10

# 全画像を処理
python main.py --mode batch
```

---

## 🔧 トラブルシューティング

### CUDAが利用できない場合

CPUモードで実行可能です（処理は遅くなります）：

```powershell
python main.py --mode single --image "data/1_Test_images-kensg/kensg-rebarexposureRb_001.png" --no_fp16 --no_quantization
```

### パッケージのインストールエラー

個別にインストールしてください：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install matplotlib
pip install numpy
pip install Pillow
pip install tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### gitがインストールされていない場合

SAMを手動でインストール：

```powershell
# 代替方法（SAMのwhlファイルを使用）
pip install segment-anything
```

---

## 📝 Conda不要の理由

このプロジェクトは標準的なPythonパッケージのみを使用しているため、**既存のPython環境で問題なく動作**します。

- ✅ 仮想環境不要
- ✅ 既存のPythonバージョンで動作（3.10+）
- ✅ pipでの簡単インストール

---

## 🎯 次のアクション

1. **Step 1**: パッケージをインストール
   ```powershell
   pip install -r requirements.txt
   ```

2. **Step 2**: SAMモデルをダウンロード
   ```powershell
   cd models
   Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "sam_vit_h_4b8939.pth"
   cd ..
   ```

3. **Step 3**: 実行！
   ```powershell
   python main.py --mode batch --max_images 5
   ```

---

**簡単！Conda不要で今すぐ使えます！** 🚀
