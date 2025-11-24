# SAM Models Directory

このディレクトリにSAMモデルファイルを配置してください。

## ダウンロード方法

### ViT-H (推奨・最高精度)
```bash
# PowerShell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "sam_vit_h_4b8939.pth"
```
- ファイル名: `sam_vit_h_4b8939.pth`
- サイズ: 約2.4GB

### ViT-L (中精度・高速)
```bash
# PowerShell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" -OutFile "sam_vit_l_0b3195.pth"
```
- ファイル名: `sam_vit_l_0b3195.pth`
- サイズ: 約1.2GB

### ViT-B (高速・低メモリ)
```bash
# PowerShell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b_01ec64.pth"
```
- ファイル名: `sam_vit_b_01ec64.pth`
- サイズ: 約375MB

## 使用モデルの変更

`src/config.py`で設定を変更してください：

```python
MODEL_CONFIG = {
    "model_type": "vit_h",  # "vit_h", "vit_l", "vit_b"
    "checkpoint_path": MODELS_DIR / "sam_vit_h_4b8939.pth",
    ...
}
```
