# SAM3 Damage Detector - GPU対応セットアップ完了

## ✅ セットアップ完了

GPU対応のPyTorch環境が正常に構築されました！

### 環境情報
- **Python**: 3.11.9
- **PyTorch**: 2.7.1+cu118
- **CUDA**: 11.8
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB)

---

## 🚀 使い方

### 単一画像の損傷検出

```powershell
python main.py --mode single --image "data/1_Test_images-kensg/kensg-rebarexposureRb_001.png"
```

### バッチ処理（複数画像）

```powershell
# 最初の10枚を処理
python main.py --mode batch --max_images 10

# 全254枚を処理
python main.py --mode batch
```

---

## ⚙️ 設定について

### GPU + FP32（デフォルト・推奨）

- **メモリ使用量**: 約2.4GB
- **処理速度**: 高速
- **精度**: 最高
- **推奨環境**: GPU 8GB以上

```powershell
python main.py --mode single --image "path/to/image.png"
```

### CPU + FP32（GPUなし環境）

- **メモリ使用量**: 約2.4GB (RAM)
- **処理速度**: 遅い
- **精度**: GPU版と同じ

```powershell
python main.py --mode single --image "path/to/image.png" --no_fp16 --no_quantization
```

---

## 📝 技術的な注意点

### INT8量子化について

- ❌ **CUDA非対応**: PyTorchのINT8動的量子化はCPUのみサポート
- ✅ **代替案**: FP32でGPU実行（RTX 4060 Ti 16GBなら十分対応可能）

### FP16推論について

- ⚠️ **型の不一致**: SAMモデルの一部でFP32とFP16の混在により動作不安定
- ✅ **推奨**: FP32で安定動作

### メモリ最適化

RTX 4060 Ti 16GBの場合：
- FP32: 約2.4GB使用 → **余裕で対応可能**
- バッチ処理も問題なし

---

## 🎯 パフォーマンス

| 設定 | GPU メモリ | 処理速度 | 備考 |
|------|-----------|---------|------|
| GPU + FP32 | ~2.4GB | 高速 | **推奨** |
| CPU + FP32 | RAM 2.4GB | 遅い | GPU非対応環境用 |

---

## ✅ 動作確認済み

```
✓ 単一画像検出: 正常動作（スコア: 0.9935）
✓ GPU使用: CUDA 11.8対応
✓ メモリ効率: 2.4GB / 16GB使用
✓ 結果保存: results/single/に出力
```

---

**GPU環境で高速・高精度な損傷検出が可能です！** 🚀
