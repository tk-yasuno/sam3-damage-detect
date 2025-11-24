"""
検出されたrust領域のHSV色空間を分析
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import MODEL_CONFIG, RESULTS_DIR
from src.model_loader import create_sam_predictor
from src.image_processor import ImageProcessor, DamageDetector

def analyze_rust_hsv(image_path: str):
    """Rust領域のHSV値を分析"""
    
    print("="*60)
    print("Rust Color Analysis")
    print("="*60)
    
    # モデル読み込み
    print("\n[Loading Model]")
    predictor = create_sam_predictor(
        model_type=MODEL_CONFIG["model_type"],
        checkpoint_path=MODEL_CONFIG["checkpoint_path"],
        use_fp16=MODEL_CONFIG.get("use_fp16", False),
        use_quantization=MODEL_CONFIG.get("use_quantization", False)
    )
    
    # 画像読み込み
    print("\n[Loading Image]")
    image_processor = ImageProcessor()
    image = image_processor.load_and_preprocess(image_path)
    print(f"Image shape: {image.shape}")
    
    # RGB -> HSV変換
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Rust領域を検出
    print("\n[Detecting Rust Regions]")
    detector = DamageDetector(predictor)
    detector.set_image(image)
    
    # Rust prompts method
    rust_results = detector.detect_rust_regions_with_prompts()
    print(f"\nDetected {len(rust_results)} rust regions")
    
    # 各rust領域のHSV値を収集
    all_h_values = []
    all_s_values = []
    all_v_values = []
    
    for i, result in enumerate(rust_results):
        mask = result['mask']
        
        # マスク領域のHSV値を取得
        h_values = hsv[:, :, 0][mask]
        s_values = hsv[:, :, 1][mask]
        v_values = hsv[:, :, 2][mask]
        
        all_h_values.extend(h_values.tolist())
        all_s_values.extend(s_values.tolist())
        all_v_values.extend(v_values.tolist())
        
        print(f"\n[Region {i+1}] Area: {result['area']}px, Score: {result['score']:.4f}")
        print(f"  H: min={h_values.min()}, max={h_values.max()}, "
              f"mean={h_values.mean():.1f}, median={np.median(h_values):.1f}")
        print(f"  S: min={s_values.min()}, max={s_values.max()}, "
              f"mean={s_values.mean():.1f}, median={np.median(s_values):.1f}")
        print(f"  V: min={v_values.min()}, max={v_values.max()}, "
              f"mean={v_values.mean():.1f}, median={np.median(v_values):.1f}")
    
    # 全体の統計
    print("\n" + "="*60)
    print("Overall Statistics (all rust regions)")
    print("="*60)
    
    all_h = np.array(all_h_values)
    all_s = np.array(all_s_values)
    all_v = np.array(all_v_values)
    
    print(f"\nH (Hue):")
    print(f"  Range: [{all_h.min()}, {all_h.max()}]")
    print(f"  Mean: {all_h.mean():.1f}")
    print(f"  Median: {np.median(all_h):.1f}")
    print(f"  Percentiles: 5%={np.percentile(all_h, 5):.1f}, "
          f"95%={np.percentile(all_h, 95):.1f}")
    
    print(f"\nS (Saturation):")
    print(f"  Range: [{all_s.min()}, {all_s.max()}]")
    print(f"  Mean: {all_s.mean():.1f}")
    print(f"  Median: {np.median(all_s):.1f}")
    print(f"  Percentiles: 5%={np.percentile(all_s, 5):.1f}, "
          f"95%={np.percentile(all_s, 95):.1f}")
    
    print(f"\nV (Value):")
    print(f"  Range: [{all_v.min()}, {all_v.max()}]")
    print(f"  Mean: {all_v.mean():.1f}")
    print(f"  Median: {np.median(all_v):.1f}")
    print(f"  Percentiles: 5%={np.percentile(all_v, 5):.1f}, "
          f"95%={np.percentile(all_v, 95):.1f}")
    
    # 推奨されるHSV範囲を計算（5%～95%パーセンタイル）
    print("\n" + "="*60)
    print("Recommended HSV Ranges (5th-95th percentile)")
    print("="*60)
    
    h_lower = int(np.percentile(all_h, 5))
    h_upper = int(np.percentile(all_h, 95))
    s_lower = int(np.percentile(all_s, 5))
    s_upper = int(np.percentile(all_s, 95))
    v_lower = int(np.percentile(all_v, 5))
    v_upper = int(np.percentile(all_v, 95))
    
    print(f"\nH: [{h_lower}, {h_upper}]")
    print(f"S: [{s_lower}, {s_upper}]")
    print(f"V: [{v_lower}, {v_upper}]")
    
    print(f"\nCode suggestion:")
    print(f"lower_rust = np.array([{h_lower}, {s_lower}, {v_lower}])")
    print(f"upper_rust = np.array([{h_upper}, {s_upper}, {v_upper}])")
    
    # JSON形式で保存
    results = {
        "h_range": [int(all_h.min()), int(all_h.max())],
        "s_range": [int(all_s.min()), int(all_s.max())],
        "v_range": [int(all_v.min()), int(all_v.max())],
        "h_percentile_5_95": [h_lower, h_upper],
        "s_percentile_5_95": [s_lower, s_upper],
        "v_percentile_5_95": [v_lower, v_upper],
        "h_mean": float(all_h.mean()),
        "s_mean": float(all_s.mean()),
        "v_mean": float(all_v.mean()),
    }
    
    output_path = RESULTS_DIR / "rust_color_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    analyze_rust_hsv(args.image)
