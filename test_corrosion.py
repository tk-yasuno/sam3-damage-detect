"""
SAM3 Damage Detector - 腐食領域検出テスト
コンクリート全体と腐食領域の両方を検出
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.model_loader import create_sam_predictor
from src.image_processor import ImageProcessor, DamageDetector
from src.visualizer import Visualizer, ResultSaver
from src.config import MODEL_CONFIG, RESULTS_DIR


def detect_signboard_regions(image: np.ndarray, detector: DamageDetector) -> list:
    """
    工事看板領域を検出
    
    Args:
        image: RGB画像
        detector: SAM検出器
    
    Returns:
        看板マスクのリスト
    """
    # 白色看板を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # 連結成分分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    signboard_masks = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10000 or area > 100000:
            continue
        
        # SAMで精密なマスクを生成
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        masks, scores, _ = detector.predict_with_points(
            point_coords=[[cx, cy]],
            point_labels=[1],
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        mask_area = int(np.sum(best_mask))
        
        # SAMマスクのサイズチェック
        if mask_area >= 5000 and mask_area <= 50000:
            signboard_masks.append(best_mask)
    
    return signboard_masks


def apply_privacy_mask(image: np.ndarray, signboard_masks: list) -> np.ndarray:
    """
    工事看板にプライバシーマスクを適用
    
    Args:
        image: RGB画像
        signboard_masks: 看板マスクのリスト
    
    Returns:
        プライバシーマスク適用済み画像
    """
    privacy_image = image.copy()
    
    if len(signboard_masks) > 0:
        # 強いブラー処理
        blur_strength = 51
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        
        # 全ての看板マスクを適用
        for mask in signboard_masks:
            privacy_image = np.where(mask[:, :, np.newaxis], blurred, privacy_image)
    
    return privacy_image

def detect_corrosion_test(image_path: str):
    """
    腐食領域検出のテスト
    
    Args:
        image_path: テスト画像のパス
    """
    print("="*60)
    print("SAM3 Corrosion Detection Test")
    print("="*60)
    
    # モデルロード
    print("\n[Loading Model]")
    predictor = create_sam_predictor(
        checkpoint_path=str(MODEL_CONFIG["checkpoint_path"]),
        model_type=MODEL_CONFIG["model_type"],
        use_fp16=MODEL_CONFIG["use_fp16"],
        use_quantization=MODEL_CONFIG["use_quantization"]
    )
    
    # 画像読み込み
    print("\n[Loading Image]")
    image_processor = ImageProcessor()
    image = image_processor.load_and_preprocess(image_path)
    print(f"Image shape: {image.shape}")
    
    # 損傷検出
    print("\n[Detecting Multiple Regions]")
    detector = DamageDetector(predictor)
    detector.set_image(image)
    
    # 複数領域を検出（サンプリング1024点で細かい腐食領域も検出）
    print("\n[Method 1: Auto-detect with color information (1024 sampling points)]")
    results = detector.detect_corrosion_areas(num_points=1024, use_color=True)
    
    print(f"\nDetected {len(results)} regions:")
    for i, result in enumerate(results):
        rust_info = f" | Rust: {result['rust_overlap']:>6} px" if result.get('is_rust_based') else ""
        print(f"  [{i+1}] Type: {result['type']:<18} | Score: {result['score']:.4f} | "
              f"Area: {result['area']:>8} px ({result['area_ratio']*100:.1f}%){rust_info}")
    
    # 色ベースの腐食領域検出も試す
    print("\n[Method 2: Rust detection with color prompts]")
    rust_results = detector.detect_rust_regions_with_prompts()
    
    print(f"\nDetected {len(rust_results)} rust regions:")
    for i, result in enumerate(rust_results):
        print(f"  [{i+1}] Type: {result['type']:<18} | Score: {result['score']:.4f} | "
              f"Area: {result['area']:>8} px | Centroid: {result['centroid']}")
    
    # 鉄筋の配置パターンを分析
    print("\n[Analyzing Rebar Pattern]")
    pattern = detector.find_rebar_pattern(rust_results)
    print(f"Detected pattern:")
    print(f"  Lines: {pattern['num_lines']}")
    print(f"  Average spacing: {pattern['spacing']:.1f} px")
    print(f"  Angle: {pattern['angle']:.2f}°")
    
    # パターンに基づいて追加プロンプトを生成
    if pattern['num_lines'] >= 2:
        print("\n[Method 3: Pattern-based additional detection]")
        additional_prompts = detector.generate_pattern_based_prompts(
            pattern, 
            image.shape[:2]
        )
        print(f"Generated {len(additional_prompts)} additional prompts")
        
        # 追加プロンプトで検出
        additional_results = []
        if additional_prompts:
            # バッチでSAM推論
            rust_mask = detector.detect_rust_color(image)
            
            for prompt in additional_prompts:
                # プロンプト位置が錆色領域内にあるかチェック
                px, py = prompt
                if rust_mask[py, px]:
                    masks, scores, _ = detector.predict_with_points(
                        point_coords=[[px, py]],
                        point_labels=[1],
                        multimask_output=True
                    )
                    
                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]
                    mask_area = int(np.sum(best_mask))
                    
                    # 既存の検出と重複していないかチェック
                    is_duplicate = False
                    for existing in rust_results:
                        overlap = np.sum(np.logical_and(best_mask, existing['mask']))
                        if overlap > mask_area * 0.5:  # 50%以上重複
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and mask_area >= 70 and mask_area < 2000:
                        # 形状チェック
                        y_mask, x_mask = np.where(best_mask)
                        if len(y_mask) > 0:
                            width = x_mask.max() - x_mask.min() + 1
                            height = y_mask.max() - y_mask.min() + 1
                            bbox_area = width * height
                            aspect_ratio = max(width, height) / max(min(width, height), 1)
                            fill_ratio = mask_area / max(bbox_area, 1)
                            
                            # 鉄筋形状チェック
                            if aspect_ratio >= 2.0 or (aspect_ratio >= 1.5 and mask_area < 300):
                                if not (fill_ratio >= 0.7 and aspect_ratio < 2.5):
                                    additional_results.append({
                                        'mask': best_mask,
                                        'score': float(scores[best_idx]),
                                        'area': mask_area,
                                        'centroid': (px, py),
                                        'type': 'rust_corrosion',
                                        'aspect_ratio': aspect_ratio
                                    })
        
        print(f"Detected {len(additional_results)} new rust regions from pattern-based search")
        for i, result in enumerate(additional_results):
            print(f"  [{i+1}] Type: {result['type']:<18} | Score: {result['score']:.4f} | "
                  f"Area: {result['area']:>8} px | Centroid: {result['centroid']}")
        
        # 追加結果をrust_resultsに統合
        rust_results.extend(additional_results)
    
    # 両方の結果を統合（損傷のないコンクリート領域は除外）
    all_results = []
    
    # Method 1の結果から損傷領域のみを追加（ただし2000px未満の小さい領域のみ）
    for r in results:
        if r['type'] in ['rust_corrosion', 'corrosion'] and r['area'] < 2000:
            all_results.append(r)
        # damageタイプは除外（損傷のないコンクリート領域）
        # 2000px以上の大きい領域も除外（広いコンクリート領域）
    
    # Method 2のrust領域を追加
    all_results.extend(rust_results)
    
    # 可視化
    print("\n[Visualization]")
    visualizer = Visualizer(alpha=0.5)
    
    # 結果保存
    print("\n[Saving Results]")
    output_dir = RESULTS_DIR / "corrosion_test"
    saver = ResultSaver(output_dir)
    image_name = Path(image_path).stem
    
    # rust_corrosion領域を1枚にまとめる
    rust_results = [r for r in all_results if r['type'] == 'rust_corrosion']
    other_results = [r for r in all_results if r['type'] != 'rust_corrosion']
    
    if rust_results:
        print(f"\n[Combined Rust Corrosion: {len(rust_results)} regions]")
        # 複数のrust_corrosionマスクを異なる色でオーバーレイ
        combined_overlay = image.copy()
        rust_colors = [
            (255, 0, 0),      # 赤
            (255, 128, 0),    # オレンジ
            (255, 200, 0),    # 濃い黄色
            (255, 255, 0),    # 黄色
            (128, 255, 0),    # 黄緑
            (0, 255, 0),      # 緑
            (0, 255, 128),    # 青緑
            (0, 255, 255),    # シアン
            (0, 128, 255),    # 空色
            (0, 0, 255),      # 青
            (128, 0, 255),    # 紫
            (255, 0, 255),    # マゼンタ
        ]
        
        for i, rust_result in enumerate(rust_results):
            color = rust_colors[i % len(rust_colors)]
            mask = rust_result['mask']
            combined_overlay[mask] = (
                combined_overlay[mask] * (1 - visualizer.alpha) +
                np.array(color) * visualizer.alpha
            ).astype(np.uint8)
        
        # 統合画像を保存
        combined_path = saver.save_image(
            combined_overlay,
            f"{image_name}_rust_corrosion_combined.png"
        )
        print(f"  ✓ Combined: {combined_path.name}")
        
        # プライバシー保護版を作成（工事看板をマスキング）
        print("\n[Applying Privacy Protection]")
        signboard_masks = detect_signboard_regions(image, detector)
        print(f"  Detected {len(signboard_masks)} signboard regions")
        
        if len(signboard_masks) > 0:
            # 元画像にプライバシーマスク適用
            privacy_original = apply_privacy_mask(image, signboard_masks)
            privacy_original_path = saver.save_image(
                privacy_original,
                f"{image_name}_privacy_masked.png"
            )
            print(f"  ✓ Privacy-masked original: {privacy_original_path.name}")
            
            # 統合画像にもプライバシーマスク適用
            privacy_combined = apply_privacy_mask(combined_overlay, signboard_masks)
            privacy_combined_path = saver.save_image(
                privacy_combined,
                f"{image_name}_rust_combined_privacy.png"
            )
            print(f"  ✓ Privacy-masked combined: {privacy_combined_path.name}")
        
        # 統合マスクも保存
        combined_mask = np.zeros_like(rust_results[0]['mask'])
        for rust_result in rust_results:
            combined_mask = np.logical_or(combined_mask, rust_result['mask'])
        combined_mask_path = saver.save_mask(
            combined_mask,
            f"{image_name}_rust_corrosion_combined_mask.png"
        )
    
    # その他の領域は個別に保存
    if other_results:
        print(f"\n[Other Regions: {len(other_results)}]")
        for i, result in enumerate(other_results):
            overlay = visualizer.show_mask(image, result['mask'], show=False)
            overlay_path = saver.save_image(
                overlay,
                f"{image_name}_{result['type']}_{i+1}_overlay.png"
            )
            print(f"  [{i+1}] {result['type']}: {overlay_path.name}")
    
    # 統計情報を表示
    concrete_results = [r for r in all_results if r['type'] == 'concrete']
    rust_results_all = [r for r in all_results if 'rust' in r['type']]
    corrosion_results = [r for r in all_results if r['type'] == 'corrosion']
    
    print(f"\n✓ Detection completed!")
    print(f"  Concrete regions: {len(concrete_results)}")
    print(f"  Rust corrosion regions: {len(rust_results_all)}")
    print(f"  Other corrosion regions: {len(corrosion_results)}")
    print(f"  Total regions: {len(all_results)}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="腐食領域検出テスト")
    parser.add_argument(
        "--image",
        type=str,
        default="data/1_Test_images-kensg/kensg-rebarexposureRb_001.png",
        help="テスト画像のパス"
    )
    
    args = parser.parse_args()
    
    try:
        detect_corrosion_test(args.image)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
