"""
SAM3 Damage Detector v0.3 - 鉄筋腐食領域バッチ検出 + プライバシー保護
254枚の画像を一括処理
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.model_loader import create_sam_predictor
from src.image_processor import ImageProcessor, DamageDetector
from src.visualizer import Visualizer, ResultSaver
from src.config import MODEL_CONFIG, RESULTS_DIR


def detect_signboard_regions(image: np.ndarray, detector: DamageDetector) -> list:
    """工事看板領域を検出"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    signboard_masks = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10000 or area > 100000:
            continue
        
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        masks, scores, _ = detector.predict_with_points(
            point_coords=[[cx, cy]],
            point_labels=[1],
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        mask_area = int(np.sum(best_mask))
        
        if mask_area >= 5000 and mask_area <= 100000:
            signboard_masks.append(best_mask)
    
    return signboard_masks


def apply_privacy_mask(image: np.ndarray, signboard_masks: list) -> np.ndarray:
    """工事看板にプライバシーマスクを適用"""
    privacy_image = image.copy()
    
    if len(signboard_masks) > 0:
        blur_strength = 51
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        
        for mask in signboard_masks:
            privacy_image = np.where(mask[:, :, np.newaxis], blurred, privacy_image)
    
    return privacy_image


def batch_corrosion_detection(input_dir: str, max_images: int = None):
    """
    複数画像の鉄筋腐食領域を一括検出
    
    Args:
        input_dir: 入力画像ディレクトリ
        max_images: 処理する最大画像数（Noneの場合は全て）
    """
    print("="*60)
    print("SAM3 Corrosion Detection - Batch Processing v0.3")
    print("="*60)
    
    # モデルロード
    print("\n[Loading Model]")
    predictor = create_sam_predictor(
        checkpoint_path=str(MODEL_CONFIG["checkpoint_path"]),
        model_type=MODEL_CONFIG["model_type"],
        use_fp16=MODEL_CONFIG["use_fp16"],
        use_quantization=MODEL_CONFIG["use_quantization"]
    )
    
    # 入力画像の取得
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n[Input Directory]")
    print(f"Path: {input_path}")
    print(f"Total images: {len(image_files)}")
    
    # 出力ディレクトリ
    output_dir = RESULTS_DIR / "batch_corrosion_v0.3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像処理
    image_processor = ImageProcessor()
    detector = DamageDetector(predictor)
    visualizer = Visualizer(alpha=0.5)
    saver = ResultSaver(output_dir)
    
    # 統計情報
    batch_stats = {
        'version': 'v0.3',
        'timestamp': datetime.now().isoformat(),
        'total_images': len(image_files),
        'processed_images': 0,
        'failed_images': 0,
        'total_rust_regions': 0,
        'processing_times': [],
        'results': []
    }
    
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
    
    print(f"\n[Processing Images]")
    print("-" * 60)
    
    for idx, image_file in enumerate(image_files, 1):
        start_time = time.time()
        
        try:
            print(f"\n[{idx}/{len(image_files)}] {image_file.name}")
            
            # 画像読み込み
            image = image_processor.load_and_preprocess(str(image_file))
            detector.set_image(image)
            
            # Method 1: Auto-detect
            results = detector.detect_corrosion_areas(num_points=1024, use_color=True)
            
            # Method 2: Rust detection with color prompts
            rust_results = detector.detect_rust_regions_with_prompts()
            print(f"  Method 2: {len(rust_results)} rust regions")
            
            # 錆領域のHSV値をサンプリング
            rust_hsv_samples = []
            if rust_results:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                for rust in rust_results:
                    # 各錆領域のピクセルからHSV値を抽出
                    rust_pixels = hsv_image[rust['mask']]
                    if len(rust_pixels) > 0:
                        # 統計情報を記録
                        h_values = rust_pixels[:, 0]
                        s_values = rust_pixels[:, 1]
                        v_values = rust_pixels[:, 2]
                        
                        rust_hsv_samples.append({
                            'area': rust['area'],
                            'h_mean': float(np.mean(h_values)),
                            'h_std': float(np.std(h_values)),
                            'h_min': int(np.min(h_values)),
                            'h_max': int(np.max(h_values)),
                            'h_percentile_5': int(np.percentile(h_values, 5)),
                            'h_percentile_95': int(np.percentile(h_values, 95)),
                            's_mean': float(np.mean(s_values)),
                            's_std': float(np.std(s_values)),
                            's_min': int(np.min(s_values)),
                            's_max': int(np.max(s_values)),
                            's_percentile_5': int(np.percentile(s_values, 5)),
                            's_percentile_95': int(np.percentile(s_values, 95)),
                            'v_mean': float(np.mean(v_values)),
                            'v_std': float(np.std(v_values)),
                            'v_min': int(np.min(v_values)),
                            'v_max': int(np.max(v_values)),
                            'v_percentile_5': int(np.percentile(v_values, 5)),
                            'v_percentile_95': int(np.percentile(v_values, 95)),
                        })
            
            # Method 3: Pattern-based additional detection
            pattern = detector.find_rebar_pattern(rust_results)
            additional_results = []
            
            if pattern and pattern.get('num_lines', 0) >= 2:
                additional_prompts = detector.generate_pattern_based_prompts(
                    pattern, 
                    image.shape[:2]
                )
                
                rust_mask = detector.detect_rust_color(image)
                
                for prompt in additional_prompts:
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
                        
                        # 重複チェック
                        is_duplicate = False
                        for existing in rust_results:
                            overlap = np.sum(np.logical_and(best_mask, existing['mask']))
                            if overlap > mask_area * 0.5:
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
                
                rust_results.extend(additional_results)
                print(f"  Method 3: +{len(additional_results)} regions (pattern-based)")
            
            # 統合画像の生成
            combined_overlay = image.copy()
            for i, rust_result in enumerate(rust_results):
                color = rust_colors[i % len(rust_colors)]
                mask = rust_result['mask']
                combined_overlay[mask] = (
                    combined_overlay[mask] * (1 - visualizer.alpha) +
                    np.array(color) * visualizer.alpha
                ).astype(np.uint8)
            
            # 保存
            image_name = image_file.stem
            combined_path = saver.save_image(
                combined_overlay,
                f"{image_name}_rust_combined.png"
            )
            
            # プライバシー保護版を作成
            signboard_masks = detect_signboard_regions(image, detector)
            signboard_count = len(signboard_masks)
            
            if signboard_count > 0:
                # 統合画像にプライバシーマスク適用
                privacy_combined = apply_privacy_mask(combined_overlay, signboard_masks)
                privacy_path = saver.save_image(
                    privacy_combined,
                    f"{image_name}_rust_combined_privacy.png"
                )
                print(f"  Privacy: {signboard_count} signboard(s) masked")
            
            # 統計情報を記録
            processing_time = time.time() - start_time
            batch_stats['processing_times'].append(processing_time)
            batch_stats['processed_images'] += 1
            batch_stats['total_rust_regions'] += len(rust_results)
            
            image_stat = {
                'filename': image_file.name,
                'rust_regions': len(rust_results),
                'signboard_regions': signboard_count,
                'pattern_lines': pattern.get('num_lines', 0) if pattern else 0,
                'pattern_spacing': pattern.get('spacing', 0) if pattern else 0,
                'pattern_angle': pattern.get('angle', 0) if pattern else 0,
                'processing_time': processing_time,
                'rust_areas': [r['area'] for r in rust_results],
                'rust_scores': [r['score'] for r in rust_results],
                'rust_hsv_samples': rust_hsv_samples
            }
            batch_stats['results'].append(image_stat)
            
            print(f"  ✓ Total: {len(rust_results)} rust regions")
            print(f"  ✓ Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            batch_stats['failed_images'] += 1
            batch_stats['results'].append({
                'filename': image_file.name,
                'error': str(e)
            })
            continue
    
    # 統計情報の保存
    print("\n" + "="*60)
    print("[Batch Processing Summary]")
    print("="*60)
    print(f"Total images: {batch_stats['total_images']}")
    print(f"Processed successfully: {batch_stats['processed_images']}")
    print(f"Failed: {batch_stats['failed_images']}")
    print(f"Total rust regions detected: {batch_stats['total_rust_regions']}")
    
    if batch_stats['processing_times']:
        avg_time = np.mean(batch_stats['processing_times'])
        total_time = np.sum(batch_stats['processing_times'])
        print(f"\nProcessing time:")
        print(f"  Average: {avg_time:.2f}s per image")
        print(f"  Total: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    # 統計分析
    if batch_stats['processed_images'] > 0:
        all_rust_counts = [r['rust_regions'] for r in batch_stats['results'] if 'rust_regions' in r]
        all_rust_areas = []
        all_rust_scores = []
        
        for r in batch_stats['results']:
            if 'rust_areas' in r:
                all_rust_areas.extend(r['rust_areas'])
            if 'rust_scores' in r:
                all_rust_scores.extend(r['rust_scores'])
        
        print(f"\nRust regions statistics:")
        print(f"  Average per image: {np.mean(all_rust_counts):.1f}")
        print(f"  Min: {np.min(all_rust_counts)}, Max: {np.max(all_rust_counts)}")
        
        if all_rust_areas:
            print(f"\nRust area statistics:")
            print(f"  Average: {np.mean(all_rust_areas):.0f} px")
            print(f"  Median: {np.median(all_rust_areas):.0f} px")
            print(f"  Range: {np.min(all_rust_areas):.0f} - {np.max(all_rust_areas):.0f} px")
        
        if all_rust_scores:
            print(f"\nDetection score statistics:")
            print(f"  Average: {np.mean(all_rust_scores):.4f}")
            print(f"  Min: {np.min(all_rust_scores):.4f}, Max: {np.max(all_rust_scores):.4f}")
    
    # JSON保存
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_stats, f, indent=2, ensure_ascii=False)
    
    # HSV範囲の統計分析
    print("\n" + "="*60)
    print("[HSV Color Space Analysis]")
    print("="*60)
    
    all_hsv_samples = []
    for result in batch_stats['results']:
        if 'rust_hsv_samples' in result:
            all_hsv_samples.extend(result['rust_hsv_samples'])
    
    if all_hsv_samples:
        # 全錆領域のHSV統計
        h_5th = np.percentile([s['h_percentile_5'] for s in all_hsv_samples], 5)
        h_95th = np.percentile([s['h_percentile_95'] for s in all_hsv_samples], 95)
        s_5th = np.percentile([s['s_percentile_5'] for s in all_hsv_samples], 5)
        s_95th = np.percentile([s['s_percentile_95'] for s in all_hsv_samples], 95)
        v_5th = np.percentile([s['v_percentile_5'] for s in all_hsv_samples], 5)
        v_95th = np.percentile([s['v_percentile_95'] for s in all_hsv_samples], 95)
        
        print(f"\nRecommended HSV range (5th-95th percentile across all images):")
        print(f"  H: [{int(h_5th)}, {int(h_95th)}]")
        print(f"  S: [{int(s_5th)}, {int(s_95th)}]")
        print(f"  V: [{int(v_5th)}, {int(v_95th)}]")
        
        print(f"\nCurrent HSV range (from image_processor.py):")
        print(f"  H: [0, 177]")
        print(f"  S: [31, 135]")
        print(f"  V: [28, 142]")
        
        # HSV統計をCSVで保存
        hsv_analysis_path = output_dir / "hsv_color_analysis.csv"
        with open(hsv_analysis_path, 'w', encoding='utf-8') as f:
            f.write("filename,region_index,area,h_mean,h_std,h_min,h_max,h_5th,h_95th,")
            f.write("s_mean,s_std,s_min,s_max,s_5th,s_95th,")
            f.write("v_mean,v_std,v_min,v_max,v_5th,v_95th\n")
            
            for result in batch_stats['results']:
                if 'rust_hsv_samples' in result:
                    for idx, sample in enumerate(result['rust_hsv_samples']):
                        f.write(f"{result['filename']},{idx},{sample['area']},")
                        f.write(f"{sample['h_mean']:.2f},{sample['h_std']:.2f},")
                        f.write(f"{sample['h_min']},{sample['h_max']},")
                        f.write(f"{sample['h_percentile_5']},{sample['h_percentile_95']},")
                        f.write(f"{sample['s_mean']:.2f},{sample['s_std']:.2f},")
                        f.write(f"{sample['s_min']},{sample['s_max']},")
                        f.write(f"{sample['s_percentile_5']},{sample['s_percentile_95']},")
                        f.write(f"{sample['v_mean']:.2f},{sample['v_std']:.2f},")
                        f.write(f"{sample['v_min']},{sample['v_max']},")
                        f.write(f"{sample['v_percentile_5']},{sample['v_percentile_95']}\n")
        
        print(f"\n✓ HSV analysis saved to: {hsv_analysis_path}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="鉄筋腐食領域バッチ検出 v0.3 + プライバシー保護")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/1_Test_images-kensg",
        help="入力画像ディレクトリ"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="処理する最大画像数（デフォルト: 10）"
    )
    
    args = parser.parse_args()
    
    try:
        batch_corrosion_detection(args.input_dir, args.max_images)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
