"""
SAM3 + OCR - 工事看板の文字情報抽出テスト
看板領域をSAMで検出し、OCRで文字を読み取る
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

try:
    import pytesseract
    from PIL import Image
    
    # Tesseractのパスを設定（Windows）
    import os
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\yasun\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract not installed. Install with: pip install pytesseract pillow")


def preprocess_for_ocr(img):
    """
    OCR用の高度な画像前処理（複数手法）
    
    Args:
        img: 入力画像（BGR）
    
    Returns:
        dict: 各前処理手法の二値化画像
    """
    # グレースケール化
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 解像度向上（300 DPI相当にスケール）
    scale_factor = 2.0
    height, width = gray.shape
    gray = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), 
                      interpolation=cv2.INTER_CUBIC)
    
    # ノイズ除去（メディアンフィルタ）
    denoised = cv2.medianBlur(gray, 3)
    
    # ノイズ除去（バイラテラルフィルタ - エッジ保持）
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 複数の二値化手法を試す
    methods = {}
    
    # 1. Otsu法（大域的閾値処理）
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods['otsu'] = otsu
    
    # 2. 適応的二値化（ガウシアン）
    adaptive_gaussian = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )
    methods['adaptive_gaussian'] = adaptive_gaussian
    
    # 3. 適応的二値化（平均）
    adaptive_mean = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2
    )
    methods['adaptive_mean'] = adaptive_mean
    
    # 4. Sauvola法（局所適応的二値化 - 文書画像に最適）
    window_size = 15
    k = 0.5
    R = 128
    mean = cv2.blur(enhanced.astype(float), (window_size, window_size))
    mean_sq = cv2.blur((enhanced.astype(float))**2, (window_size, window_size))
    std = np.sqrt(mean_sq - mean**2)
    threshold = mean * (1 + k * ((std / R) - 1))
    sauvola = np.where(enhanced > threshold, 255, 0).astype(np.uint8)
    methods['sauvola'] = sauvola
    
    # 形態学的処理（ノイズ除去と文字修復）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    for name in methods:
        # 小さなノイズを除去（Opening）
        methods[name] = cv2.morphologyEx(methods[name], cv2.MORPH_OPEN, kernel, iterations=1)
        # 文字の穴を埋める（Closing）
        methods[name] = cv2.morphologyEx(methods[name], cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return methods


def detect_signboard_color(image: np.ndarray) -> np.ndarray:
    """
    工事看板の色（白色背景）を検出
    
    Args:
        image: RGB画像
    
    Returns:
        看板領域のバイナリマスク
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 白色看板の範囲（高明度、低彩度）
    # H: 任意, S: 0-50（低彩度）, V: 180-255（高明度）
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    return white_mask > 0


def extract_signboard_text(image_path: str):
    """
    工事看板から文字情報を抽出
    
    Args:
        image_path: テスト画像のパス
    """
    print("="*60)
    print("SAM3 + OCR - Signboard Text Extraction")
    print("="*60)
    
    if not OCR_AVAILABLE:
        print("\n❌ OCR機能が利用できません")
        print("以下のコマンドでインストールしてください:")
        print("  pip install pytesseract pillow")
        print("\nまた、Tesseract OCRエンジンのインストールが必要です:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        return
    
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
    print("\n[Detecting Signboard]")
    detector = DamageDetector(predictor)
    detector.set_image(image)
    
    # 白色看板領域を検出
    signboard_mask = detect_signboard_color(image)
    signboard_area = np.sum(signboard_mask)
    print(f"White signboard detection: {signboard_area} pixels ({signboard_area / (image.shape[0] * image.shape[1]) * 100:.1f}%)")
    
    # 連結成分分析で個々の看板領域を特定
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        signboard_mask.astype(np.uint8), connectivity=8
    )
    
    signboard_regions = []
    for i in range(1, num_labels):  # 0は背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10000 or area > 100000:  # 看板サイズの範囲をフィルタリング（10000-100000px）
            continue
        
        # 各看板領域の中心点を取得
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # SAMで精密なマスクを生成
        masks, scores, _ = detector.predict_with_points(
            point_coords=[[cx, cy]],
            point_labels=[1],
            multimask_output=True
        )
        
        # 最高スコアのマスクを選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        mask_area = int(np.sum(best_mask))
        
        # SAMマスクのサイズチェック（5,000 ~ 100,000 px）
        if mask_area < 5000 or mask_area > 100000:
            continue
        
        # バウンディングボックスを取得
        y_coords, x_coords = np.where(best_mask)
        if len(y_coords) > 0:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            signboard_regions.append({
                'mask': best_mask,
                'score': float(best_score),
                'area': mask_area,
                'bbox': (x_min, y_min, width, height),
                'centroid': (cx, cy)
            })
    
    print(f"\nDetected {len(signboard_regions)} signboard regions:")
    for i, region in enumerate(signboard_regions):
        print(f"  [{i+1}] Score: {region['score']:.4f} | Area: {region['area']:>8} px | "
              f"BBox: {region['bbox']}")
    
    # OCR処理
    if len(signboard_regions) > 0:
        print("\n[OCR Text Extraction]")
        output_dir = RESULTS_DIR / "ocr_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        saver = ResultSaver(output_dir)
        
        all_text_results = []
        
        for i, region in enumerate(signboard_regions):
            x_min, y_min, width, height = region['bbox']
            
            # 看板領域を切り出し（マージン追加）
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.shape[1], x_min + width + margin * 2)
            y_max = min(image.shape[0], y_min + height + margin * 2)
            
            signboard_crop = image[y_min:y_max, x_min:x_max]
            
            # RGB to BGR (for OpenCV)
            signboard_bgr = cv2.cvtColor(signboard_crop, cv2.COLOR_RGB2BGR)
            
            # 複数の前処理手法で二値化
            binary_methods = preprocess_for_ocr(signboard_bgr)
            
            # OCR実行（複数の前処理手法で試行）
            try:
                # Tesseract設定
                config = '--psm 6 --oem 3'  # PSM 6: 単一ブロックのテキスト
                
                # 各手法で日本語と英語のOCRを実行
                ocr_results = {}
                best_method = None
                best_text = ""
                max_length = 0
                
                for method_name, binary in binary_methods.items():
                    pil_image = Image.fromarray(binary)
                    
                    # 日本語OCR
                    text_ja = pytesseract.image_to_string(pil_image, lang='jpn', config=config)
                    
                    # 英語OCR
                    text_en = pytesseract.image_to_string(pil_image, lang='eng', config=config)
                    
                    ocr_results[method_name] = {
                        'text_japanese': text_ja.strip(),
                        'text_english': text_en.strip()
                    }
                    
                    # 最も長いテキストを取得した手法を記録（より多くの文字を認識できた）
                    combined_length = len(text_ja.strip()) + len(text_en.strip())
                    if combined_length > max_length:
                        max_length = combined_length
                        best_method = method_name
                        best_text = text_ja.strip() if len(text_ja.strip()) > len(text_en.strip()) else text_en.strip()
                
                print(f"\n[Signboard {i+1}] - Best Method: {best_method}")
                print(f"  Best Text ({len(best_text)} chars):")
                for line in best_text.split('\n'):
                    if line.strip():
                        print(f"    {line}")
                
                print(f"\n  All OCR Results:")
                for method, texts in ocr_results.items():
                    ja_len = len(texts['text_japanese'])
                    en_len = len(texts['text_english'])
                    print(f"    [{method}] JA:{ja_len} EN:{en_len}")
                    if ja_len > 0:
                        preview = texts['text_japanese'][:80].replace('\n', ' ')
                        print(f"      JA: {preview}")
                    if en_len > 0:
                        preview = texts['text_english'][:80].replace('\n', ' ')
                        print(f"      EN: {preview}")
                
                # 文字が含まれているかチェック（いずれかの手法で有意な文字が検出された）
                has_valid_text = False
                max_valid_chars = 0
                
                for method_texts in ocr_results.values():
                    ja_text = method_texts['text_japanese'].strip()
                    en_text = method_texts['text_english'].strip()
                    
                    # 有効な文字のみをカウント（記号、空白、改行を除外）
                    import re
                    # 日本語文字（ひらがな、カタカナ、漢字）、英数字のみ
                    ja_valid = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\w]', '', ja_text)
                    en_valid = re.sub(r'[^\w]', '', en_text)
                    
                    valid_char_count = len(ja_valid) + len(en_valid)
                    max_valid_chars = max(max_valid_chars, valid_char_count)
                    
                    # 10文字以上の有効な文字が検出された場合、有効な看板とみなす
                    if valid_char_count >= 10:
                        has_valid_text = True
                        break
                
                if not has_valid_text:
                    print(f"  ⚠️ Insufficient text detected ({max_valid_chars} valid chars) - Skipping (not a signboard)")
                    continue
                
                all_text_results.append({
                    'signboard_index': i + 1,
                    'bbox': [int(x) for x in region['bbox']],
                    'score': region['score'],
                    'best_method': best_method,
                    'best_text': best_text,
                    'text_length': len(best_text),
                    'ocr_results': ocr_results
                })
                
                # 切り出し画像を保存
                crop_path = output_dir / f"signboard_{i+1}_crop.png"
                cv2.imwrite(str(crop_path), signboard_bgr)
                
                # 各手法の二値化画像を保存
                for method_name, binary in binary_methods.items():
                    binary_path = output_dir / f"signboard_{i+1}_binary_{method_name}.png"
                    cv2.imwrite(str(binary_path), binary)
                
            except Exception as e:
                print(f"  ✗ OCR Error: {e}")
        
        # 結果をJSONで保存
        import json
        json_path = output_dir / "ocr_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_text_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ OCR completed!")
        print(f"  Valid signboards (with text): {len(all_text_results)} / {len(signboard_regions)} detected regions")
        print(f"  Results saved to: {output_dir}")
        print(f"  JSON saved to: {json_path}")
        
        # プライバシー保護画像を作成（看板をマスキング）
        print("\n[Privacy Protection]")
        privacy_image = image.copy()
        for region in signboard_regions:
            mask = region['mask']
            # 看板領域を強いブラー処理でマスキング
            blur_strength = 51  # 奇数である必要がある
            blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
            # マスク領域のみブラー画像で置き換え
            privacy_image = np.where(mask[:, :, np.newaxis], blurred, privacy_image)
        
        # プライバシー保護画像を保存
        image_name = Path(image_path).stem
        privacy_path = saver.save_image(
            privacy_image,
            f"{image_name}_privacy_masked.png"
        )
        print(f"  Privacy-masked image saved: {privacy_path.name}")
        
        # 可視化（検出領域を緑色でオーバーレイ）
        visualizer = Visualizer(alpha=0.3)
        combined_overlay = image.copy()
        
        for region in signboard_regions:
            mask = region['mask']
            color = (0, 255, 0)  # 緑色
            combined_overlay[mask] = (
                combined_overlay[mask] * (1 - visualizer.alpha) +
                np.array(color) * visualizer.alpha
            ).astype(np.uint8)
        
        combined_path = saver.save_image(
            combined_overlay,
            f"{image_name}_signboard_detected.png"
        )
        print(f"  Visualization saved: {combined_path.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="工事看板OCRテスト")
    parser.add_argument(
        "--image",
        type=str,
        default="data/1_Test_images-kensg/kensg-rebarexposureRb_001.png",
        help="テスト画像のパス"
    )
    
    args = parser.parse_args()
    
    try:
        extract_signboard_text(args.image)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
