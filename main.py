"""
SAM3 Damage Detector MVP - Main Script
GPU対応構造物損傷検出システム

使用方法:
    # GPU使用時（推奨）
    python main.py --mode single --image <画像パス>
    
    # CPU使用時
    python main.py --mode single --image <画像パス> --no_fp16 --no_quantization
    
    # バッチ処理
    python main.py --mode batch --input_dir <ディレクトリパス>
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートとsrcをパスに追加
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.model_loader import create_sam_predictor
from src.image_processor import ImageProcessor, BatchProcessor
from src.visualizer import Visualizer, ResultSaver, create_comparison_figure
from src.config import (
    MODEL_CONFIG, IMAGE_CONFIG, DETECTION_CONFIG,
    VISUALIZATION_CONFIG, TEST_IMAGES_DIR, RESULTS_DIR, MODEL_URLS
)


def check_environment():
    """環境チェック"""
    print("="*60)
    print("SAM3 Damage Detector MVP")
    print("="*60)
    print("\n[Environment Check]")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
        
        # GPUメモリの推奨事項
        if gpu_memory_gb < 8:
            print("⚠️  Warning: GPU memory < 8GB. バッチ処理時は --max_images で画像数を制限してください。")
    else:
        print("⚠️  Warning: CUDA not available. Running on CPU (slower)")
        print("    CPUでの実行には --no_fp16 --no_quantization オプションを使用してください。")
    
    print()


def check_model_file():
    """モデルファイルの存在確認"""
    checkpoint_path = MODEL_CONFIG["checkpoint_path"]
    
    if not checkpoint_path.exists():
        print(f"⚠️  Model file not found: {checkpoint_path}")
        print(f"\nPlease download the model file from:")
        print(f"  {MODEL_URLS[MODEL_CONFIG['model_type']]}")
        print(f"\nAnd save it to:")
        print(f"  {checkpoint_path}")
        return False
    
    print(f"✓ Model file found: {checkpoint_path}")
    return True


def single_image_detection(image_path: str, output_dir: str = None):
    """
    単一画像の損傷検出
    
    Args:
        image_path: 画像ファイルのパス
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
    """
    print("\n[Mode: Single Image Detection]")
    print(f"Input image: {image_path}")
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = RESULTS_DIR / "single"
    else:
        output_dir = Path(output_dir)
    
    # モデルロード
    print("\n[Loading Model]")
    predictor = create_sam_predictor(
        checkpoint_path=str(MODEL_CONFIG["checkpoint_path"]),
        model_type=MODEL_CONFIG["model_type"],
        use_fp16=MODEL_CONFIG["use_fp16"],
        use_quantization=MODEL_CONFIG["use_quantization"]
    )
    
    # 画像処理
    print("\n[Processing Image]")
    image_processor = ImageProcessor(target_size=IMAGE_CONFIG["target_size"])
    image = image_processor.load_and_preprocess(image_path)
    print(f"Image shape: {image.shape}")
    
    # 損傷検出
    print("\n[Detecting Damage]")
    from src.image_processor import DamageDetector
    detector = DamageDetector(predictor)
    detector.set_image(image)
    
    # 自動検出
    best_mask, best_score = detector.auto_detect(
        num_points=DETECTION_CONFIG["num_auto_points"]
    )
    print(f"Detection score: {best_score:.4f}")
    
    # 可視化
    print("\n[Visualization]")
    visualizer = Visualizer(alpha=VISUALIZATION_CONFIG["alpha"])
    overlay = visualizer.show_mask(
        image,
        best_mask,
        color=VISUALIZATION_CONFIG["mask_color"],
        show=VISUALIZATION_CONFIG["show_plots"]
    )
    
    # 結果保存
    print("\n[Saving Results]")
    saver = ResultSaver(output_dir)
    result = {
        'filename': Path(image_path).name,
        'image': image,
        'mask': best_mask,
        'score': float(best_score)
    }
    
    saved_paths = saver.save_result(result, visualizer)
    
    # 比較図の作成
    comparison_path = output_dir / f"{Path(image_path).stem}_comparison.png"
    create_comparison_figure(image, overlay, best_mask, best_score, comparison_path)
    
    print("\n✓ Detection completed!")
    print(f"\nResults saved to: {output_dir}")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")
    print(f"  comparison: {comparison_path}")


def batch_detection(input_dir: str, output_dir: str = None, max_images: int = None):
    """
    バッチ処理による複数画像の損傷検出
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
        max_images: 処理する最大画像数（Noneの場合は全て）
    """
    print("\n[Mode: Batch Detection]")
    print(f"Input directory: {input_dir}")
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = RESULTS_DIR / "batch"
    else:
        output_dir = Path(output_dir)
    
    # モデルロード
    print("\n[Loading Model]")
    predictor = create_sam_predictor(
        checkpoint_path=str(MODEL_CONFIG["checkpoint_path"]),
        model_type=MODEL_CONFIG["model_type"],
        use_fp16=MODEL_CONFIG["use_fp16"],
        use_quantization=MODEL_CONFIG["use_quantization"]
    )
    
    # バッチ処理
    print("\n[Batch Processing]")
    image_processor = ImageProcessor(target_size=IMAGE_CONFIG["target_size"])
    batch_processor = BatchProcessor(predictor, image_processor)
    
    results = batch_processor.process_directory(
        input_dir=input_dir,
        pattern=IMAGE_CONFIG["image_pattern"]
    )
    
    # 最大画像数の制限
    if max_images is not None and len(results) > max_images:
        print(f"\nLimiting to first {max_images} images")
        results = results[:max_images]
    
    # 結果保存
    print("\n[Saving Results]")
    visualizer = Visualizer(alpha=VISUALIZATION_CONFIG["alpha"])
    saver = ResultSaver(output_dir)
    summary_path = saver.save_batch_results(results, visualizer)
    
    print("\n✓ Batch detection completed!")
    print(f"\nProcessed: {len(results)} images")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="SAM3 Damage Detector MVP - INT8量子化 + FP16推論"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="実行モード: single (単一画像) または batch (バッチ処理)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="単一画像モード時の画像パス"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        help="バッチモード時の入力ディレクトリパス"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="出力ディレクトリパス（デフォルト: results/[mode]）"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="バッチモード時の最大処理画像数"
    )
    
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="INT8量子化を無効化"
    )
    
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="FP16推論を無効化"
    )
    
    args = parser.parse_args()
    
    # 環境チェック
    check_environment()
    
    # モデルファイルチェック
    if not check_model_file():
        return
    
    # 設定の上書き
    if args.no_quantization:
        MODEL_CONFIG["use_quantization"] = False
        print("⚠️  INT8 quantization disabled")
    elif MODEL_CONFIG["use_quantization"] and torch.cuda.is_available():
        print("⚠️  Warning: INT8 quantization is enabled but not supported on CUDA.")
        print("    自動的にスキップされます。")
    
    if args.no_fp16:
        MODEL_CONFIG["use_fp16"] = False
        print("⚠️  FP16 inference disabled")
    elif MODEL_CONFIG["use_fp16"] and torch.cuda.is_available():
        print("⚠️  Warning: FP16 inference may cause dtype mismatch errors on GPU.")
        print("    問題が発生した場合は --no_fp16 を使用してください。")
    
    try:
        if args.mode == "single":
            if args.image is None:
                print("Error: --image is required for single mode")
                parser.print_help()
                return
            
            single_image_detection(args.image, args.output_dir)
        
        elif args.mode == "batch":
            if args.input_dir is None:
                # デフォルトのテストディレクトリを使用
                args.input_dir = TEST_IMAGES_DIR
                print(f"Using default test directory: {args.input_dir}")
            
            batch_detection(args.input_dir, args.output_dir, args.max_images)
    
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
