"""
SAM3 Damage Detector Configuration
設定ファイル
"""

from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
TEST_IMAGES_DIR = DATA_DIR / "1_Test_images-kensg"

# モデルディレクトリ
MODELS_DIR = PROJECT_ROOT / "models"

# 結果ディレクトリ
RESULTS_DIR = PROJECT_ROOT / "results"

# モデル設定
MODEL_CONFIG = {
    "model_type": "vit_h",  # vit_h, vit_l, vit_b
    "checkpoint_path": MODELS_DIR / "sam_vit_h_4b8939.pth",
    "use_fp16": False,  # GPU使用時はFP32推奨（型の不一致を避けるため）
    "use_quantization": False,  # INT8量子化はCPUのみサポート
}

# 画像処理設定
IMAGE_CONFIG = {
    "target_size": None,  # (width, height) or None for original size
    "image_pattern": "*.png",
}

# 検出設定
DETECTION_CONFIG = {
    "multimask_output": True,
    "num_auto_points": 9,  # 自動検出時のサンプリング点数
}

# 可視化設定
VISUALIZATION_CONFIG = {
    "alpha": 0.5,  # マスクの透明度
    "mask_color": (255, 0, 0),  # RGB
    "show_plots": False,  # プロットを表示するか（バッチ処理時はFalse推奨）
}

# SAMモデルのダウンロードURL
MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
