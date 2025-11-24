"""
SAM3 Model Loader with INT8 Quantization and FP16 Inference
GPU 16GB対応のための最適化実装
"""

import torch
import torch.quantization
from segment_anything import sam_model_registry, SamPredictor
import warnings
from pathlib import Path

class SAM3ModelLoader:
    """SAM3モデルのロード、量子化、推論を管理するクラス"""
    
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cuda"):
        """
        Args:
            model_type (str): SAMモデルのタイプ (vit_h, vit_l, vit_b)
            checkpoint_path (str): モデルチェックポイントのパス
            device (str): 使用デバイス (cuda or cpu)
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.predictor = None
        
        if self.device == "cpu":
            warnings.warn("CUDAが利用できません。CPUモードで実行します。")
    
    def load_model(self, use_fp16=True, use_quantization=True):
        """
        SAMモデルをロードし、FP16変換とINT8量子化を適用
        
        Args:
            use_fp16 (bool): FP16推論を使用するか
            use_quantization (bool): INT8量子化を使用するか
        
        Returns:
            SamPredictor: 設定済みのSAM Predictor
        """
        print(f"Loading SAM model: {self.model_type}")
        print(f"Device: {self.device}")
        
        # チェックポイントパスの検証
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_pathが指定されていません")
        
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")
        
        # SAMモデルのロード
        self.sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        self.sam.eval()
        
        # デバイスへ移動
        self.sam = self.sam.to(self.device)
        
        # FP16変換の警告（GPU使用時）
        if use_fp16:
            if self.device == "cuda":
                warnings.warn(
                    "FP16推論は型の不一致エラーを引き起こす可能性があります。\n"
                    "GPU使用時はFP32推奨。問題が発生する場合は --no_fp16 オプションを使用してください。"
                )
                print("Applying FP16 conversion (experimental)...")
                self.sam = self.sam.half()
            else:
                warnings.warn("FP16はGPU使用時のみ有効です。CPUではFP32を使用します。")
        
        # INT8量子化の警告（CUDA非対応）
        if use_quantization:
            if self.device == "cuda":
                warnings.warn(
                    "INT8量子化はCUDAでサポートされていません。\n"
                    "GPU使用時は量子化なしで実行します。CPUで量子化を使用する場合は --no_fp16 --no_quantization なしで実行してください。"
                )
                print("Skipping INT8 quantization (CUDA not supported)")
            else:
                print("Applying INT8 quantization (CPU only)...")
                self.sam = self._quantize_model(self.sam)
        
        # Predictorの初期化
        self.predictor = SamPredictor(self.sam)
        
        print("Model loaded successfully!")
        self._print_model_info()
        
        return self.predictor
    
    def _quantize_model(self, model):
        """
        モデルにINT8量子化を適用（CPU専用）
        
        Args:
            model: 量子化対象のモデル
        
        Returns:
            量子化されたモデル
        
        Note:
            INT8動的量子化はCPUでのみサポートされています。
            CUDAデバイスでは `quantized::linear_dynamic` エラーが発生します。
        """
        # CUDAデバイスチェック
        if next(model.parameters()).is_cuda:
            warnings.warn(
                "INT8量子化はCUDAデバイスでサポートされていません。量子化をスキップします。\n"
                "詳細: quantized::linear_dynamic はCPUバックエンドでのみ利用可能です。"
            )
            return model
        
        try:
            # 動的量子化を適用（線形層をINT8に変換）
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("INT8 quantization applied successfully (CPU only)")
            return quantized_model
        except Exception as e:
            warnings.warn(f"量子化に失敗しました: {e}. 元のモデルを使用します。")
            return model
    
    def _print_model_info(self):
        """モデル情報を出力"""
        if self.sam is not None:
            total_params = sum(p.numel() for p in self.sam.parameters())
            print(f"\nModel Information:")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Device: {self.device}")
            print(f"  Dtype: {next(self.sam.parameters()).dtype}")
    
    def get_predictor(self):
        """Predictorを取得"""
        if self.predictor is None:
            raise RuntimeError("モデルがロードされていません。load_model()を先に実行してください。")
        return self.predictor
    
    @staticmethod
    def get_recommended_checkpoint_path(model_type="vit_h"):
        """
        推奨されるチェックポイントパスを返す
        
        Args:
            model_type (str): モデルタイプ
        
        Returns:
            str: チェックポイントパス
        """
        checkpoint_urls = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        return checkpoint_urls.get(model_type, "sam_vit_h_4b8939.pth")
    
    def estimate_memory_usage(self):
        """推定メモリ使用量を計算"""
        if self.sam is None:
            return 0
        
        total_memory = 0
        for param in self.sam.parameters():
            param_memory = param.numel() * param.element_size()
            total_memory += param_memory
        
        # GB単位で返す
        return total_memory / (1024 ** 3)


def create_sam_predictor(checkpoint_path, model_type="vit_h", use_fp16=False, use_quantization=False):
    """
    SAM Predictorを簡単に作成するヘルパー関数
    
    Args:
        checkpoint_path (str): モデルチェックポイントのパス
        model_type (str): モデルタイプ
        use_fp16 (bool): FP16推論を使用（デフォルト: False、GPU使用時は型の不一致エラーに注意）
        use_quantization (bool): INT8量子化を使用（デフォルト: False、CPUのみサポート）
    
    Returns:
        SamPredictor: 設定済みのPredictor
    
    Note:
        推奨設定:
        - GPU使用時: use_fp16=False, use_quantization=False (FP32で安定動作)
        - CPU使用時: use_fp16=False, use_quantization=True (INT8で高速化)
    """
    loader = SAM3ModelLoader(model_type=model_type, checkpoint_path=checkpoint_path)
    predictor = loader.load_model(use_fp16=use_fp16, use_quantization=use_quantization)
    
    # メモリ使用量の推定
    memory_gb = loader.estimate_memory_usage()
    print(f"\nEstimated GPU Memory Usage: {memory_gb:.2f} GB")
    
    return predictor


if __name__ == "__main__":
    # テスト用コード
    print("SAM3 Model Loader Test")
    print("="*50)
    
    # モデルパスの例
    model_path = "../models/sam_vit_h_4b8939.pth"
    
    try:
        predictor = create_sam_predictor(
            checkpoint_path=model_path,
            model_type="vit_h",
            use_fp16=True,
            use_quantization=True
        )
        print("\n✓ Model loaded successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nモデルファイルをダウンロードしてください:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
