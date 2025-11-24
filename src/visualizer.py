"""
Visualization and Result Saving for SAM3 Damage Detector
検出結果の可視化と保存機能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union
import json
from datetime import datetime


class Visualizer:
    """損傷検出結果の可視化クラス"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: マスクの透明度 (0.0-1.0)
        """
        self.alpha = alpha
    
    def show_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        show: bool = True
    ) -> np.ndarray:
        """
        画像にマスクをオーバーレイ
        
        Args:
            image: 元画像 (RGB)
            mask: マスク (H, W) boolean配列
            color: マスクの色 (R, G, B)
            show: 結果を表示するか
        
        Returns:
            マスクオーバーレイ済み画像
        """
        # マスクを3チャンネルに変換
        mask_colored = np.zeros_like(image)
        mask_colored[mask] = color
        
        # オーバーレイ
        overlay = image.copy()
        overlay[mask] = cv2.addWeighted(
            image[mask],
            1 - self.alpha,
            mask_colored[mask],
            self.alpha,
            0
        )
        
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(overlay)
            plt.title("Damage Detection Result")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return overlay
    
    def show_multiple_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        scores: List[float],
        labels: List[str] = None,
        show: bool = True
    ) -> List[np.ndarray]:
        """
        複数のマスクを表示
        
        Args:
            image: 元画像
            masks: マスクのリスト
            scores: スコアのリスト
            labels: ラベルのリスト（オプション）
            show: 結果を表示するか
        
        Returns:
            オーバーレイ済み画像のリスト
        """
        colors = [
            (255, 0, 0),    # 赤（コンクリート全体）
            (255, 165, 0),  # オレンジ（腐食領域1）
            (255, 255, 0),  # 黄色（腐食領域2）
            (0, 255, 0),    # 緑
            (0, 0, 255),    # 青
        ]
        
        results = []
        
        if show:
            n_masks = len(masks)
            fig, axes = plt.subplots(1, n_masks + 1, figsize=(4 * (n_masks + 1), 4))
            
            if n_masks == 0:
                axes = [axes]
            
            # 元画像
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # 各マスク
            for i, (mask, score) in enumerate(zip(masks, scores)):
                color = colors[i % len(colors)]
                overlay = self.show_mask(image, mask, color, show=False)
                results.append(overlay)
                
                label = labels[i] if labels and i < len(labels) else f"Mask {i+1}"
                axes[i + 1].imshow(overlay)
                axes[i + 1].set_title(f"{label}\nScore: {score:.3f}")
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                overlay = self.show_mask(image, mask, color, show=False)
                results.append(overlay)
        
        return results
    
    def show_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        show: bool = True
    ) -> np.ndarray:
        """
        画像に指定点を表示
        
        Args:
            image: 元画像
            point_coords: 座標配列 [[x, y], ...]
            point_labels: ラベル配列 [1, 0, ...]
            show: 結果を表示するか
        
        Returns:
            点を描画した画像
        """
        result = image.copy()
        
        for (x, y), label in zip(point_coords, point_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # 正例=緑、負例=赤
            cv2.circle(result, (int(x), int(y)), 10, color, -1)
            cv2.circle(result, (int(x), int(y)), 12, (255, 255, 255), 2)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(result)
            plt.title("Input Points")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return result


class ResultSaver:
    """検出結果の保存クラス"""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_image(
        self,
        image: np.ndarray,
        filename: str,
        subdir: Optional[str] = None
    ) -> Path:
        """
        画像を保存
        
        Args:
            image: 画像 (RGB)
            filename: ファイル名
            subdir: サブディレクトリ名
        
        Returns:
            保存先のパス
        """
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        save_path = save_dir / filename
        
        # RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), image_bgr)
        
        return save_path
    
    def save_mask(
        self,
        mask: np.ndarray,
        filename: str,
        subdir: Optional[str] = None
    ) -> Path:
        """
        マスクを保存
        
        Args:
            mask: マスク (boolean配列)
            filename: ファイル名
            subdir: サブディレクトリ名
        
        Returns:
            保存先のパス
        """
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        save_path = save_dir / filename
        
        # マスクを画像として保存
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), mask_img)
        
        return save_path
    
    def save_result(
        self,
        result: dict,
        visualizer: Optional[Visualizer] = None
    ) -> dict:
        """
        検出結果を保存
        
        Args:
            result: 検出結果の辞書
            visualizer: Visualizerインスタンス
        
        Returns:
            保存先パスを含む辞書
        """
        filename = result['filename']
        base_name = Path(filename).stem
        
        saved_paths = {}
        
        # 元画像を保存
        original_path = self.save_image(
            result['image'],
            f"{base_name}_original.png",
            subdir="original"
        )
        saved_paths['original'] = str(original_path)
        
        # マスクを保存
        mask_path = self.save_mask(
            result['mask'],
            f"{base_name}_mask.png",
            subdir="masks"
        )
        saved_paths['mask'] = str(mask_path)
        
        # 可視化画像を保存
        if visualizer:
            overlay = visualizer.show_mask(
                result['image'],
                result['mask'],
                show=False
            )
            overlay_path = self.save_image(
                overlay,
                f"{base_name}_overlay.png",
                subdir="overlay"
            )
            saved_paths['overlay'] = str(overlay_path)
        
        return saved_paths
    
    def save_batch_results(
        self,
        results: List[dict],
        visualizer: Optional[Visualizer] = None,
        save_summary: bool = True
    ) -> Path:
        """
        バッチ処理結果を保存
        
        Args:
            results: 検出結果のリスト
            visualizer: Visualizerインスタンス
            save_summary: サマリーJSONを保存するか
        
        Returns:
            サマリーファイルのパス
        """
        print(f"\nSaving {len(results)} results to {self.output_dir}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'results': []
        }
        
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] Saving: {result['filename']}")
            
            saved_paths = self.save_result(result, visualizer)
            
            summary['results'].append({
                'filename': result['filename'],
                'score': result['score'],
                'paths': saved_paths
            })
        
        # サマリーを保存
        if save_summary:
            summary_path = self.output_dir / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Summary saved to: {summary_path}")
            return summary_path
        
        return self.output_dir


def create_comparison_figure(
    original: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    score: float,
    save_path: Optional[Path] = None
) -> None:
    """
    比較用の図を作成
    
    Args:
        original: 元画像
        overlay: オーバーレイ画像
        mask: マスク
        score: スコア
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Detection Result\nScore: {score:.3f}")
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization and Result Saving Test")
    print("="*50)
    
    # テスト用ダミーデータ
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_mask = np.random.rand(480, 640) > 0.7
    
    visualizer = Visualizer(alpha=0.5)
    
    print("Creating visualization...")
    overlay = visualizer.show_mask(dummy_image, dummy_mask, show=False)
    print(f"✓ Visualization created! Shape: {overlay.shape}")
    
    # 保存テスト
    print("\nTesting result saver...")
    saver = ResultSaver("../results/test")
    test_result = {
        'filename': 'test_001.png',
        'image': dummy_image,
        'mask': dummy_mask,
        'score': 0.95
    }
    
    saved_paths = saver.save_result(test_result, visualizer)
    print(f"✓ Results saved:")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")
