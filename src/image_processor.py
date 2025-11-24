"""
Image Processing Pipeline for SAM3 Damage Detector
損傷画像の読み込み、前処理、マスク生成を実装
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings


class ImageProcessor:
    """画像の読み込みと前処理を行うクラス"""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            target_size: リサイズ後のサイズ (width, height)、Noneの場合リサイズしない
        """
        self.target_size = target_size
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        画像を読み込む
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            RGB形式のnumpy配列
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理（リサイズなど）
        
        Args:
            image: 入力画像 (RGB)
        
        Returns:
            前処理済み画像
        """
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def load_and_preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        画像の読み込みと前処理を一括で行う
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            前処理済み画像
        """
        image = self.load_image(image_path)
        image = self.preprocess(image)
        return image


class DamageDetector:
    """損傷検出クラス"""
    
    def __init__(self, predictor):
        """
        Args:
            predictor: SamPredictor インスタンス
        """
        self.predictor = predictor
        self.current_image = None
    
    def detect_rust_color(self, image: np.ndarray) -> np.ndarray:
        """
        腐食（錆び、焼け茶色、黒焦げた茶色）の色域を検出
        
        Args:
            image: RGB画像
        
        Returns:
            腐食領域のバイナリマスク
        """
        # RGBかHSVに変換
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 腐食（錆び）の色範囲を定義
        # 実際のrust領域から抽出したHSV値に基づく（5th-95thパーセンタイル）
        # H: [0, 177] - 赤茶色から暗色まで広範囲
        # S: [31, 135] - 中～高彩度（低彩度のコンクリートを除外）
        # V: [28, 142] - 暗～中明度（明るすぎる領域を除外）
        
        # 実測値ベースの単一範囲
        lower_rust = np.array([0, 31, 28])
        upper_rust = np.array([177, 135, 142])
        rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
        
        # コンクリートの灰色領域を除外（彩度が低く、色相が広範囲）
        # コンクリート：H=任意, S=0-40, V=60-200（低彩度の灰色）
        concrete_mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 40, 200]))
        rust_mask = cv2.bitwise_and(rust_mask, cv2.bitwise_not(concrete_mask))
        
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        
        return rust_mask > 0
    
    def set_image(self, image: np.ndarray):
        """
        推論対象の画像を設定
        
        Args:
            image: RGB形式の画像
        """
        self.current_image = image
        self.predictor.set_image(image)
    
    def predict_with_points(
        self,
        point_coords: List[List[int]],
        point_labels: List[int],
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        座標指定による損傷領域の予測
        
        Args:
            point_coords: 座標のリスト [[x1, y1], [x2, y2], ...]
            point_labels: ラベルのリスト [1, 1, 0, ...] (1=正例, 0=負例)
            multimask_output: 複数のマスクを出力するか
        
        Returns:
            masks: 予測されたマスク (N, H, W)
            scores: 各マスクのスコア
            logits: 各マスクのロジット
        """
        if self.current_image is None:
            raise RuntimeError("画像が設定されていません。set_image()を先に実行してください。")
        
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
        
        return masks, scores, logits
    
    def predict_with_box(
        self,
        box: List[int],
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バウンディングボックス指定による損傷領域の予測
        
        Args:
            box: バウンディングボックス [x_min, y_min, x_max, y_max]
            multimask_output: 複数のマスクを出力するか
        
        Returns:
            masks: 予測されたマスク
            scores: 各マスクのスコア
            logits: 各マスクのロジット
        """
        if self.current_image is None:
            raise RuntimeError("画像が設定されていません。set_image()を先に実行してください。")
        
        box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output
        )
        
        return masks, scores, logits
    
    def auto_detect(self, num_points: int = 5, return_all: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        自動的に損傷領域を検出（簡易版）
        
        Args:
            num_points: サンプリングする点の数
            return_all: Trueの場合、全マスクとスコアを返す
        
        Returns:
            return_all=False: (best_mask, best_score)
            return_all=True: (all_masks, all_scores)
        """
        if self.current_image is None:
            raise RuntimeError("画像が設定されていません。")
        
        height, width = self.current_image.shape[:2]
        
        # グリッド状に点をサンプリング
        grid_size = int(np.sqrt(num_points))
        x_coords = np.linspace(width // 4, 3 * width // 4, grid_size, dtype=int)
        y_coords = np.linspace(height // 4, 3 * height // 4, grid_size, dtype=int)
        
        point_coords = [[x, y] for x in x_coords for y in y_coords]
        point_labels = [1] * len(point_coords)
        
        masks, scores, _ = self.predict_with_points(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        if return_all:
            return masks, scores
        
        # 最高スコアのマスクを選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask, best_score
    
    def detect_corrosion_areas(self, num_points: int = 1024, use_color: bool = True) -> List[dict]:
        """
        腐食領域を複数検出（色情報を使用）
        
        Args:
            num_points: サンプリング点数（デフォルト: 256、細かい腐食領域検出のため）
            use_color: 色情報を使用して腐食領域を特定（デフォルト: True）
        
        Returns:
            検出結果のリスト [{'mask': mask, 'score': score, 'area': area, 'type': type}, ...]
        """
        if self.current_image is None:
            raise RuntimeError("画像が設定されていません。")
        
        # 色ベースで腐食領域を検出
        rust_color_mask = None
        if use_color:
            rust_color_mask = self.detect_rust_color(self.current_image)
            rust_area = np.sum(rust_color_mask)
            print(f"Color-based rust detection: {rust_area} pixels ({rust_area / (rust_color_mask.shape[0] * rust_color_mask.shape[1]) * 100:.1f}%)")
        
        # 複数マスクを取得
        masks, scores = self.auto_detect(num_points=num_points, return_all=True)
        
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            area = np.sum(mask)
            
            # マスクのサイズで種類を判定
            total_pixels = mask.shape[0] * mask.shape[1]
            area_ratio = area / total_pixels
            
            # 色情報で腐食かどうかを判定
            rust_overlap = 0
            is_rust = False
            if use_color and rust_color_mask is not None:
                rust_overlap = np.sum(np.logical_and(mask, rust_color_mask))
                rust_overlap_ratio = rust_overlap / area if area > 0 else 0
                # マスクの10%以上が腐食色なら腐食領域と判定（闾値を下げて感度向上）
                is_rust = rust_overlap_ratio > 0.1
            
            # 種類を判定
            if area_ratio > 0.6:
                mask_type = "concrete"  # コンクリート全体
            elif is_rust:
                mask_type = "rust_corrosion"  # 腐食領域（色ベース）
            elif area_ratio > 0.2:
                mask_type = "damage"  # 中規模損傷
            elif area_ratio > 0.05:
                mask_type = "corrosion"  # 腐食領域（サイズベース）
            else:
                mask_type = "small_defect"  # 小さな欠陥
            
            results.append({
                'mask': mask,
                'score': float(score),
                'area': int(area),
                'area_ratio': float(area_ratio),
                'type': mask_type,
                'index': i,
                'rust_overlap': int(rust_overlap) if use_color else 0,
                'is_rust_based': is_rust if use_color else False
            })
        
        # 腐飽領域を優先してソート
        results.sort(key=lambda x: (x['is_rust_based'], x['score']), reverse=True)
        
        return results
    
    def detect_rust_regions_with_prompts(self) -> List[dict]:
        """
        色情報を使って腐食領域を検出し、密なプロンプトとしてSAMに渡す
        
        Returns:
            腐食領域のリスト
        """
        if self.current_image is None:
            raise RuntimeError("画像が設定されていません。")
        
        # 色ベースで腐飭領域を検出
        rust_mask = self.detect_rust_color(self.current_image)
        
        # 連結成分分析で個々の腐飕領域を特定
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            rust_mask.astype(np.uint8), connectivity=8
        )
        
        rust_regions = []
        for i in range(1, num_labels):  # 0は背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 70:  # 小さすぎる領域をフィルタリング
                continue
            
            # 各領域内に複数のプロンプトをグリッド状に密に配置
            region_mask = (labels == i)
            y_coords, x_coords = np.where(region_mask)
            
            # バウンディングボックスを取得
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # グリッド密度を領域サイズに応じて調整
            # 小さい領域: 5x5グリッド、中: 7x7、大: 9x9
            if area < 200:
                grid_size = 5
            elif area < 800:
                grid_size = 7
            else:
                grid_size = 9
            
            # グリッド状にサンプリングポイントを配置
            prompt_points = []
            step_x = max(1, width // grid_size)
            step_y = max(1, height // grid_size)
            
            for gy in range(grid_size):
                for gx in range(grid_size):
                    px = x_min + gx * step_x + step_x // 2
                    py = y_min + gy * step_y + step_y // 2
                    
                    # グリッド点が実際に領域内にあるかチェック
                    if 0 <= py < region_mask.shape[0] and 0 <= px < region_mask.shape[1]:
                        if region_mask[py, px]:
                            prompt_points.append([int(px), int(py)])
            
            # グリッドで十分なポイントが取れない場合は、領域内のランダムポイントを追加
            if len(prompt_points) < 3:
                indices = np.random.choice(len(y_coords), min(5, len(y_coords)), replace=False)
                prompt_points = [[int(x_coords[idx]), int(y_coords[idx])] for idx in indices]
            
            prompt_labels = [1] * len(prompt_points)
            
            # SAMで詳細なマスクを生成
            masks, scores, _ = self.predict_with_points(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                multimask_output=True
            )
            
            # 最高スコアのマスクを選択
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # マスクの面積を計算
            mask_area = int(np.sum(best_mask))
            
            # 異常に大きい領域を除外（鉄筋は2000px未満）
            if mask_area >= 2000:
                continue
            
            # 鉄筋の細長い形状特性をチェック（アスペクト比と充填率）
            y_mask, x_mask = np.where(best_mask)
            if len(y_mask) > 0:
                width = x_mask.max() - x_mask.min() + 1
                height = y_mask.max() - y_mask.min() + 1
                bbox_area = width * height
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                fill_ratio = mask_area / max(bbox_area, 1)  # マスクがバウンディングボックスをどれだけ埋めているか
                
                # 鉄筋の形状条件：
                # 1. アスペクト比が2.0以上（細長い）
                # 2. または、小さい領域（300px未満）は許容
                # 3. 充填率が0.7以上の場合は広いコンクリート領域の可能性があるので厳格に判定
                
                if mask_area >= 300:
                    # 充填率が高い（≥0.7）場合は、より厳格なアスペクト比（≥2.5）を要求
                    if fill_ratio >= 0.7 and aspect_ratio < 2.5:
                        continue
                    # 充填率が中程度の場合は、アスペクト比2.0以上を要求
                    elif aspect_ratio < 2.0:
                        continue
            
            # 中心座標を計算
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            rust_regions.append({
                'mask': best_mask,
                'score': float(best_score),
                'area': mask_area,
                'centroid': (cx, cy),
                'type': 'rust_corrosion',
                'color_area': int(area),
                'num_prompts': len(prompt_points),
                'aspect_ratio': aspect_ratio if len(y_mask) > 0 else 0
            })
        
        return rust_regions
    
    def find_rebar_pattern(self, regions: List[dict]) -> dict:
        """
        検出された鉄筋領域から配置パターン（直線、等間隔）を分析
        
        Args:
            regions: 検出された鉄筋領域のリスト
        
        Returns:
            パターン情報: {'lines': [...], 'spacing': float, 'angle': float}
        """
        if len(regions) < 2:
            return {'lines': [], 'spacing': 0, 'angle': 0}
        
        centroids = np.array([r['centroid'] for r in regions])
        
        # RANSACで直線を検出（複数の平行線を想定）
        from sklearn.cluster import DBSCAN
        
        # 1. Y座標でクラスタリング（水平方向の鉄筋群を検出）
        y_coords = centroids[:, 1].reshape(-1, 1)
        clustering = DBSCAN(eps=50, min_samples=2).fit(y_coords)
        
        lines = []
        for label in set(clustering.labels_):
            if label == -1:  # ノイズを除外
                continue
            
            cluster_points = centroids[clustering.labels_ == label]
            if len(cluster_points) >= 2:
                # 直線を最小二乗法でフィッティング
                x_vals = cluster_points[:, 0]
                y_vals = cluster_points[:, 1]
                
                # y = ax + b の形式
                A = np.vstack([x_vals, np.ones(len(x_vals))]).T
                a, b = np.linalg.lstsq(A, y_vals, rcond=None)[0]
                
                lines.append({
                    'slope': a,
                    'intercept': b,
                    'points': cluster_points.tolist(),
                    'y_mean': float(np.mean(y_vals))
                })
        
        # 2. 平行線の間隔を計算
        if len(lines) >= 2:
            lines = sorted(lines, key=lambda x: x['y_mean'])
            spacings = [lines[i+1]['y_mean'] - lines[i]['y_mean'] for i in range(len(lines)-1)]
            avg_spacing = np.mean(spacings) if spacings else 0
        else:
            avg_spacing = 0
        
        # 3. 角度を計算（平均勾配から）
        if lines:
            avg_slope = np.mean([line['slope'] for line in lines])
            angle = np.degrees(np.arctan(avg_slope))
        else:
            angle = 0
        
        return {
            'lines': lines,
            'spacing': float(avg_spacing),
            'angle': float(angle),
            'num_lines': len(lines)
        }
    
    def generate_pattern_based_prompts(self, pattern: dict, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        鉄筋パターンに基づいて追加の検出プロンプトを生成
        
        Args:
            pattern: find_rebar_pattern()の出力
            image_shape: (height, width)
        
        Returns:
            追加プロンプト座標のリスト [(x, y), ...]
        """
        height, width = image_shape
        additional_prompts = []
        
        if pattern['num_lines'] < 1 or pattern['spacing'] < 10:
            return additional_prompts
        
        lines = pattern['lines']
        spacing = pattern['spacing']
        
        # 既存の直線の間と外側に新しいプロンプトを配置
        for i, line in enumerate(lines):
            points = np.array(line['points'])
            x_min, x_max = int(points[:, 0].min()), int(points[:, 0].max())
            
            # 各直線上に等間隔でプロンプトを配置
            num_samples = max(3, int((x_max - x_min) / 80))  # 80px間隔
            for j in range(num_samples):
                x = x_min + (x_max - x_min) * j / (num_samples - 1) if num_samples > 1 else (x_min + x_max) / 2
                y = line['slope'] * x + line['intercept']
                
                if 0 <= x < width and 0 <= y < height:
                    additional_prompts.append((int(x), int(y)))
            
            # 次の直線との間にプロンプトを配置（中間線を予測）
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                mid_y = (line['y_mean'] + next_line['y_mean']) / 2
                
                # 中間線上にプロンプトを配置
                for j in range(num_samples):
                    x = x_min + (x_max - x_min) * j / (num_samples - 1) if num_samples > 1 else (x_min + x_max) / 2
                    # 傾きは既存の直線の平均を使用
                    avg_slope = (line['slope'] + next_line['slope']) / 2
                    y = mid_y
                    
                    if 0 <= x < width and 0 <= y < height:
                        additional_prompts.append((int(x), int(y)))
        
        # 最初の直線の上と最後の直線の下にも配置
        if len(lines) >= 2:
            first_line = lines[0]
            last_line = lines[-1]
            points_first = np.array(first_line['points'])
            x_min, x_max = int(points_first[:, 0].min()), int(points_first[:, 0].max())
            num_samples = max(3, int((x_max - x_min) / 80))
            
            # 上側
            y_above = first_line['y_mean'] - spacing
            if 0 <= y_above < height:
                for j in range(num_samples):
                    x = x_min + (x_max - x_min) * j / (num_samples - 1) if num_samples > 1 else (x_min + x_max) / 2
                    if 0 <= x < width:
                        additional_prompts.append((int(x), int(y_above)))
            
            # 下側
            y_below = last_line['y_mean'] + spacing
            if 0 <= y_below < height:
                for j in range(num_samples):
                    x = x_min + (x_max - x_min) * j / (num_samples - 1) if num_samples > 1 else (x_min + x_max) / 2
                    if 0 <= x < width:
                        additional_prompts.append((int(x), int(y_below)))
        
        return additional_prompts


class BatchProcessor:
    """複数画像の一括処理クラス"""
    
    def __init__(self, predictor, image_processor: Optional[ImageProcessor] = None):
        """
        Args:
            predictor: SamPredictor インスタンス
            image_processor: ImageProcessor インスタンス
        """
        self.detector = DamageDetector(predictor)
        self.image_processor = image_processor or ImageProcessor()
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        point_coords: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None,
        pattern: str = "*.png"
    ) -> List[dict]:
        """
        ディレクトリ内の画像を一括処理
        
        Args:
            input_dir: 入力ディレクトリ
            point_coords: 座標のリスト（全画像共通）
            point_labels: ラベルのリスト（全画像共通）
            pattern: 画像ファイルのパターン
        
        Returns:
            結果のリスト（各画像の結果を含む辞書）
        """
        input_dir = Path(input_dir)
        image_files = sorted(input_dir.glob(pattern))
        
        if len(image_files) == 0:
            warnings.warn(f"画像が見つかりません: {input_dir}/{pattern}")
            return []
        
        results = []
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            try:
                # 画像読み込み
                image = self.image_processor.load_and_preprocess(image_file)
                self.detector.set_image(image)
                
                # 損傷検出
                if point_coords is not None and point_labels is not None:
                    masks, scores, _ = self.detector.predict_with_points(
                        point_coords=point_coords,
                        point_labels=point_labels
                    )
                else:
                    # 自動検出
                    best_mask, best_score = self.detector.auto_detect()
                    masks = np.array([best_mask])
                    scores = np.array([best_score])
                
                # 最良のマスクを選択
                best_idx = np.argmax(scores)
                
                results.append({
                    'filename': image_file.name,
                    'image': image,
                    'mask': masks[best_idx],
                    'score': float(scores[best_idx]),
                    'all_masks': masks,
                    'all_scores': scores
                })
                
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
                continue
        
        print(f"\nCompleted: {len(results)}/{len(image_files)} images processed successfully")
        
        return results


if __name__ == "__main__":
    print("Image Processing Pipeline Test")
    print("="*50)
    
    # テスト用コード
    test_image_path = "../data/1_Test_images-kensg/kensg-rebarexposureRb_001.png"
    
    processor = ImageProcessor()
    
    try:
        image = processor.load_and_preprocess(test_image_path)
        print(f"✓ Image loaded successfully!")
        print(f"  Shape: {image.shape}")
        print(f"  Dtype: {image.dtype}")
    except Exception as e:
        print(f"✗ Error: {e}")
