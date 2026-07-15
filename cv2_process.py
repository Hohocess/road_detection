
#用LOYO模型得到的方块图，进一步用opencv得到车道线


from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from yolo_process import YOLOprocess


class LaneDetection:
    #1.根据YOLO模型，得到置信度较高的方框
    #2.根据方框，对原图进行裁剪
    #3.利用灰度处理、高斯模糊、Canny边缘检测得到边线图
    #4.利用霍夫变换筛选直线
    #5.如果是视频，根据判断结果，预测后续车道线
    #6.通过线性回归拟合车道线
    #7.绘制并输出

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best.pt",    # YOLO分割模型路径
        confidence: float = 0.6,       # 模型置信度阈值（0-1）
        #Canny边缘检测参数
        canny_low: int = 50,            # Canny边缘检测的低阈值
        canny_high: int = 150,          # Canny边缘检测的高阈值
        #Hough变换参数
        hough_threshold: int = 25,      # Hough变换的投票阈值
        min_line_length: int = 30,      # Hough变换检测的最小直线长度
        max_line_gap: int = 40,         # Hough变换检测的最大直线间隔
        #车道预测参数
        temporal_alpha: float = 0.35,   # 时间平滑系数（0-1），值越大越依赖当前帧
        max_prediction_frames: int = 24, # 在没有检测到时最多预测的帧数
    ):
        #Canny边缘检测参数
        self.canny_low = canny_low
        self.canny_high = canny_high

        #Hough变换参数
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        #加载模型
        self.yolo = YOLOprocess(model_path, confidence)

        #车道预测参数
        self.temporal_alpha = temporal_alpha
        self.max_prediction_frames = max_prediction_frames

        # 存储左右车道的模型
        self._lane_models = {"left": None, "right": None}
        # 记录没有检测到车道的连续帧数
        self._missed_frames = {"left": 0, "right": 0}

    def reset(self) -> None:

        self._lane_models = {"left": None, "right": None}
        self._missed_frames = {"left": 0, "right": 0}

    def __call__(self, image: np.ndarray, image_path: Optional[Union[str, Path]] = None) -> np.ndarray:

        # 检查输入图像有效性
        if image is None or image.size == 0:
            raise ValueError("未找到或文件不合规")

        # 使用YOLO模型分割

        if self.yolo is not None:

            polygons = self.yolo.process_frame(image)
            models = self._update_lane_models(image.shape[:2], polygons)
            # 如果检测到任何车道，绘制结果
            if any(model is not None for model in models.values()):
                return self._draw_temporal_lanes(image, models)

        # 如果没有YOLO检测到结果，使用Hough变换作为备选
        # 获取感兴趣区域（ROI）
        rois = self._lane_rois(image, image_path)
        # 创建透明覆盖层用于绘制车道线
        overlay = np.zeros_like(image)
        # 对每个ROI进行处理
        for x1, y1, x2, y2 in rois:
            crop = image[y1:y2, x1:x2]
            # 使用Hough变换检测直线
            lines = self._hough_lines(crop)
            # 拟合直线并绘制到覆盖层
            self._draw_fitted_lines(overlay, lines, (x1, y1), crop.shape[:2])
        # 将结果与原图像融合
        return cv2.addWeighted(image, 1.0, overlay, 0.9, 0.0)

    def _update_lane_models(self, shape: Tuple[int, int], polygons: List[np.ndarray]):

        height, width = shape
        # 候选车道列表（左、右两侧）
        candidates = {"left": [], "right": []}
        
        # 处理每个分割多边形
        for polygon in polygons:
            # 拟合多边形中心线
            model = self._fit_polygon_centreline(polygon, height, width)
            if model is None:
                continue
            coeff, y_start, y_end = model
            # 计算曲线在图像底部的x坐标，用于判断左右侧
            bottom_x = np.polyval(coeff, y_end / height) * width
            # 根据x坐标判断是左车道还是右车道
            side = "left" if bottom_x < width / 2 else "right"
            candidates[side].append(model)

        # 更新左右两条车道的模型
        for side in ("left", "right"):
            previous = self._lane_models[side]
            if candidates[side]:
                # 如果有新的检测结果
                # 优先选择与前一帧最接近的检测，否则选择垂直跨度最大的
                if previous is not None:
                    previous_coeff = previous[0]
                    # 选择与前一帧轨迹最接近的候选
                    current = min(
                        candidates[side],
                        key=lambda item: abs(np.polyval(item[0], 0.8) - np.polyval(previous_coeff, 0.8)),
                    )
                else:
                    # 没有前一帧时，选择垂直跨度最大的
                    current = max(candidates[side], key=lambda item: item[2] - item[1])
                coeff, y_start, y_end = current
                
                # 与前一帧进行时间平滑
                if previous is not None:
                    old_coeff, old_start, old_end = previous
                    # 检查帧间跳跃是否合理（检测物体边缘可能导致不合理跳跃）
                    jump = abs(np.polyval(coeff, 0.8) - np.polyval(old_coeff, 0.8)) * width
                    # 如果跳跃在合理范围内（图像宽度的18%），进行平滑
                    if jump <= width * 0.18:
                        # 加权平均当前帧和前一帧的信息
                        coeff = self.temporal_alpha * coeff + (1.0 - self.temporal_alpha) * old_coeff
                        y_start = int(self.temporal_alpha * y_start + (1.0 - self.temporal_alpha) * old_start)
                        y_end = int(self.temporal_alpha * y_end + (1.0 - self.temporal_alpha) * old_end)
                    else:
                        # 跳跃太大，保持前一帧的结果
                        coeff, y_start, y_end = old_coeff, old_start, old_end
                # 保存更新后的模型
                self._lane_models[side] = (coeff, y_start, y_end)
                # 重置未检测帧计数
                self._missed_frames[side] = 0
            elif previous is not None:
                # 当前帧未检测到，但之前有检测结果
                self._missed_frames[side] += 1
                # 超过最大预测帧数后，清除模型
                if self._missed_frames[side] > self.max_prediction_frames:
                    self._lane_models[side] = None
        return self._lane_models

    @staticmethod
    def _fit_polygon_centreline(polygon: np.ndarray, height: int, width: int):

        # 创建二值掩膜
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        
        # 找到掩膜非零行
        y_values = np.flatnonzero(mask.any(axis=1))
        # 如果有效行数太少，无法进行拟合
        if len(y_values) < 12:
            return None
        
        # 采样点：每隔3行取一个样本点
        samples_y, samples_x = [], []
        for y in y_values[::3]:
            # 找到该行的所有x坐标
            x_values = np.flatnonzero(mask[y])
            if len(x_values):
                # 归一化坐标到[0, 1]范围
                samples_y.append(y / height)
                # 取该行的中位数作为车道中心
                samples_x.append(np.median(x_values) / width)
        
        # 需要足够的样本进行拟合
        if len(samples_y) < 8:
            return None
        
        # 拟合二次曲线 x = a*y^2 + b*y + c
        coeff = np.polyfit(samples_y, samples_x, 2)
        # 验证拟合结果是否合理（x坐标不能超出图像范围）
        probe = np.polyval(coeff, np.asarray(samples_y))
        if np.any(probe < -0.1) or np.any(probe > 1.1):
            return None
        
        return coeff, int(y_values[0]), int(y_values[-1])

    @staticmethod
    def _draw_temporal_lanes(image: np.ndarray, models) -> np.ndarray:

        height, width = image.shape[:2]
        overlay = np.zeros_like(image)
        curves = {}  # 存储左右曲线的像素坐标
        
        # 处理左右两条车道
        for side, model in models.items():
            if model is None:
                continue
            
            coeff, y_start, y_end = model
            # 限制绘制范围：从图像中点往下42%开始，到底部结束
            start = max(int(height * 0.42), y_start)
            end = min(height - 1, y_end)
            # 有效的垂直范围需要至少20像素
            if end - start < 20:
                continue
            
            # 生成曲线上的点
            y_values = np.linspace(start, end, 70)
            # 根据多项式计算对应的x坐标
            x_values = np.polyval(coeff, y_values / height) * width
            # 组合成坐标点，并确保在图像范围内
            points = np.column_stack((x_values, y_values)).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            curves[side] = points
            
            # 根据曲率选择绘制方式
            # 低曲率（|a| < 0.035）用直线表示，高曲率保留曲线
            if abs(coeff[0]) < 0.035:
                # 绘制直线
                cv2.line(overlay, tuple(points[0]), tuple(points[-1]), (0, 255, 0), 8, cv2.LINE_AA)
            else:
                # 绘制曲线
                cv2.polylines(overlay, [points], False, (0, 255, 0), 8, cv2.LINE_AA)

        # 如果检测到左右两条车道，填充走廊区域
        if "left" in curves and "right" in curves:
            # 合并左右曲线形成走廊的边界（右侧曲线需要反向）
            corridor = np.vstack((curves["left"], curves["right"][::-1]))
            # 用深绿色填充走廊
            cv2.fillPoly(overlay, [corridor], (0, 90, 0))
            # 重新绘制边界线，确保不被走廊颜色覆盖
            for points in curves.values():
                cv2.polylines(overlay, [points], False, (0, 255, 0), 8, cv2.LINE_AA)
        
        # 将覆盖层与原图像融合
        return cv2.addWeighted(image, 1.0, overlay, 0.55, 0.0)

    def _lane_rois(self, image, image_path):

        height, width = image.shape[:2]
        boxes = []
        
        # 如果有YOLO模型且图像路径有效，获取YOLO检测结果
        if self.yolo is not None and image_path is not None and Path(image_path).is_file():
            boxes = self.yolo.process_image(image_path)

        # 处理YOLO边界框
        expanded = []
        for x1, y1, x2, y2 in boxes:
            # 水平方向缩小：框宽度的15%或最小12像素
            pad_x = max(12, (x2 - x1) * 0.15)
            # 垂直方向扩展：框高度的8%或最小12像素
            pad_y = max(12, int((y2 - y1) * 0.08))
            # 确保扩展后的框在图像范围内
            expanded.append((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))
        
        # 如果有扩展后的框，返回
        if expanded:
            return expanded

        # 默认道路区域：图像的下半部分（从45%高度到底部）
        # 这个区域覆盖宽度分辨率独立
        return [(0, int(height * 0.45), width, height)]

    def _hough_lines(self, crop):

        # 转换为灰度图像
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # Hough概率变换
        return cv2.HoughLinesP(
            edges, 1, np.pi / 180, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap,
        )

    def _draw_fitted_lines(self, canvas, lines, crop_shape):

        if lines is None:
            return
        
        # 分别存储左侧和右侧的直线端点
        left, right = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            # 垂直直线
            if dx == 0:
                continue


            k = dy / dx
            
            # 有效的车道线斜率范围在0.35-4.0之间
            if 0.35 <= k <= 4.0:
                left.extend(((x1, y1), (x2, y2)))
            elif -4.0 <= k <= -0.35:
                right.extend(((x1, y1), (x2, y2)))

        # 处理左右两侧的直线
        for points in (left, right):

            # 需要至少4个点来进行有效的拟合
            if len(points) < 6:
                continue
            
            # 转换为numpy数组
            pts = np.asarray(points, dtype=np.float32)
            # 使用 x = a*y + b 的线性模型
            a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
            
            # 计算直线在顶部和底部的x坐标
            top = int(crop_shape[0] * 0.12)         #从裁剪区域顶部12%处
            bottom = crop_shape[0] - 1             # 裁剪区域底部
            x_top, x_bottom = int(a * top + b), int(a * bottom + b)
            
            cv2.line(canvas, (x_top , top ), (x_bottom , bottom ), (0, 255, 0), 7, cv2.LINE_AA)
