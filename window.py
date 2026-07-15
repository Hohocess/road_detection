

import sys
from pathlib import Path
from typing import Optional

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from cv2_process import LaneDetection


class LaneDetectionWindow(QtWidgets.QMainWindow):


    def __init__(self):
        super().__init__()
        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.video_path: Optional[Path] = None
        self.image_path: Optional[Path] = None
        self.image_result = None
        self.detector = LaneDetection()  # Constructed once; no per-frame setup.
        self.frame_index = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("车道线识别 - 视频处理")
        self.resize(1100, 760)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QHBoxLayout()

        self.open_image_button = QtWidgets.QPushButton("选择图片")
        self.open_image_button.clicked.connect(self.select_image)
        self.open_video_button = QtWidgets.QPushButton("选择视频")
        self.open_video_button.clicked.connect(self.select_video)
        self.start_button = QtWidgets.QPushButton("开始识别")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_or_pause)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        self.save_checkbox = QtWidgets.QCheckBox("保存识别视频")
        self.save_checkbox.setChecked(True)
        controls.addWidget(self.open_image_button)
        controls.addWidget(self.open_video_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.save_checkbox)
        controls.addStretch()
        layout.addLayout(controls)

        self.preview = QtWidgets.QLabel("请选择图片或视频文件")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(800, 550)
        self.preview.setStyleSheet("background: #1e1e1e; color: #dddddd; font-size: 18px;")
        layout.addWidget(self.preview, 1)

        self.status = QtWidgets.QLabel("就绪")
        layout.addWidget(self.status)
        self.setCentralWidget(central)

    def select_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )
        if not file_name:
            return
        self.release_resources()
        self.video_path = None
        self.image_path = Path(file_name)
        self.detector.reset()
        image = cv2.imread(str(self.image_path))
        if image is None:
            self.status.setText("无法读取该图片文件")
            return

        self.status.setText("正在识别图片…")
        QtWidgets.QApplication.processEvents()
        try:
            self.image_result = self.detector(image, self.image_path)
        except Exception as error:
            self.image_result = None
            self.status.setText(f"图片识别失败：{error}")
            return

        self.show_frame(self.image_result)
        output = self.image_path.with_name(f"{self.image_path.stem}_lane.jpg")
        if cv2.imwrite(str(output), self.image_result):
            self.status.setText(f"图片识别完成，已保存：{output.name}")
        else:
            self.status.setText("图片识别完成，但无法保存结果")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def select_video(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if not file_name:
            return
        self.release_resources()
        self.image_path = None
        self.image_result = None
        self.detector.reset()
        self.video_path = Path(file_name)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.capture = None
            self.status.setText("无法打开该视频文件")
            return

        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_index = 0
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.start_button.setText("开始识别")
        self.status.setText(f"已加载：{self.video_path.name}，{frame_count} 帧，{fps:.1f} FPS")
        self.show_first_frame()

    def show_first_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            self.show_frame(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def start_or_pause(self):
        if self.capture is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.start_button.setText("继续识别")
            self.status.setText("已暂停")
            return

        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.timer.start(max(1, int(1000 / fps)))
        self.start_button.setText("暂停")
        self.stop_button.setEnabled(True)
        self.status.setText("正在识别…")

    def process_next_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.finish_video()
            return

        result = self.detector(frame)
        self.frame_index += 1
        self.show_frame(result)
        if self.save_checkbox.isChecked():
            self.write_frame(result)
        self.status.setText(f"正在识别：第 {self.frame_index} 帧")

    def write_frame(self, frame):
        if self.writer is None:
            assert self.video_path is not None and self.capture is not None
            output = self.video_path.with_name(f"{self.video_path.stem}_lane.mp4")
            fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                self.writer = None
                self.save_checkbox.setChecked(False)
                self.status.setText("无法创建输出视频，将仅预览")
                return
        self.writer.write(frame)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QtGui.QImage(rgb.data, width, height, channels * width, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def stop_processing(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0
        self.release_writer()
        self.start_button.setText("开始识别")
        self.stop_button.setEnabled(False)
        self.status.setText("已停止，可再次开始")

    def finish_video(self):
        self.timer.stop()
        self.release_writer()
        self.start_button.setText("重新开始")
        self.stop_button.setEnabled(False)
        self.status.setText("识别完成")

    def release_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def release_resources(self):
        self.timer.stop()
        self.release_writer()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def closeEvent(self, event):
        self.release_resources()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LaneDetectionWindow()
    window.show()
    sys.exit(app.exec_())
