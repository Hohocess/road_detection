from ultralytics import YOLO


def train():

    model = YOLO(
        "runs\segment\CULane_YOLOv8_seg-2\weights\last.pt"
    )


    model.train(
        resume=True
    )

if __name__ == "__main__":
    train()