
import argparse
from pathlib import Path

import cv2

from cv2_process import LaneDetection


def parse_args():
    parser = argparse.ArgumentParser(description="车道检测")
    parser.add_argument("-i","--input", nargs="?", default="00240.jpg", help="检测文件的路径")
    parser.add_argument("-o", "--output", default="result.jpg", help="输出的路径")
    parser.add_argument("--classical", action="store_true", help="use the lower-accuracy classical fallback only")
    parser.add_argument("-s","--show", action="store_true", help="显示处理后的文件")
    return parser.parse_args()


def main():

    args = parse_args()

    image = cv2.imread(args.input)

    if image is None:
        raise SystemExit(f"未找到文件")


    detector = LaneDetection(args.model, use_yolo=not args.classical)
    result = detector(image, args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None
    if not cv2.imwrite(str(output), result):
        raise SystemExit(f"Unable to write output image: {output}")
    print(f"Result saved to: {output}")

    if args.show:
        cv2.imshow("Lane detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



