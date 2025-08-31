import argparse
from ultralytics import YOLO
import cv2

def run(weights, source, conf=0.25, save=True, show=True):
    # Load model
    model = YOLO(weights)

    # Run inference
    results = model.predict(
        source=source,  # image, video, directory, or webcam (0)
        conf=conf,      # confidence threshold
        save=save,      # save predictions
        show=show       # display in window
    )

    # Loop through results
    for r in results:
        im_array = r.plot()  # plot predictions
        cv2.imshow("Mussel Detection", im_array)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="model path")
    parser.add_argument("--source", type=str, default="test.jpg", help="file/dir/URL/glob/webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nosave", action="store_true", help="do not save results")
    parser.add_argument("--noshow", action="store_true", help="do not display results")

    args = parser.parse_args()

    run(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        save=not args.nosave,
        show=not args.noshow
    )

