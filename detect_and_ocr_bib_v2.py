from cProfile import run
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
import cv2
from skimage import io
import numpy as np
from pathos.threading import ThreadPool

# import mediapipe as mp
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# mp_face_detection = mp.solutions.face_detection
# from easyocr import Reader
import pandas as pd
from functools import wraps
import time
import multiprocessing
from utils import forward_passer, box_extractor
from imutils.object_detection import non_max_suppression

detector = "frozen_east_text_detection.pb"
NETWORK = cv2.dnn.readNet(detector)

def build_tesseract_options(psm):
    alphanumeric = "0123456789"
    options = f"-c tessedit_char_whitelist={alphanumeric}"
    options += "-c tessedit_char_blacklist='., '"
    options += f" --psm {psm}"
    return options

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def show2(
    title, image, resize=False
):  # If debug argument (-d) is set to 1, the script will show the whole image processing pipeline
    cv2.namedWindow(title)  # Create a named window
    cv2.moveWindow(title, 50, 50)
    if resize:
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    cv2.waitKey(0)



def resize_image(image, width, height):
    """
    Re-sizes image to given width & height
    :param image: image to resize
    :param width: new width
    :param height: new height
    :return: modified image, ratio of new & old height and width
    """
    h, w = image.shape[:2]

    ratio_w = w / width
    ratio_h = h / height

    image = cv2.resize(image, (width, height))

    return image, ratio_w, ratio_h


def resize_to_32(image):
    h, w = image.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32

    ratio_h = h / new_h
    ratio_w = w / new_w

    image = cv2.resize(image, (new_w, new_h))

    return image, ratio_w, ratio_h

def clean_string(val):
    return "".join([c if 48 < ord(c) < 57 else "" for c in str(val)])




class Frame():
    def __init__(self, image) -> None:
        self.image = image

    def expand_roi(self, roi, extend = 10):
        x, y, w, h = roi
        new_x = max(0, x - extend)
        new_y = max(0, y - extend)
        x2 = min(self.image.shape[1], x+w+extend)
        y2 = min(self.image.shape[0], y+h+extend)
        new_w = x2 - new_x
        new_h = y2- new_y 
        return x,y,new_w, new_h



    def find_text_on_person(self, roi):
        # TODO crop roi to avoid resizing
        x, y, w, h = roi
        roi_image = self.image[y : y + h, x : x + w]

        min_confidence = 0.5
        resized, ratio_w, ratio_h = resize_to_32(roi_image)

        # layers used for ROI recognition
        layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # self.compute_mask()
        # masked_img_rgb = cv2.cvtColor(self.img_masked, cv2.COLOR_GRAY2RGB)
        # getting results from the model
        scores, geometry = forward_passer(NETWORK, resized, layers=layer_names)

        # decoding results from the model
        rectangles, confidences = box_extractor(scores, geometry, min_confidence)

        rectangles = np.array(rectangles)
        confidences = np.array(confidences)
        max_rect = 10
        inds = confidences.argsort()[::-1][:max_rect]
        rectangles = rectangles[inds]
        confidences = confidences[inds]

        # applying non-max suppression to get boxes depicting text regions
        boxes = non_max_suppression(rectangles, probs=confidences)
        rois = []

        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)
            new_x = x + start_x
            new_y = y + start_y
            new_w =  end_x - start_x
            new_h =  end_y - start_y
            rois.append((new_x, new_y, new_w, new_h))
            cv2.rectangle(self.image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 1)

        return rois



    def draw_ocr_result(self, ocr_df, image):
        for index, row in ocr_df.iterrows():
            try:
                num = int(row["text"])
            except ValueError:
                continue
            if num < 10_000:
                continue
            # Take the first 5 digits
            text = str(num[:5])
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (
                    0,
                    255,
                ),
                3,
            )
        

    def perform_ocr_for_text(self,roi):
        x, y, w, h = self.expand_roi(roi)
        image_roi = self.image[y:y+h, x:x+w]
        ocr_options = build_tesseract_options(psm=3)
        df = pytesseract.image_to_data(image_roi, config=ocr_options, output_type=Output.DATAFRAME)
        df = df.dropna()
        if df.empty:
            return df
        df["text"] = df["text"].apply(clean_string)
        df["text"] = pd.to_numeric(df.text, errors="coerce")
        df = df.dropna()
        if df.empty:
            return df
        print(df["text"].values)
        df["x"] = x
        df["y"] = y
        df["w"] = w
        df["h"] = h
        return df



    @timeit
    def find_text_on_every_person(self, persons_rois):
        every_rois_text = []
        for roi in persons_rois:
            rois_text = self.find_text_on_person(roi)
            every_rois_text.extend(rois_text)
        return every_rois_text

    @timeit
    def find_persons(self):
        yunet = cv2.FaceDetectorYN.create(
            model=r"C:\Users\User\Desktop\MarathonVisionOCR\libfacedetection\yunet.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        yunet.setInputSize([self.image.shape[1], self.image.shape[0]])
        _, faces = yunet.detect(self.image)  # # faces: None, or nx15 np.array

        # todo crop roi to a multiple of 32
        # todo remove head from roi
        persons_roi = []
        extend = 40
        for face in faces:
            coords = face[:-1].astype(np.int32)
            x, y, w, h, *_ = coords
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            x = max(0, x - extend)
            w = min(self.image.shape[1], w + 2 * extend)
            h = self.image.shape[0]
            # Draw face bounding box
            cv2.rectangle(self.image, (x, y), (x + w, h), (0, 255, 0), 2)

            persons_roi.append((x, y, w, h - y))

        return persons_roi



def single(image=None):
    if image is None:
        image = cv2.imread(".\data\snapshot_3.png")
    frame = Frame(image)
    persons_rois = frame.find_persons()
    
    text_rois = frame.find_text_on_every_person(persons_rois)
    
    for roi in text_rois:
        frame.perform_ocr_for_text(roi)
    
    show(image)

def run_video():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(".\data\marathon_finish_short_1.mp4")

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    # Read until video is completed
    image_list = []
    i = 0
    while cap.isOpened():
        i = i + 1
        if i > 400:
            break
        if i % 2 == 0:
            continue
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        single(frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break   

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    # height, width, _ = image_list[0].shape
    # size = (width, height)
    # fps = 5
    # video = cv2.VideoWriter(".\data\marathon_ocr.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, size)

    # for image in image_list:
    #     video.write(image)


# TODO :
# filter box after east detection, could be done on multiprocessing
# Use EAST on threshed ?
# Use EAST text detector on black hat ?


#todo move everything outside of loops
def main():
    # single()
    run_video()


if __name__ == "__main__":
    main()
