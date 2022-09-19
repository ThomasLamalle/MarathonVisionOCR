from cProfile import run
from email.mime import image
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


def find_text_on_person(roi, full_image):
    # TODO crop roi to avoid resizing
    x, y, w, h = roi
    roi_image = full_image[y : y + h, x : x + w]

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
        cv2.rectangle(full_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2)

    return rois


def show(
    title, image, resize=False
):  # If debug argument (-d) is set to 1, the script will show the whole image processing pipeline
    cv2.namedWindow(title)  # Create a named window
    cv2.moveWindow(title, 50, 50)
    if resize:
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow(title, image)
    cv2.waitKey(0)


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
        ret, rgb = cap.read()
        if not ret:
            break
        rgb = rgb[600:, 20 : rgb.shape[1] - 20]

        frame = Frame(rgb, 640, 320, multiprocess=True)
        frame.locate_bib_candidates_east()
        frame.perform_ocr_for_bib_candidates()
        frame.draw_ocr_result()

        # resized = cv2.resize(frame.res_image, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)

        # Display the resulting frame
        # cv2.imshow("Frame", resized)

        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        # break

        image_list.append(frame.res_image)
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    height, width, _ = image_list[0].shape
    size = (width, height)
    fps = 5
    video = cv2.VideoWriter(".\data\marathon_ocr.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, size)

    for image in image_list:
        video.write(image)


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


def find_text(image, width, height, min_confidence=0.5):

    # reading in image
    orig_image = image.copy()

    # resizing image
    image, ratio_w, ratio_h = resize_image(image, width, height)

    # layers used for ROI recognition
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # getting results from the model
    scores, geometry = forward_passer(NETWORK, image, layers=layer_names)

    # decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    # applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)
    rois = []
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w) - 20
        start_y = int(start_y * ratio_h) - 20
        end_x = int(end_x * ratio_w) + 20
        end_y = int(end_y * ratio_h) + 20
        roi = (start_x, start_y, end_x - start_x, end_y - start_y)
        rois.append(roi)
        cv2.rectangle(orig_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    return rois, orig_image


def clean_string(val):
    return "".join([c if 48 < ord(c) < 57 else "" for c in str(val)])


class Frame:
    def __init__(self, rgb_image, width, height, multiprocess=False, psm=3) -> None:
        self.rgb = rgb_image
        self.res_image = rgb_image.copy()
        self.ocr_options = self.build_tesseract_options(psm)
        self.resized, self.ratio_w, self.ratio_h = resize_image(self.rgb, width, height)
        self.masked_rgb = self.compute_mask()
        self.multiprocess = multiprocess

    def compute_mask(self):
        threshold = 150
        threshed_morph = (self.resized > threshold).all(axis=2)
        threshed_morph = threshed_morph * np.uint8(255)
        # self.img_masked = threshed
        # border = 100
        # eroded[:border, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.morphologyEx(threshed_morph, cv2.MORPH_ERODE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        mask = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=1)
        # threshed_inv = np.bitwise_not(threshed)
        # self.img_masked = np.bitwise_and(threshed_inv, mask)
        gray = cv2.cvtColor(self.resized, cv2.COLOR_RGB2GRAY)
        threshed_text = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.img_masked = threshed_text

    @timeit
    def locate_bib_candidates_east(self, min_confidence=0.8):

        # layers used for ROI recognition
        layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # self.compute_mask()
        # masked_img_rgb = cv2.cvtColor(self.img_masked, cv2.COLOR_GRAY2RGB)
        # getting results from the model
        scores, geometry = forward_passer(NETWORK, self.resized, layers=layer_names)

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

        extension = 0
        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * self.ratio_w) - extension
            start_y = int(start_y * self.ratio_h) - extension
            end_x = int(end_x * self.ratio_w) + extension
            end_y = int(end_y * self.ratio_h) + extension
            roi = (start_x, start_y, end_x - start_x, end_y - start_y)
            rois.append(roi)
            cv2.rectangle(self.res_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        self.bib_candidates_rois = rois

    def locate_bib_candidates(self, keep=5):
        cnts = cv2.findContours(self.morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts[0], key=cv2.contourArea, reverse=True)[:keep]

        min_ar = 0.9
        max_ar = 2
        min_area = 500

        rois = []
        contours_kept = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            area = w * h

            if ar >= min_ar and ar <= max_ar and area > min_area:
                contours_kept.append(c)
                rois.append((x, y, w, h))

        self.bib_candidates_contours = contours_kept
        self.bib_candidates_rois = rois

    def show_bib_candidates_contours(self):
        contour_image = cv2.drawContours(
            self.rgb.copy(), self.bib_candidates_contours, -1, (0, 255, 0), 2
        )
        show("Bib candidates", contour_image)

    @staticmethod
    def build_tesseract_options(psm):
        alphanumeric = "0123456789"
        options = f"-c tessedit_char_whitelist={alphanumeric}"
        options += "-c tessedit_char_blacklist='., '"
        options += f" --psm {psm}"
        return options

    @timeit
    def perform_ocr_for_bib_candidates(self):
        if self.multiprocess:
            rgb_and_rois = [
                (self.rgb[y : y + h, x : x + w], (x, y, w, h))
                for x, y, w, h in self.bib_candidates_rois
            ]
            pool = ThreadPool(nodes=8)
            df_list = list(pool.imap(self.perform_ocr, rgb_and_rois))
        else:
            df_list = []
            for x, y, w, h in self.bib_candidates_rois:
                df = self.perform_ocr((self.rgb[y : y + h, x : x + w], (x, y, w, h)))
                df_list.append(df)

        self.ocr_result_df = pd.concat(df_list)

    @staticmethod
    def perform_ocr(rgb_and_roi):
        image_roi, roi = rgb_and_roi
        x, y, w, h = roi

        # gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_RGB2GRAY)
        # threshed_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # todo test if can be moved outside with multiprocessing
        ocr_options = Frame.build_tesseract_options(psm=3)
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

    def draw_ocr_result(self):
        for index, row in self.ocr_result_df.iterrows():
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
            cv2.rectangle(self.res_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                self.res_image,
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


# TODO :
# filter box after east detection, could be done on multiprocessing
# Use EAST on threshed ?
# Use EAST text detector on black hat ?

debug = True

def perform_ocr_for_text(roi, image):
    x, y, w, h = roi
    image_roi = image[y:y+h, x:x+h]
    ocr_options = Frame.build_tesseract_options(psm=3)
    df = pytesseract.image_to_data(image, config=ocr_options, output_type=Output.DATAFRAME)
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

def ocr_snapshot():
    image = cv2.imread(".\data\snapshot_3.png")

    persons_rois = find_persons(image)
    
    text_rois = find_text_on_every_person(image, persons_rois)
    
    show("image", image, True)

@timeit
def find_text_on_every_person(image, persons_rois):
    for roi in persons_rois:
        rois_text = find_text_on_person(roi, image)

@timeit
def find_persons(frame):
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
    yunet.setInputSize([frame.shape[1], frame.shape[0]])
    _, faces = yunet.detect(frame)  # # faces: None, or nx15 np.array

    # todo crop roi to a multiple of 32
    # todo remove head from roi
    persons_roi = []
    for face in faces:
        coords = face[:-1].astype(np.int32)
        x, y, w, h, *_ = coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        extend = 40
        x = max(0, x - extend)
        w = min(frame.shape[1], w + 2 * extend)
        h = frame.shape[0]
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, h), (0, 255, 0), 2)

        persons_roi.append((x, y, w, h - y))

    return persons_roi


detector = "frozen_east_text_detection.pb"
NETWORK = cv2.dnn.readNet(detector)

#todo move everything outside of loops
def main():
    ocr_snapshot()
    # run_video()


if __name__ == "__main__":
    main()
