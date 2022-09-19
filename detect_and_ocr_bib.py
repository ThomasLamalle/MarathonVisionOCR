from cProfile import run
from email.mime import image
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
import cv2
from skimage import io
import numpy as np
from pathos.threading import ThreadPool

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


def debug_imshow(
    title, image, convert_rgb=True
):  # If debug argument (-d) is set to 1, the script will show the whole image processing pipeline
    if debug:  # and wait for user input before continuing to the next step.
        if convert_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resized = cv2.resize(image, None, fx=0.2, fy=0.1, interpolation=cv2.INTER_AREA)
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
        i = i+1
        if i> 400 :
            break
        if i%2==0:
            continue
        # Capture frame-by-frame
        ret, rgb = cap.read()
        if not ret:
            break
        rgb = rgb[600:, 20 : rgb.shape[1] - 20]

        frame = Frame(rgb, 640,320, multiprocess=True)
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
    size = (width,height)
    fps = 5
    video = cv2.VideoWriter('.\data\marathon_ocr.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
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
        mask =  cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=1)
        # threshed_inv = np.bitwise_not(threshed)
        # self.img_masked = np.bitwise_and(threshed_inv, mask)
        gray = cv2.cvtColor(self.resized, cv2.COLOR_RGB2GRAY)
        threshed_text = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
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
        debug_imshow("Bib candidates", contour_image)

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
        df['text'] = df['text'].apply(clean_string)
        df['text'] = pd.to_numeric(df.text, errors='coerce')
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
            if num < 10_000 :
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


def ocr_snapshot():

    img_rgb = io.imread(".\data\snapshot.jpeg")
    # img_rgb = cv2.imread(".\data\snapshot.jpeg", cv2.IMREAD_UNCHANGED)
    img_rgb = img_rgb[500:, :]
    # _,detection = find_text(img_rgb, 640, 320, 0.999)
    # cv2.imshow("res", detection)
    # cv2.waitKey(0)
    frame = Frame(img_rgb, 640, 320, multiprocess=False)
    frame.compute_mask()
    # frame.locate_bib_candidates_east()
    plt.figure(figsize=(10, 10))
    plt.imshow(frame.img_masked, cmap='gray')
    plt.show()
    cv2.waitKey()
    # frame.perform_ocr_for_bib_candidates()

    # frame.draw_ocr_result()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(frame.res_image)
    # plt.show()
    # cv2.waitKey()

detector = "frozen_east_text_detection.pb"
NETWORK = cv2.dnn.readNet(detector)


def main():
    # ocr_snapshot()
    run_video()


if __name__ == "__main__":
    main()
