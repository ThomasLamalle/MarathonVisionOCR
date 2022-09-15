from audioop import mul
from tkinter import Image
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
        resized = cv2.resize(image, None, fx=0.2, fy=0.1, interpolation=cv2.INTER_AREA)
        cv2.imshow(title, resized)
        cv2.waitKey(0)


def run_video():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(".\marathon_finish_10s.mp4")

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, rgb = cap.read()
        rgb = rgb[600:, 200 : rgb.shape[1] - 200]

        frame = Frame(rgb)
        frame.locate_bib_candidates()
        # frame.show_bib_candidates_contours()
        frame.perform_ocr_for_bib_candidates()
        result = frame.draw_ocr_result(rgb, frame.ocr_result_df)

        resized = cv2.resize(result, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)

        # Display the resulting frame
        cv2.imshow("Frame", resized)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


class Frame:
    def __init__(self, rgb_image) -> None:
        self.rgb = rgb_image
        self.morphed = self.morphology(rgb_image)
        self.ocr_options = self.build_tesseract_options(psm=1)

    def morphology(self, image_rgb):
        threshold = 100
        threshed = (image_rgb > threshold).all(axis=2)
        threshed = threshed * np.uint8(255)
        threshed_2 = threshed[..., np.newaxis]
        eroded = cv2.morphologyEx(threshed_2, cv2.MORPH_ERODE, (10, 10), iterations=10)
        border = 100
        eroded[:border, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        return cv2.morphologyEx(eroded, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2)

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
        return f"-c tessedit_char_whitelist={alphanumeric}" + f" --psm {psm}"

    @timeit
    def perform_ocr_for_bib_candidates(self):
        df_list = []
        # rgb_list = []*len(self.bib_candidates_rois)
        # rgb_and_rois = [*zip(rgb_list,self.bib_candidates_rois)]
        rgb_and_rois = [
            (self.rgb[y : y + h, x : x + w], (x, y, w, h))
            for x, y, w, h in self.bib_candidates_rois
        ]
        pool = ThreadPool(nodes=8)
        df_list = list(pool.imap(Frame.perform_ocr, rgb_and_rois))
        self.ocr_result_df = pd.concat(df_list)
        

    @staticmethod
    @timeit
    def perform_ocr(rgb_and_roi):
        image_roi, roi = rgb_and_roi
        x, y, w, h = roi
        gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_RGB2GRAY)
        threshed_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        ocr_options = Frame.build_tesseract_options(psm=3)
        df = pytesseract.image_to_data(
            threshed_roi, config=ocr_options, output_type=Output.DATAFRAME
        )
        df = df.dropna()
        # text = df['text'].values[0]
        # if df.empty :
        #     continue
        # if not str(text).replace(" ", ""):
        #     continue
        df["x"] = x
        df["y"] = y
        df["w"] = w
        df["h"] = h
        return df

    @staticmethod
    def draw_ocr_result(rgb, ocr_result_df):
        for index, row in ocr_result_df.iterrows():
            if row["text"] == " ":
                continue
            text = str(int(row["text"]))
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                rgb,
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

        return rgb


# TODO :
# Use multiprocessing to perform ocr
#  fix possible whites spaces and options min charac
#  fix take letter per letter and take the 5 most confident letters
# Use EAST text detector on black hat ?

debug = True


def main():

    img_rgb = io.imread(".\snapshot.jpeg")
    img_rgb = img_rgb[600:, :]
    frame = Frame(img_rgb)
    frame.locate_bib_candidates()
    # frame.show_bib_candidates_contours()
    frame.perform_ocr_for_bib_candidates()
    res = frame.draw_ocr_result(frame.rgb, frame.ocr_result_df)
    debug_imshow("Res image", res)


if __name__ == "__main__":
    # main()
    run_video()