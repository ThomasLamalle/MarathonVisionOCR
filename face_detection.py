import cv2
import imutils
import numpy as np

yunet = cv2.FaceDetectorYN.create(
    model=r"C:\Users\User\Desktop\MarathonVisionOCR\libfacedetection\yunet.onnx",
    config='',
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=50,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# For real-time sample video detection
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\MarathonVisionOCR\data\finish_line_1.mp4")

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
yunet.setInputSize([frame_w, frame_h])

while cap.isOpened():
    ret, frame = cap.read()
    _, faces = yunet.detect(frame) # # faces: None, or nx15 np.array

    for face in faces:
        coords = face[:-1].astype(np.int32)
        x,y,w,h, *_ = coords
        # Draw face bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    winname = "MediaPipe Selfie Segmentation"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 0, 0)
    output_image = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow(winname, output_image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
