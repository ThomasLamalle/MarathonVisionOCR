import cv2
import imutils

cascade_classifier = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml")

# Uncomment this for real-time webcam detection
# If you have more than one webcam & your 1st/original webcam is occupied,
# you may increase the parameter to 1 or respectively to detect with other webcams, depending on which one you wanna use.

# video_capture = cv2.VideoCapture(0)

# For real-time sample video detection
video_capture = cv2.VideoCapture(r"C:\Users\User\Desktop\MarathonVisionOCR\data\finish_line_1.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)

while True:
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale

    upper_body = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        # minSize = (20, 50), # Min size for valid detection, changes according to video size or body size in the video.
        # flags = cv2.CASCADE_SCALE_IMAGE
    )
    # upper_body = cascade_classifier.detectMultiScale(gray, minSize=(50, 50))
    # Draw a rectangle around the upper bodies
    for (x, y, w, h) in upper_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "Face", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video

    # stop script when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
video_capture.release()
cv2.destroyAllWindows()
