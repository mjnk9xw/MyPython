# import cv2
# video_capture = cv2.VideoCapture(0)
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break