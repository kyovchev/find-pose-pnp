import cv2
import time

# Params
DEVICE_ID = 0
PATH = './capture'

cap = cv2.VideoCapture(DEVICE_ID)

if not cap.isOpened():
    print('Cannot open camera!')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot retrieve camera frame!')
        break

    frame2 = frame.copy()

    cv2.putText(frame2, 'C=Capture, Esc=Quit', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Camera', frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f'{PATH}/capture_{int(time.time())}.png'
        cv2.imwrite(filename, frame)
        print(f'Image saved as {filename}')

    elif key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
