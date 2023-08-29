import cv2
import mediapipe as mp
import pyautogui as pg

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:

    success, image = video.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Hand.process(image)
    h, w, _ = image.shape
    # print(f'altura: {h}, largura: {w}')
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.line(image, (int(w / 2), 0), (int(w / 2), int(h)), (255,0,0), 2)
    cv2.line(image, (0, int(h / 2)), (w, int(h / 2)), (255,0,0), 2)

    handsPoints = results.multi_hand_landmarks
    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(image, points, hand.HAND_CONNECTIONS)

            for id, cord in enumerate(points.landmark):
                if id == 8:
                    # print(int(cord.x * w), int(cord.y * h))
                    posX, posY = int(cord.x * w), int(cord.y * h)
                    if posX >  320:
                        # -abs(posX)
                        posX = posX * (-1)
                    if posY < 240:
                        posY = posY * (-1)
                    posX, posY = posX / 10, posY / 10
                    currentX, currentY = pg.position()
                    # print(f'atual X: {currentX}, atual Y: {currentY}')
                    # print(posX, posY)
                    # print(f'prox X: {currentX + posX}, prox Y: {(currentY + posY)}')
                    # print('--------------------------------')
                    if (currentX + posX) >= 1915:
                        pg.moveTo(1900, (currentY + posY))
                    elif (currentY + posY) >= 1075:
                        pg.moveTo((currentX + posX), 1070)
                    else:
                        pg.move(posX, posY)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()