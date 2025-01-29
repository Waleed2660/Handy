import cv2
import mediapipe as mp
import time
import pyautogui

last_cursor_x = 0
last_cursor_y = 0

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon)
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist
    

def checkIfPinching(lmlist):
    thumb_x = lmlist[4][1]
    thumb_y = lmlist[4][2]
    finger_x = lmlist[8][1]
    finger_y = lmlist[8][2]

    return check_percentage_difference(thumb_x, finger_x, thumb_y, finger_y)



def check_percentage_difference(x1, x2, y1, y2):
    min_percent_diff=0
    max_percent_diff=18
    # Calculate percentage difference between x and y values
    x_diff_percent = abs((x2 - x1) / x1) * 100
    y_diff_percent = abs((y2 - y1) / y1) * 100

    # Check if the percentage difference is within the specified range
    return min_percent_diff <= x_diff_percent <= max_percent_diff and min_percent_diff <= y_diff_percent <= max_percent_diff
    

def moveCursor():

    pyautogui.moveRel(xOffset=50, yOffset=50, duration=0.1)

    # last_cursor_x = 
    # last_cursor_y = 


def main():
    pTime = 0
    cTime = 0

    # Cam
    webCam = 0
    phoneCam = 2

    cap = cv2.VideoCapture(webCam)
    detector = handDetector()

    while True:
        success, frame = cap.read()
        # Transpose the frame to flip 90 degrees to the left
        img = cv2.transpose(frame)

        raw_image = detector.findHands(img)
        lmlist = detector.findPosition(img)

        # Rotate the image 90 degrees to the right
        img = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        if len(lmlist) != 0 and checkIfPinching(lmlist):
            cv2.putText(img, "Pinching", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (35,77,32), 3)

        # if len(lmlist) != 0:
            # print("Thumb:" + str(lmlist[4]))
            # print("Finger:" + str(lmlist[8]))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    