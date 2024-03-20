import cv2
import mediapipe as mp
import time
import pyautogui
import threading

lmlist = []
pinchFlag = False
last_thumb_x = 0
last_thumb_y = 0

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
    max_percent_diff=8
    # Calculate percentage difference between x and y values
    x_diff_percent = abs((x2 - x1) / x1) * 100
    y_diff_percent = abs((y2 - y1) / y1) * 100

    # Check if the percentage difference is within the specified range
    if min_percent_diff <= x_diff_percent <= max_percent_diff and min_percent_diff <= y_diff_percent <= max_percent_diff:
        return True, x_diff_percent, y_diff_percent
    else:
        return False, x_diff_percent, y_diff_percent
    

def moveCursor(lmlist, img):

    # Get the dimensions of the frame
    # height, width, _ = img.shape
    # print(str(height) + " - " + str(width))
    
    global last_thumb_x, last_thumb_y, last_cursor_x, last_cursor_y

    # Thumb location:
    thumb_x = lmlist[4][1]
    thumb_y = lmlist[4][2]

    # indicates movement to left
    if thumb_x < last_thumb_x:
        thumb_x = -thumb_x

    # indicates movement upwards
    if thumb_y < last_thumb_y:
        thumb_y = -thumb_y  

    if last_thumb_x == 0 or last_thumb_y == 0:
        last_thumb_x = thumb_x
        last_thumb_y = thumb_y
        return
    
    print("Here ---- ")
    # Moouse:
    # flag, cursorSpeed_x, cursorSpeed_y = check_percentage_difference(last_thumb_x, thumb_x, last_thumb_y, thumb_y)

    # print("Speed X: " + str(cursorSpeed_x) + " , Y: " + str(cursorSpeed_y))

    cursorSpeed_x=-5
    cursorSpeed_y=-5

    pyautogui.moveRel(xOffset=cursorSpeed_x, yOffset=cursorSpeed_y, duration=0.2)

    # get last mouse position   
    last_cursor_x = pyautogui.position().x
    last_cursor_y = pyautogui.position().y
    # print("Last Mouse => (" + str(last_cursor_x) + " , " + str(last_cursor_y) + ")")

    last_thumb_x = thumb_x
    last_thumb_y = thumb_y


def moveCursorV2():
    print("Starting Mouse Mover")
    while True:
        lmlist = getHandLocation()
        if len(lmlist) != 0:
            if isPinching():
                x = lmlist[8][1]
                y = lmlist[8][2]

                screen_width, screen_height = pyautogui.size()

                print("Screen: Width = " + str(screen_width) + " , Height = " + str(screen_height))
                print("Thumb: x = " + str(x) + " , Y = " + str(y))
                
                if x <= screen_width and y <= screen_height:
                    pyautogui.moveTo(x, y)
        


def getHandLocation():
    global lmlist
    return lmlist

def isPinching():
    global pinchFlag
    return pinchFlag

def main():
    print("Starting Hand Tracking")
    global lmlist, pinchFlag
    pTime = 0
    cTime = 0

    # Cam
    webCam = 0
    phoneCam = 2

    # Get the screen resolution
    screen_width, screen_height = pyautogui.size()

    cap = cv2.VideoCapture(webCam)
    detector = handDetector()

    while True:
        success, img = cap.read()
        # img = cv2.resize(img, (screen_width, screen_height))
        # Transpose the frame to flip 90 degrees to the left
        # img = cv2.transpose(frame)

        img = detector.findHands(img)
        lmlist = detector.findPosition(img)

        if len(lmlist) != 0:
            # Pinching handler
            pinchFlag, x, y = checkIfPinching(lmlist)
        
            if pinchFlag:
                cv2.putText(img, "Pinching", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (35,77,32), 3)
                # moveCursor(lmlist)
                

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # Break the loop on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    #  Create a new thread for cursor movement
    cursor_thread = threading.Thread(target=moveCursorV2)
    
    # Start the cursor thread
    cursor_thread.start()
    
    main()
    