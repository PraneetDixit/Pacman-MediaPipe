import keyboard as kb
import cv2
import numpy as np
import mediapipe as mp
import threading
from time import sleep

#Image processing and thresholding
#img -> Game matrix
#img2 -> Maze silhouette for setting playing area
img = np.full((500, 500), 255, dtype=np.uint8)
img2 = cv2.imread(r"Maze.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if img2[i][j] < 15:
            img[i][j] = 0

#Initializations of player
#xf -> x velocity of player
#yf -> y velocity of player
x = 229
y = 183
a = 0
xf = -1
yf = 0
d = 0
score = 0
next = 'right'

#Initializations of food and enemy
food = []
enemy = []
enemyVelocity = []
enemyNext = []

while len(enemy) < 5:
    f1 = np.random.randint(20, 480)
    f2 = np.random.randint(30, 470)
    #Checking if the enemy is not spawned on the walls
    if np.all(img[f1-10:f1+10, f2-10:f2+10] == 0):
        enemy.append([f2, f1])
        enemyVelocity.append([1, 0])
        enemyNext.append('right')

t = 0
isOver = False
flag = 1

#MediaPipe Gesture Recognition setup
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'

#Processing recognized gesture
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global next
    if len(result.gestures):
        cat = result.gestures[0][0].category_name
        if cat != 'none':
            next = cat
        print('Detected gesture:', cat)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

camstarted = threading.Event()

#Gesture recognition thread
def gestureRecognition():
    global isOver, camstarted

    with GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)

        #Wait for camera to start
        sleep(2)
        camstarted.set()
        while cap.isOpened():
            camstarted = True
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            recognizer.recognize_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))
            
            cv2.imshow('Gesture Detection', frame)
            if isOver or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

#Game thread
def game():
    global isOver, t, next, x, y, xf, yf, d, a, flag, score, food, enemy, enemyVelocity, enemyNext, img, camstarted
    
    #Wait for recognition to start
    camstarted.wait()

    while isOver != True:

        #Image for maze to be displayed
        img3 = cv2.imread(r"Maze2.jpg")

        #Mouth opening angle of pacman
        if a <= 0:
            flag = 1
        if a >= 30:
            flag = 0
        if flag:
            a += 1
        else:
            a -= 1

        #Spawning food
        while len(food) < 10:
            f1 = np.random.randint(20, 480)
            f2 = np.random.randint(30, 470)
            #Checking if the food is not spawned on the walls
            if np.all(img[f1-10:f1+10, f2-10:f2+10] == 0):
                food.append([f2, f1])
                
        for x1, y1 in food:
            cv2.rectangle(img3, (x1-5, y1-5), (x1+5, y1+5), (255, 255, 0), -1)
            #Detecting collision with food
            if x1-5 <= x <= x1+5 and y1-5 <= y <= y1+5:
                food.remove([x1, y1])
                score += 1

        for i in range(len(enemy)):
            #Randomly changing direction of enemy on every 250th iteration
            #This ensures randomness in motion and avoids getting stuck in a corner
            if t % 250 == 0:
                enemyNext[i] = np.random.choice(['up', 'down', 'left', 'right'])
            if enemyVelocity[i] == [0, 0]:
                enemyNext[i] = np.random.choice(['up', 'down', 'left', 'right'])

            x1 = enemy[i][0]
            y1 = enemy[i][1]

            #Set velocity of enemy based on direction and feasibility of movement
            if enemyNext[i] == 'right' and np.all(img[y1-7:y1+7, x1+9] != 255):
                enemyVelocity[i] = [1, 0]
            elif enemyNext[i] == 'left' and np.all(img[y1-7:y1+7, x1-9] != 255):
                enemyVelocity[i] = [-1, 0]
            elif enemyNext[i] == 'down' and np.all(img[y1+9, x1-7:x1+7] != 255):
                enemyVelocity[i] = [0, 1]
            elif enemyNext[i] == 'up' and np.all(img[y1-9, x1-7:x1+7] != 255):
                enemyVelocity[i] = [0, -1]

            #If motion is not feasible, stop for a moment and change direction
            if enemyVelocity[i][0] == 1 and np.any(img[y1-7:y1+7, x1+9] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'down', 'left'])
            elif enemyVelocity[i][0] == -1 and np.any(img[y1-7:y1+7, x1-9] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'down', 'right'])
            elif enemyVelocity[i][1] == 1 and np.any(img[y1+9, x1-7:x1+7] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'left', 'right'])
            elif enemyVelocity[i][1] == -1 and np.any(img[y1-9, x1-7:x1+7] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['down', 'left', 'right'])

            #Enemy motion
            enemy[i][0] = int(round(x1 + enemyVelocity[i][0]))
            enemy[i][1] = int(round(y1 + enemyVelocity[i][1]))
            x1 = enemy[i][0]
            y1 = enemy[i][1]
            cv2.circle(img3, (x1, y1), 6, (255, 0, 255), -1)

            #Detecting collision with enemy
            if x1-5 <= x <= x1+5 and y1-5 <= y <= y1+5:
                isOver = True

        #Scoreboard
        if isOver != True:
            cv2.ellipse(img3, (x, y), (7, 7), d, a, 360-(2*a), (0, 0, 255), -1)
            cv2.putText(img3, "Score: " + str(score), (217, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(img3, "Game Over", (210, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img3, "Score: " + str(score), (217, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    

        #Keyboard input for pacman motion
        if kb.is_pressed('up'):
            next = "up"
        if kb.is_pressed('down'):
            next = "down"
        if kb.is_pressed('left'):
            next = "left"
        if kb.is_pressed('right'):
            next = "right"
        
        #Set velocity of pacman based on direction and feasibility of movement
        if next == 'up' and np.all(img[y-16, x-9:x+9] != 255):
            yf = -1
            xf = 0
        if next == 'down' and np.all(img[y+16, x-9:x+9] != 255):
            yf = 1
            xf = 0
        if next == 'left' and np.all(img[y-9:y+9, x-16] != 255):
            xf = -1
            yf = 0
        if next == 'right' and np.all(img[y-9:y+9, x+16] != 255):
            xf = 1
            yf = 0
        
        #Update position of pacman and direction of mouth opening
        if xf > 0:
            x += 1
            d = 0
        elif xf < 0:
            x -= 1
            d = 180
        if yf > 0:
            y += 1
            d = 90
        elif yf < 0:
            y -= 1
            d = -90

        #Stop pacman if motion is not feasible
        if xf == 1 and np.any(img[y-9:y+9, x+11] == 255):
            xf = 0
        if xf == -1 and np.any(img[y-9:y+9, x-11] == 255):
            xf = 0
        if yf == 1 and np.any(img[y+11, x-9:x+9] == 255):
            yf = 0
        if yf == -1 and np.any(img[y-11, x-9:x+9] == 255):
            yf = 0

        t += 1
        cv2.imshow("Pacman", img3)
        cv2.waitKey(10)

#Threading for concurrent execution of gesture recognition and game
gesture_thread = threading.Thread(target=gestureRecognition)
game_thread = threading.Thread(target=game)

gesture_thread.start()
game_thread.start()

gesture_thread.join()
game_thread.join()

cv2.waitKey(0)
cv2.destroyAllWindows()