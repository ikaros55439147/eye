import cv2
import numpy as np
import dlib
import pyautogui
from collections import deque
import math

# --- CONFIGURATION (可調整參數) ---
# 1. 時間平滑設定
DEQUE_SIZE = 5  # 越大越平滑，但延遲越高。建議 3-10
gaze_ratio_deque_h = deque(maxlen=DEQUE_SIZE)
gaze_ratio_deque_v = deque(maxlen=DEQUE_SIZE)

# 2. 座標映射範圍 (請根據您自己的情況調整)
# 執行程式後，觀察您看螢幕最左/右/上/下時的 H/V Ratio 平均值來設定
INPUT_RATIO_H_MIN = 0.30
INPUT_RATIO_H_MAX = 0.67
INPUT_RATIO_V_MIN = 0.29
INPUT_RATIO_V_MAX = 0.44

# 3. 滑鼠移動平滑設定
SMOOTHING_FACTOR = 0.1  # 越小越平滑，但延遲越高。建議 0.05 - 0.2

# 4. 靜止區設定
DEADZONE_RADIUS = 5  # 半徑 (像素)。目標點在此範圍內，滑鼠不移動

# --- 初始化 ---
print("正在啟動，請稍候...")
pyautogui.FAILSAFE = False
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法打開攝影機")
    exit()

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("錯誤：找不到 'shape_predictor_68_face_landmarks.dat' 模型檔案。")
    exit()

# --- 輔助函式 (與之前相同) ---
def get_eye_landmarks(landmarks, eye_points_indices):
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points_indices], dtype=np.int32)
    return points

def get_gaze_ratio(eye_points, gray_frame):
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_roi = gray_frame[y: y + h, x: x + w]
    if w == 0 or h == 0: return None
    
    # 增加對比度，讓瞳孔更明顯
    eye_roi = cv2.equalizeHist(eye_roi)
    
    _, threshold_eye = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    if contours:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            horizontal_ratio = cx / w
            vertical_ratio = cy / h
            return (horizontal_ratio, vertical_ratio)
    return None

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

print("\n--- 程式已啟動 ---")
print("請嘗試看向螢幕的四個角落來校準你的映射範圍。")
print(f"您的螢幕解析度為: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
print("按 'q' 鍵退出程式。")

# --- 主迴圈 ---
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        right_eye_points = get_eye_landmarks(landmarks, RIGHT_EYE_POINTS)
        left_eye_points = get_eye_landmarks(landmarks, LEFT_EYE_POINTS)

        gaze_ratio_left = get_gaze_ratio(left_eye_points, gray)
        gaze_ratio_right = get_gaze_ratio(right_eye_points, gray)

        if gaze_ratio_left and gaze_ratio_right:
            current_h_ratio = (gaze_ratio_left[0] + gaze_ratio_right[0]) / 2
            current_v_ratio = (gaze_ratio_left[1] + gaze_ratio_right[1]) / 2

            # NEW: 將當前偵測到的比例值存入 deque
            gaze_ratio_deque_h.append(current_h_ratio)
            gaze_ratio_deque_v.append(current_v_ratio)

            # NEW: 計算移動平均值
            avg_h_ratio = np.mean(gaze_ratio_deque_h)
            avg_v_ratio = np.mean(gaze_ratio_deque_v)

            # 將平滑後的值映射到螢幕座標
            screen_x = np.interp(avg_h_ratio, [INPUT_RATIO_H_MIN, INPUT_RATIO_H_MAX], [0, SCREEN_WIDTH])
            screen_y = np.interp(avg_v_ratio, [INPUT_RATIO_V_MIN, INPUT_RATIO_V_MAX], [0, SCREEN_HEIGHT])
            
            # 限制座標在螢幕範圍內
            screen_x = max(0, min(SCREEN_WIDTH - 1, screen_x))
            screen_y = max(0, min(SCREEN_HEIGHT - 1, screen_y))

            # NEW: 引入平滑移動和靜止區
            current_mouse_x, current_mouse_y = pyautogui.position()
            target_x = current_mouse_x + (screen_x - current_mouse_x) * SMOOTHING_FACTOR
            target_y = current_mouse_y + (screen_y - current_mouse_y) * SMOOTHING_FACTOR

            # 計算目標與當前的距離
            distance = math.sqrt((target_x - current_mouse_x)**2 + (target_y - current_mouse_y)**2)
            
            # 只有在距離大於靜止區半徑時才移動
            if distance > DEADZONE_RADIUS:
                pyautogui.moveTo(target_x, target_y)

            # 在畫面上顯示更多除錯資訊
            cv2.putText(display_frame, f"H_Ratio (Avg): {avg_h_ratio:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, f"V_Ratio (Avg): {avg_v_ratio:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Target: ({int(screen_x)}, {int(screen_y)})", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Advanced Eye Tracking (Press 'q' to quit)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程式已結束。")