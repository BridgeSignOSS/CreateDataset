import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

actions = ['hospital', 'doctor', 'medicine']
seq_length = 30

# 모델 파일 확장자를 .keras로 변경
model = load_model('model2.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 최대 두 손 인식
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
prev_time = 0  # 이전 프레임 시간 초기화

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cur_time = time.time()
    if cur_time - prev_time > 1./20:  # 20 프레임으로 제한
        prev_time = cur_time

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            joint_list = []
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]  # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize v

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])
                joint_list.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(joint_list) == 2:  # 두 손이 인식된 경우
                d = np.concatenate(joint_list)  # 두 손의 데이터를 결합
                seq.append(d)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                print(f"Input data shape: {input_data.shape}")  # 디버깅 메시지

                try:
                    y_pred = model.predict(input_data).squeeze()
                    print(f"Prediction: {y_pred}")

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    print(f"Predicted action: {actions[i_pred]} with confidence {conf}")

                    if conf < 0.9:
                        print("Confidence too low, skipping")
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)
                    print(f"Action sequence: {action_seq}")

                    if len(action_seq) < 3:
                        continue

                    this_action = '?'
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action

                    cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    print(f"Action recognized: {this_action}")
                except Exception as e:
                    print(f"Error during prediction: {e}")

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Program ended successfully")
