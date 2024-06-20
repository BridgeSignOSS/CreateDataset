import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 병원 관련 수어 액션 리스트
actions = ['hospital', 'doctor', 'medicine', 'cough', 'runnynose', 'painful']
seq_length = 30
secs_for_action = 30

# MediaPipe hands 모델 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        img = cv2.flip(img, 1)
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 216, 173), 2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            joint_list = []
            if result.multi_hand_landmarks is not None:
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
                data.append(d)
            elif len(joint_list) == 1 and action in ['cough', 'runnynose', 'painful']:  # 한 손만 인식된 경우
                empty_joint = np.zeros((21, 4)).flatten()  # 빈 데이터를 생성
                empty_angle = np.zeros(15)  # 빈 각도 데이터를 생성
                d = np.concatenate([joint_list[0], empty_joint, empty_angle])  # 빈 데이터로 채움
                data.append(d)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

cap.release()
cv2.destroyAllWindows()

