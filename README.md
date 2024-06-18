# CreateDataset

## 프로젝트 개요
이 프로젝트는 수어 인식 및 변환을 위한 데이터셋을 생성하고, 이를 활용한 모델을 학습 및 테스트하기 위한 것입니다.

=> 손 동작을 인식하고 이를 수어로 변환하는 기능을 제공합니다

 
## 파일 설명
- `speech_to_sign_language.py`: 음성을 텍스트로 변환하고, 이를 수어로 변환하는 스크립트
- `sign_language_converter.py`: 텍스트를 수어로 변환하는 스크립트
- `data_collection2.py`: 손 동작 데이터를 수집하는 스크립트
- `test.py`: 손 동작 인식 및 예측을 수행하는 스크립트
- `test2.py`: 손 동작 인식 및 예측을 수행하는 스크립트 (두 손, 수정필요)
- `train.ipynb`: 모델 학습을 위한 Jupyter Notebook 파일
- `train2.ipynb`: 모델 학습을 위한 Jupyter Notebook 파일 (두 손)
- `model.keras`: 학습된 수어 인식 모델 파일
- `model2.keras`: 학습된 수어 인식 모델 파일 (두 손)

## 설치 방법
프로젝트를 실행하기 위해서는 다음 라이브러리들이 필요합니다:
```bash
pip install speechrecognition pyttsx3 mediapipe numpy opencv-python tensorflow
