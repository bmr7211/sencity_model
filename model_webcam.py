import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 모델 로드 (SavedModel → TFSMLayer 사용)
model_path = 'converted_savedmodel/model.savedmodel'
model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# 클래스 레이블 불러오기
with open('converted_savedmodel/labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    # 카메라 프레임 전처리
    img = cv2.resize(frame, (224, 224))  # 모델 입력 크기 맞추기
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_frame(frame):
    # 프레임에서 예측
    img_array = preprocess_frame(frame)
    result = model_layer(img_array)

    if isinstance(result, dict):
        predictions = list(result.values())[0].numpy()
    else:
        predictions = result.numpy()

    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence, predictions[0]

# 웹캠 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없음")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없음")
        break

    # 예측
    pred_class, conf, preds = predict_frame(frame)
    label_text = f"{class_names[pred_class]} ({conf:.2%})"

    # 화면에 결과 표시
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Classification", frame)

    # 이거 없으면 웹캠이 실행이 안돼요 왜인지는 모르겠어요...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()