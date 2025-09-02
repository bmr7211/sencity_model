import tensorflow as tf
import numpy as np
from PIL import Image
import os

# SavedModel 경로
model_path = 'converted_savedmodel/model.savedmodel'

print(f"모델 경로: {model_path}")
print(f"saved_model.pb 존재 여부: {os.path.exists(os.path.join(model_path, 'saved_model.pb'))}")

try:
    # SavedModel --> TFSMLayer 로드
    model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    print("모델 로드 성공")

    # 클래스 레이블
    labels_path = 'converted_savedmodel/labels.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"클래스 개수: {len(class_names)}")
        print(f"클래스 목록: {class_names}")


    # 예측 함수
    def classify_image(image_path):
        # 이미지 전처리
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        result = model_layer(img_array)

        # 결과 처리
        if isinstance(result, dict):
            predictions = list(result.values())[0]
        else:
            predictions = result

        return predictions


    print("\n모델 준비 완료")

except Exception as e:
    print(f"모델 로드 실패: {e}")