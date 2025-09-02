import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
# 위 폰트 안되면
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows

# 음수 표시 문제 해결
plt.rcParams['axes.unicode_minus'] = False

# 모델 로드
model_path = 'converted_savedmodel/model.savedmodel'
model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# 클래스 레이블
with open('converted_savedmodel/labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]


def classify_image(image_path):
    # 이미지 전처리
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()  # 원본 이미지 보관

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    result = model_layer(img_array)
    if isinstance(result, dict):
        predictions = list(result.values())[0].numpy()
    else:
        predictions = result.numpy()

    # 가장 높은 확률의 클래스
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # 이미지 출력
    plt.figure(figsize=(12, 6))

    # 왼쪽에 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f'테스트 이미지\n{os.path.basename(image_path)}', fontsize=12)
    plt.axis('off')

    # 오른쪽에 상위 3개 막대그래프
    plt.subplot(1, 2, 2)
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_names = [class_names[idx].split(' ', 1)[1] if ' ' in class_names[idx] else class_names[idx] for idx in
                  top3_indices]
    top3_scores = [predictions[0][idx] for idx in top3_indices]

    bars = plt.barh(range(len(top3_names)), top3_scores)
    plt.yticks(range(len(top3_names)), top3_names)
    plt.xlabel('신뢰도')
    plt.title('상위 3개 예측 결과', fontsize=12)
    plt.xlim(0, 1)

    # 막대에 퍼센트 표시
    for i, (bar, score) in enumerate(zip(bars, top3_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.1%}', va='center')

    plt.tight_layout()
    plt.show()

    print(f"이미지: {image_path}")
    print(f"예측 결과: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence:.2%}")

    print(f"\n상위 3개 예측:")
    for i, idx in enumerate(top3_indices, 1):
        print(f"  {i}. {class_names[idx]}: {predictions[0][idx]:.2%}")

    return predicted_class, confidence


# 현재 폴더의 이미지 파일
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
for file in os.listdir('.'):
    if any(file.lower().endswith(ext) for ext in image_extensions):
        print(f"{file}")

# 테스트
print("\n" + "=" * 50)
classify_image('사용자가 업로드한 이미지파일')
