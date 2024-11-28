from PIL import Image
import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 기본 폴더 설정
train_dir = '/content/drive/MyDrive/animals/animals'  # train 디렉토리 경로
classes = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", 
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", 
    "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", 
    "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", 
    "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird", 
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", 
    "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", 
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig", 
    "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", 
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", 
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", 
    "wombat", "woodpecker", "zebra"]  # 클래스 이름 리스트

# 모든 이미지 경로를 담을 리스트 생성
train_image_paths = []
train_labels = []

# 각 클래스 폴더에서 이미지 파일 경로와 레이블 가져오기
for label, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    image_files = glob(os.path.join(class_dir, '*.jpg'))  # 예: train/cat/*.jpg
    
    train_image_paths.extend(image_files)  # 파일 경로 추가
    train_labels.extend([label] * len(image_files))  # 각 이미지에 대한 클래스 레이블 추가    
# 이미지 리사이즈 및 정규화를 위한 빈 리스트 생성
resized_images = []

# 이미지 파일 경로를 하나씩 열어 리사이즈 및 정규화
for image_path in train_image_paths:
    print(f"{image_path} 전처리중...")
    img = Image.open(image_path)
    img_resized = img.resize((224, 224))  # 224x224로 리사이즈
    img_array = np.array(img_resized) / 255.0  # 0-1로 정규화
    resized_images.append(img_array)

print(f"총 {len(resized_images)}개의 이미지가 전처리되었습니다.")

# 데이터 배열 변환
resized_images = np.array(resized_images)  # 이미지를 numpy 배열로 변환
train_labels = np.array(train_labels)  # 레이블을 numpy 배열로 변환

# 데이터 분리 (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    resized_images, train_labels, test_size=0.2, random_state=42
)

# CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classes), activation='softmax')  # 클래스 개수만큼 출력
])

# 모델 요약
model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습
history = model.fit(
    X_train, y_train,
    epochs=10,  # 에포크 수는 데이터 크기에 따라 조정
    batch_size=32,
    validation_data=(X_val, y_val)
)

# 학습 결과 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 모델 저장
model.save('animal_classifier_model.h5')

print("모델 학습이 완료되었습니다. 모델이 저장되었습니다.")