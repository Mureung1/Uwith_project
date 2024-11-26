from PIL import Image
import os
from glob import glob
import numpy as np

# 기본 폴더 설정
train_dir = 'animals'  # train 디렉토리 경로
classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", 
    "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow", 
    "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly", 
    "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", 
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", 
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", 
    "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", 
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", 
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", 
    "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", 
    "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", 
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", 
    "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra" ]  # 클래스 이름 리스트

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
