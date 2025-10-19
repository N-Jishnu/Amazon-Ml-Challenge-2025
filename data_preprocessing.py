import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import download_images
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def load_data(dataset_folder):
    train_path = os.path.join(dataset_folder, 'train.csv')
    test_path = os.path.join(dataset_folder, 'test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_catalog_content(df):
    df['catalog_content'] = df['catalog_content'].fillna('').str.strip()
    return df

def extract_text_features(df, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['catalog_content'].tolist(), show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, index=df.index)
    emb_df.columns = [f'text_emb_{i}' for i in range(emb_df.shape[1])]
    df = pd.concat([df, emb_df], axis=1)
    return df

def extract_image_features(df, image_folder='images'):
    # Download images
    print('Downloading images...')
    download_images(df['image_link'].tolist(), image_folder)
    # Load pre-trained ResNet
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_features = []
    for link in df['image_link']:
        filename = os.path.basename(link)
        image_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = resnet(input_tensor)
            image_features.append(features.numpy().flatten())
        except Exception as e:
            image_features.append(np.zeros(resnet.fc.in_features))
    image_df = pd.DataFrame(image_features, index=df.index)
    image_df.columns = [f'image_emb_{i}' for i in range(image_df.shape[1])]
    df = pd.concat([df, image_df], axis=1)
    return df

def main():
    DATASET_FOLDER = 'dataset/'
    IMAGE_FOLDER = 'images'
    train, test = load_data(DATASET_FOLDER)
    train = preprocess_catalog_content(train)
    test = preprocess_catalog_content(test)
    print('Extracting text features for train...')
    train = extract_text_features(train)
    print('Extracting text features for test...')
    test = extract_text_features(test)
    print('Extracting image features for train...')
    train = extract_image_features(train, IMAGE_FOLDER)
    print('Extracting image features for test...')
    test = extract_image_features(test, IMAGE_FOLDER)
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)
    print('Sample train row:', train.iloc[0])
    print('Sample test row:', test.iloc[0])

if __name__ == "__main__":
    main()