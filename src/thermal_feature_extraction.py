import numpy as np
import h5py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image

class EmbeddingExtractor:
    def __init__(self, thermal_model_path, rgb_pretrained='vggface2'):
        # Load models
        self.thermal_model = load_model(thermal_model_path, compile=False)
        self.rgb_model = InceptionResnetV1(pretrained=rgb_pretrained).eval()

    def preprocess_thermal(self, image_path, target_size=(72, 96)):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array

    def preprocess_rgb(self, image_path, target_size=(160, 160)):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img)
        img = (img - 127.5) / 128.0  # Normalization for InceptionResnetV1
        img = np.transpose(img, (2, 0, 1))  # Change to channel-first format
        return torch.tensor(img).float().unsqueeze(0)

    def extract_thermal_embedding(self, image_path):
        preprocessed_img = self.preprocess_thermal(image_path)
        embedding = self.thermal_model.predict(preprocessed_img)
        return embedding.flatten()

    def extract_rgb_embedding(self, image_path):
        preprocessed_img = self.preprocess_rgb(image_path)
        with torch.no_grad():
            embedding = self.rgb_model(preprocessed_img).numpy()
        return embedding.flatten()

    def process_images(self, base_dir, modality, extract_func):
        embeddings = []
        file_paths = []

        for root, dirs, files in os.walk(os.path.join(base_dir, modality)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    embedding = extract_func(file_path)
                    embeddings.append(embedding)
                    file_paths.append(file_path)

        return np.array(embeddings), file_paths

    def save_embeddings(self, embeddings, file_paths, output_file):
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('file_paths', data=np.array(file_paths, dtype=dt))

# Example usage:
if __name__ == "__main__":
    extractor = EmbeddingExtractor('/path/to/thermal/model.h5')
    rgb_base_dir = '/path/to/rgb/images'
    thermal_base_dir = '/path/to/thermal/images'

    rgb_embeddings, rgb_file_paths = extractor.process_images(rgb_base_dir, 'rgb', extractor.extract_rgb_embedding)
    thermal_embeddings, thermal_file_paths = extractor.process_images(thermal_base_dir, 'thermal', extractor.extract_thermal_embedding)

    extractor.save_embeddings(rgb_embeddings, rgb_file_paths, 'rgb_embeddings.h5')
    extractor.save_embeddings(thermal_embeddings, thermal_file_paths, 'thermal_embeddings.h5')
