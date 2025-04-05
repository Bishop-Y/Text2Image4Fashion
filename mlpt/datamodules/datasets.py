import os
import json
import pickle
from PIL import Image
import torch
import torch.utils.data as data
import mlpt.config.config as cfg

class DeepFashionCaptionDataset(data.Dataset):
    def __init__(self, data_dir, split='train', transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        captions_path = os.path.join(data_dir, 'captions.json')
        
        with open(captions_path, 'r', encoding='utf-8') as f:
            self.captions_dict = json.load(f)
        
        embedding_path = os.path.join(data_dir, 'embeddings.pickle')
        if os.path.isfile(embedding_path):
            with open(embedding_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = None

        # Отфильтруем записи: оставляем только те, для которых существует изображение
        valid_items = []
        for fname in self.captions_dict.keys():
            img_path = os.path.join(self.image_dir, fname)
            if os.path.isfile(img_path):
                valid_items.append(fname)
            valid_items = valid_items[:max_samples]
        self.filenames = valid_items
        print(f"Доступных файлов после фильтрации: {len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        if self.embeddings is not None and filename in self.embeddings:
            text_embedding = torch.tensor(self.embeddings[filename])
        else:
            text_embedding = torch.randn(cfg.TEXT_DIMENSION)

        prompt = self.captions_dict.get(filename, "Промпт не найден")

        return image, text_embedding, prompt
