import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import tqdm
import config.config as cfg

captions_path = os.path.join(cfg.DATA_DIR, 'captions.json')

with open(captions_path, 'r', encoding='utf-8') as f:
    captions = json.load(f)

captions_subset = {k: captions[k] for k in list(captions.keys())}
print(f"Будет использовано {len(captions_subset)} описаний.")

model = SentenceTransformer('stsb-roberta-large')

embeddings = {}
for filename, caption in tqdm.tqdm(captions_subset.items()):
    embedding = model.encode(caption)
    embeddings[filename] = embedding

# Сохраняем эмбеддинги в файл
embedding_save_path = os.path.join(cfg.DATA_DIR, 'embeddings.pickle')
with open(embedding_save_path, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Эмбеддинги сохранены в файле {embedding_save_path}")
