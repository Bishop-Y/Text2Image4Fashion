import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import tqdm
import hydra
from omegaconf import DictConfig
MODEL_NAME = 'stsb-roberta-large'


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def get_embeddings(cfg: DictConfig):
    data_dir = cfg.dataset.data_dir
    captions_path = os.path.join(data_dir, 'captions.json')

    with open(captions_path, 'r', encoding='utf-8') as f:
        captions = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = {}
    for filename, caption in tqdm.tqdm(captions.items()):
        embeddings[filename] = model.encode(caption)

    embedding_save_path = os.path.join(data_dir, 'embeddings.pickle')
    with open(embedding_save_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Эмбеддинги сохранены в файле {embedding_save_path}")


if __name__ == '__main__':
    get_embeddings()
