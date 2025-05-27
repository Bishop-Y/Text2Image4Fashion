import io
import numpy as np
import onnxruntime as rt
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

from config.config import MODEL_PATH, Z_DIM, DEVICE

sess = rt.InferenceSession(MODEL_PATH)
embedder = SentenceTransformer('stsb-roberta-large')

def generate_image_buf(prompt: str) -> io.BytesIO:
    text_emb = embedder.encode([prompt], convert_to_numpy=True).astype(np.float32)
    noise = np.random.randn(1, Z_DIM).astype(np.float32)
    fake = sess.run(None, {'text_emb': text_emb, 'noise': noise})[0]
    img_tensor = torch.tensor(fake[0], device=DEVICE)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
    img = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf