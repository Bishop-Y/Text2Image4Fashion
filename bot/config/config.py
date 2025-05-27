from dotenv import load_dotenv
import os
import torch

load_dotenv()

API_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN_API', '')
MODEL_PATH = os.getenv('MODEL_PATH', 'model.onnx')
DB_PATH = os.getenv('DB_PATH', 'results.db')

Z_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
