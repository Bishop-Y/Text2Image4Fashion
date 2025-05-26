import os
import io
import time
import sqlite3
import numpy as np
import onnxruntime as rt
import telebot
from telebot.types import ReplyKeyboardMarkup
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

API_TOKEN = 'TELEGAM TOKEN'
MODEL_PATH = 'PATH TO MODEL'
Z_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DB_PATH = 'PATH TO DB'
os.makedirs('results', exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS generations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    timestamp TEXT    NOT NULL,
    prompt    TEXT    NOT NULL,
    file_path TEXT    NOT NULL
)
""")
conn.commit()

sess = rt.InferenceSession(MODEL_PATH)
embedder = SentenceTransformer('stsb-roberta-large')
bot = telebot.TeleBot(API_TOKEN)


def save_generation(user_id: int, prompt: str, img_buf: io.BytesIO):
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{user_id}_{ts}.png"
    path = os.path.join('results', filename)
    with open(path, 'wb') as f:
        f.write(img_buf.getbuffer())
    c.execute(
        "INSERT INTO generations (user_id, timestamp, prompt, file_path) VALUES (?, ?, ?, ?)",
        (user_id, ts, prompt, path)
    )
    conn.commit()


def generate_image_buf(prompt: str) -> io.BytesIO:
    text_emb = embedder.encode(
        [prompt], convert_to_numpy=True).astype(np.float32)
    noise = np.random.randn(1, Z_DIM).astype(np.float32)
    fake = sess.run(None, {'text_emb': text_emb, 'noise': noise})[0]
    img_tensor = torch.tensor(fake[0], device=DEVICE)
    img_tensor = (img_tensor - img_tensor.min()) / \
        (img_tensor.max() - img_tensor.min())
    img = Image.fromarray(
        (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.send_message(
        msg.chat.id,
        "Привет! Чтобы сгенерировать изображение, напиши запрос на английском языке."
        "\nНапример: \"Woman in a black pants\"."
    )


@bot.message_handler(func=lambda m: True)
def on_prompt_or_regenerate(msg):
    user_id = msg.chat.id
    text = msg.text.strip()

    if text == "🔄 Перегенерировать последний":
        c.execute(
            "SELECT prompt FROM generations WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,)
        )
        row = c.fetchone()
        if not row:
            bot.send_message(user_id, "У вас пока нет предыдущих запросов.")
            return
        prompt = row[0]

        bot.send_message(
            user_id, f"Перегенерирую последний запрос:\n«{prompt}»…")
    else:
        prompt = text
        bot.send_message(user_id, f"Генерирую по запросу:\n«{prompt}»…")

    try:
        buf = generate_image_buf(prompt)
        save_generation(user_id, prompt, buf)
        markup = ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add("🔄 Перегенерировать последний")
        bot.send_photo(user_id, photo=buf, reply_markup=markup)
    except Exception as e:
        bot.send_message(user_id, f"Ошибка при генерации: {e}")


print("Бот запущен")
bot.polling(none_stop=True)
