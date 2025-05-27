import telebot
from telebot.types import ReplyKeyboardMarkup

from config.config import API_TOKEN
from db.db import save_generation, get_last_prompt
from generator.generator import generate_image_buf

bot = telebot.TeleBot(API_TOKEN)


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
        prompt = get_last_prompt(user_id)
        if not prompt:
            bot.send_message(user_id, "У вас пока нет предыдущих запросов.")
            return
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
