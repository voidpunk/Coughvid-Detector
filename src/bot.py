from modules.helper_functions import label_func
from modules.data_pipeline import pipeline
from fastai.vision.all import *
from pydub import AudioSegment
from secrets import bot_key
from time import time
import telebot
import os


def ogg_to_wav(path):
    file_ogg = AudioSegment.from_ogg(os.path.join(path, 'audio.ogg'))
    file_ogg.export(os.path.join(path, 'audio.wav'), format='wav')

def clear_cache(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))

bot = telebot.TeleBot(bot_key)

@bot.message_handler(commands=['start'])
def send_welcome(message):
	bot.reply_to(message, "Hello, I'm a bot!")

@bot.message_handler(commands=['help'])
def send_welcome(message):
	bot.reply_to(message, "Hello, you need help!")

@bot.message_handler(commands=['info'])
def send_welcome(message):
	bot.reply_to(message, "Hello, you need info!")

@bot.message_handler(func=lambda m: True)
def echo_all(message):
	bot.reply_to(message, message.text)

@bot.message_handler(content_types=['voice'])
def handle_audio(message):
    t0 = time()
    file_info = bot.get_file(message.voice.file_id)
    file = bot.download_file(file_info.file_path)
    with open('./cache/audio.ogg', 'wb') as audio:
        audio.write(file)
    t1 = time()
    ogg_to_wav('./cache')
    t2 = time()
    reply = pipeline('./cache', './models/xgb', './models/cnn', )
    t3 = time()
    if reply == 'NA':
        bot.reply_to(message, 'No cough detected')
    else:
        bot.reply_to(message, 'Cough detected')
    bot.send_message(message.chat.id, reply)
    print(f'Time to transfer the audio:\t{t1 - t0}')
    print(f'Time to convert the audio:\t{t2 - t1}')
    print(f'Time to predict the label:\t{t3 - t2}')
    clear_cache('./cache')

bot.infinity_polling()
