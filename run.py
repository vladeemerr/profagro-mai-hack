import os
import asyncio

from aiogram import types, Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from gc_api import GigaChatAPI

import utils

dp = Dispatcher()


class Start(StatesGroup):
    menu = State()
    chat = State()
    checklist = State()

reply_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='Задать вопрос')],
        [KeyboardButton(text='Получить чек-лист')]
    ],
    resize_keyboard=True, input_field_placeholder='Выберите пункт меню.',
    one_time_keyboard=True
)


reply_back = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='Назад')]
    ],
    resize_keyboard=True
)

raw_text = utils.get_text_from_pdf(['./data/1.pdf', 
                  './data/2.pdf', 
                  './data/3.pdf', 
                  './data/4.pdf',
                  './data/5.pdf',
                  './data/6.pdf'])
print('Processed documents')
chunks = utils.get_text_chunks(raw_text, chunk_size=1000, chunk_overlap=200)
print('Processed text')

embeddings = utils.make_embeddings(chunks)
print('Embeddings made')
collection = utils.create_vectorstorage(chunks, embeddings, '123')
print('Collection created')

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.set_state(Start.menu)
    await message.answer('ПРИВЕТ ВЫБЕРИ', reply_markup=reply_menu)

@dp.message(Start.menu, F.text == 'Задать вопрос')
async def ask_question(message: Message, state: FSMContext):
    await state.set_state(Start.chat)
    await message.answer(
        'Задайте вопрос',
        reply_markup=reply_back
    )

@dp.message(F.text == 'Назад')
async def stop_chat(message: Message, state: FSMContext):
    await state.set_state(Start.menu)
    await message.answer(
        'Выберите режим генерации',
        reply_markup=reply_menu
    )

@dp.message(Start.chat)
async def handle_message(message: types.Message):
  answer, top_chunks = utils.get_answer_llm(message.text, chunks, collection)
  await message.answer(answer)

@dp.message(Start.menu, F.text == 'Получить чек-лист')
async def check_list(message: Message, state: FSMContext):
    await state.set_state(Start.checklist)
    await message.answer(
        'С чем Вам необходима помощь?',
        reply_markup=reply_back
    )

@dp.message(Start.checklist)
async def handle_message_checklist(message: types.Message):
  answer, top_chunks = utils.get_checklist_llm(message.text, chunks, collection)
  await message.answer(answer)

async def main():
  bot = Bot(token='7698371369:AAHSSKRG55rb-WMokLhurjFrmulFaGGZx28')
  await dp.start_polling(bot)

if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('Bot stopped')
