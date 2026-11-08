import os
import asyncio
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from models import (
    recognize_entities,
    analyze_sentiment,
    answer_question
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

TOKEN = input("Введите токен Telegram-бота: ").strip()


def get_keyboard():
    keyboard = [
        [
            InlineKeyboardButton(
                "Распознавание сущностей (NER)",
                callback_data="ner"
            )
        ],
        [
            InlineKeyboardButton(
                "Анализ тональности",
                callback_data="sentiment"
            )
        ],
        [
            InlineKeyboardButton(
                "Ответы на вопросы (QA)",
                callback_data="qa"
            )
        ]
    ]

    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    await update.message.reply_text(
        "Здравствуйте! Я NLP-ассистент на основе DeepPavlov.\n\n"
        "Я умею:\n"
        "1. Распознавать имена, организации и местоположения.\n"
        "2. Определять тональность текста.\n"
        "3. Отвечать на вопросы по предоставленному тексту.\n\n"
        "Выберите нужную функцию:",
        reply_markup=get_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Как пользоваться ботом:\n\n"
        "1. Нажмите /start.\n"
        "2. Выберите функцию.\n"
        "3. Отправьте текст для анализа.\n\n"
        "Для QA сначала отправьте текст, затем вопрос.\n"
        "Чтобы выбрать другую функцию, используйте /start."
    )


async def select_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query
    await query.answer()

    mode = query.data
    context.user_data.clear()

    if mode == "ner":
        context.user_data["mode"] = "ner"
        message = (
            "Режим NER активирован.\n\n"
            "Отправьте текст, в котором нужно найти "
            "имена людей, организации и местоположения."
        )

    elif mode == "sentiment":
        context.user_data["mode"] = "sentiment"
        message = (
            "Режим анализа тональности активирован.\n\n"
            "Отправьте текст, эмоциональную окраску которого "
            "нужно определить."
        )

    elif mode == "qa":
        context.user_data["mode"] = "qa_context"
        message = (
            "Режим Question Answering активирован.\n\n"
            "Сначала отправьте текст, по которому "
            "нужно будет отвечать на вопросы."
        )

    else:
        message = "Неизвестный режим. Используйте /start."

    await query.message.reply_text(message)


async def process_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text("Отправьте непустой текст.")
        return

    mode = context.user_data.get("mode")

    if mode is None:
        await update.message.reply_text(
            "Сначала выберите функцию с помощью команды /start."
        )
        return

    if mode == "qa_context":
        context.user_data["qa_context"] = text
        context.user_data["mode"] = "qa_question"

        await update.message.reply_text(
            "Текст сохранён.\n\n"
            "Теперь отправьте вопрос по этому тексту."
        )
        return

    if mode == "qa_question":
        qa_context = context.user_data.get("qa_context")

        await update.message.reply_text(
            "Обрабатываю вопрос. Пожалуйста, подождите..."
        )

        try:
            result = await asyncio.to_thread(
                answer_question,
                qa_context,
                text
            )

            await update.message.reply_text(result)

        except Exception as error:
            logging.exception("Ошибка QA")
            await update.message.reply_text(
                f"Не удалось обработать вопрос: {error}"
            )

        context.user_data.clear()
        await update.message.reply_text(
            "Для нового анализа нажмите /start."
        )
        return

    if mode == "ner":
        await update.message.reply_text(
            "Выполняю распознавание сущностей..."
        )

        try:
            result = await asyncio.to_thread(
                recognize_entities,
                text
            )
            await update.message.reply_text(
                f"Результат NER:\n\n{result}"
            )

        except Exception as error:
            logging.exception("Ошибка NER")
            await update.message.reply_text(
                f"Ошибка при анализе текста: {error}"
            )

    elif mode == "sentiment":
        await update.message.reply_text(
            "Определяю тональность текста..."
        )

        try:
            result = await asyncio.to_thread(
                analyze_sentiment,
                text
            )
            await update.message.reply_text(
                f"Результат анализа:\n\n{result}"
            )

        except Exception as error:
            logging.exception("Ошибка анализа тональности")
            await update.message.reply_text(
                f"Ошибка при анализе текста: {error}"
            )


def main():
    if not TOKEN:
        print("Ошибка: токен не введён.")
        return

    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(
        CallbackQueryHandler(select_mode)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            process_message
        )
    )

    print("Telegram NLP-бот запущен!")

    application.run_polling()


if __name__ == "__main__":
    main()
