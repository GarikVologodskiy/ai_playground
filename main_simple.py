from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import time

# from openai.types.responses import response

# Загружаем переменные окружения из файла .env (например, OPENAI_API_KEY)
# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()

# Инициализируем клиент OpenAI
# Initialize the OpenAI client
# По умолчанию библиотека ищет ключ в переменной окружения OPENAI_API_KEY
# By default, the library looks for the key in the OPENAI_API_KEY environment variable
client = OpenAI()

# Список для хранения истории диалога.
# Первое сообщение задает контекст поведения нейросети (роль "system").
# messages = [
#     {
#         "role": "system",
#         "content": "Ты полезный и понятный ассистент"
#     }
# ]

# Системный промпт для извлечения количества людей из текста объявлений
# System prompt for extracting the number of people from advertisement texts
SYSTEM_PROMPT = """
Ты извлекаешь число людей из объявления
Верни только JSON

{amount: 4}

Правила:
- 2 семьи по 3 человека = 6
- пара = 2
- семья из 4 = 4
- с ребенком добавлять ребенка
- если неясно - 1
"""

# Читаем входные данные из CSV файла
# Read input data from a CSV file
df = pd.read_csv(
    'submission100lines.csv'
)

# Список для хранения обработанных результатов
# List to store processed results
results = []

# Проходим по каждой строке в DataFrame
# Iterate through each row in the DataFrame
for i, row in df.iterrows():
    text = row["text"]

    # Отправляем запрос к OpenAI API для анализа текста
    # Send a request to the OpenAI API to analyze the text
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens = 20,
        messages = [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user", "content": text
            }
        ],
        response_format={"type": "json_object"}
    )

    # Пытаемся распарсить JSON ответ и извлечь количество
    # Try to parse the JSON response and extract the amount
    try:
        data = json.loads(response.choices[0].message.content)
        amount = int(data["amount"])
    except:
        # Если произошла ошибка, устанавливаем значение по умолчанию
        # If an error occurs, set the default value
        amount = 1
    
    # Добавляем результат в список
    # Add the result to the list
    results.append(
        {
            "amount": amount,
            "text_id": row["text_id"],
            "text":text
        }
    )

    # Небольшая задержка, чтобы не превысить лимиты API
    # Small delay to avoid hitting API rate limits
    time.sleep(0.2)

# Создаем новый DataFrame из результатов и сохраняем его в CSV
# Create a new DataFrame from the results and save it to CSV
out = pd.DataFrame(results)
out.to_csv(
    "submission.csv",
    index = False
)

# Выводим сообщение о завершении работы
# Print a message when finished
print("Ready")

# print("Чат запущен. Для выхода нажми: Exit")
#
# # Основной цикл работы чат-бота
# while True:
#     # Получаем ввод от пользователя, обрезая лишние пробелы по краям
#     user_text = input("\nТы:").strip()
#
#     # Проверка команды на выход из программы
#     if user_text.lower() == "exit":
#         print("Чат завершен")
#         break
#
#     # Если пользователь ввел пустую строку, пропускаем итерацию
#     if not user_text:
#         continue
#
#     # Добавляем сообщение пользователя в список истории сообщений
#     messages.append({"role": "user", "content": user_text})
#
#     # Отправляем запрос к API OpenAI для генерации ответа
#     # Передаем модель и всю накопленную историю диалога для сохранения контекста
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.7,
#         max_tokens=100
#     )
#
#     # Извлекаем текстовое содержание ответа из структуры ответа API
#     answer = response.choices[0].message.content
#
#     # Выводим ответ бота в консоль
#     print(f"\nБот: {answer}")
#
#     # Добавляем ответ нейросети в историю, чтобы при следующем запросе бот "помнил" свои слова
#     messages.append({"role": "assistant", "content": answer})

