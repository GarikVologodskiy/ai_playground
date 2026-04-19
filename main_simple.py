from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из файла .env (например, OPENAI_API_KEY)
load_dotenv()

# Инициализируем клиент OpenAI
# По умолчанию библиотека ищет ключ в переменной окружения OPENAI_API_KEY
client = OpenAI()

# Список для хранения истории диалога.
# Первое сообщение задает контекст поведения нейросети (роль "system").
messages = [
    {
        "role": "system",
        "content": "Ты полезный и понятный ассистент"
    }
]

print("Чат запущен. Для выхода нажми: Exit")

# Основной цикл работы чат-бота
while True:
    # Получаем ввод от пользователя, обрезая лишние пробелы по краям
    user_text = input("\nТы:").strip()

    # Проверка команды на выход из программы
    if user_text.lower() == "exit":
        print("Чат завершен")
        break

    # Если пользователь ввел пустую строку, пропускаем итерацию
    if not user_text:
        continue

    # Добавляем сообщение пользователя в список истории сообщений
    messages.append({"role": "user", "content": user_text})

    # Отправляем запрос к API OpenAI для генерации ответа
    # Передаем модель и всю накопленную историю диалога для сохранения контекста
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )

    # Извлекаем текстовое содержание ответа из структуры ответа API
    answer = response.choices[0].message.content

    # Выводим ответ бота в консоль
    print(f"\nБот: {answer}")

    # Добавляем ответ нейросети в историю, чтобы при следующем запросе бот "помнил" свои слова
    messages.append({"role": "assistant", "content": answer})

