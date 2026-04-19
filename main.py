from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence, TextIO

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

# Значения конфигурации чата по умолчанию
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and clear assistant."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 300
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2

# Инициализация логгера
LOGGER = logging.getLogger("openai_chat")


class ConfigurationError(ValueError):
    """Исключение, возникающее при отсутствии или неверной конфигурации."""


@dataclass(frozen=True)
class AppConfig:
    """Структура данных для хранения настроек приложения."""
    api_key: str
    model: str
    system_prompt: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    stream: bool
    debug: bool
    message: str | None


class ChatSession:
    """Класс для управления сессиями чата OpenAI."""
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # Инициализация клиента OpenAI с заданными параметрами
        self._client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._previous_response_id: str | None = None
        self._turn_count = 0

    @property
    def stream_enabled(self) -> bool:
        """Проверка, включен ли режим потоковой передачи."""
        return self._config.stream

    @property
    def model_name(self) -> str:
        """Возвращает название используемой модели."""
        return self._config.model

    def reset(self) -> None:
        """Сброс контекста беседы."""
        self._previous_response_id = None
        self._turn_count = 0

    def send_message(self, user_text: str, output_stream: TextIO | None = None) -> str:
        """Отправка сообщения пользователя и получение ответа."""
        request: dict[str, Any] = {
            "model": self._config.model,
            "instructions": self._config.system_prompt,
            "input": user_text,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "truncation": "auto",
        }
        # Если есть предыдущий ответ, добавляем его ID для поддержания контекста
        if self._previous_response_id is not None:
            request["previous_response_id"] = self._previous_response_id

        LOGGER.debug(
            "Отправка запроса с моделью=%s previous_response_id=%s",
            self._config.model,
            self._previous_response_id,
        )

        started_at = time.monotonic()
        # Выполнение API-запроса
        response = self._create_response(request, output_stream=output_stream)
        duration = time.monotonic() - started_at

        # Сохранение ID текущего ответа для следующего шага
        self._previous_response_id = response.id
        self._turn_count += 1

        LOGGER.debug(
            "Получен ответ id=%s turn=%d duration=%.2fs",
            response.id,
            self._turn_count,
            duration,
        )

        return extract_response_text(response)

    def _create_response(self, request: dict[str, Any], output_stream: TextIO | None) -> Any:
        """Внутренний метод для создания ответа (с поддержкой потоковой передачи или без нее)."""
        if not self._config.stream:
            return self._client.responses.create(**request)

        # Обработка потокового ответа
        with self._client.responses.stream(**request) as stream:
            for event in stream:
                if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                    delta = getattr(event, "delta", "")
                    if delta and output_stream is not None:
                        # Печать фрагмента текста в выходной поток
                        print(delta, end="", file=output_stream, flush=True)

            if output_stream is not None:
                print(file=output_stream, flush=True)

            return stream.get_final_response()


def load_config(argv: Sequence[str]) -> AppConfig:
    """Загрузка конфигурации из переменных окружения и аргументов командной строки."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Готовый к использованию терминальный чат-клиент для OpenAI Responses API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Определение аргументов командной строки
    parser.add_argument(
        "--message",
        help="Отправить одно сообщение и выйти вместо запуска интерактивной сессии.",
    )
    parser.add_argument(
        "--model",
        default=_get_env_str("OPENAI_MODEL", DEFAULT_MODEL),
        help="ID модели для генерации.",
    )
    parser.add_argument(
        "--system-prompt",
        default=_get_env_str("OPENAI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="Инструкции, отправляемые модели при каждом запросе.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_get_env_float("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE),
        help="Температура выборки (от 0 до 2).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=_get_env_int("OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
        help="Максимальное количество токенов в ответе.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_get_env_float("OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        help="Тайм-аут запроса в секундах.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=_get_env_int("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        help="Количество повторных попыток при временных ошибках API.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=_get_env_bool("OPENAI_STREAM", False),
        help="Включить потоковую передачу токенов.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить отладочное логирование.",
    )

    args = parser.parse_args(argv)

    # Проверка обязательных параметров и валидация значений
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY отсутствует. Установите его в окружении или в файле .env."
        )
    if not args.model.strip():
        raise ConfigurationError("Название модели не может быть пустым.")
    if not args.system_prompt.strip():
        raise ConfigurationError("Системный промпт не может быть пустым.")
    if not 0 <= args.temperature <= 2:
        raise ConfigurationError("Температура должна быть от 0 до 2.")
    if args.max_output_tokens <= 0:
        raise ConfigurationError("max-output-tokens должен быть больше 0.")
    if args.timeout <= 0:
        raise ConfigurationError("timeout должен быть больше 0.")
    if args.max_retries < 0:
        raise ConfigurationError("max-retries не может быть отрицательным.")

    message = args.message.strip() if args.message is not None else None
    if args.message is not None and not message:
        raise ConfigurationError("--message не может быть пустым.")

    return AppConfig(
        api_key=api_key,
        model=args.model.strip(),
        system_prompt=args.system_prompt.strip(),
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        stream=args.stream,
        debug=args.debug,
        message=message,
    )


def run_interactive(session: ChatSession) -> int:
    """Запуск интерактивного режима чата в терминале."""
    print(f"Чат запущен с моделью '{session.model_name}'.")
    print("Команды: /help, /reset, /exit")

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nСессия завершена.")
            return 0

        if not user_text:
            continue

        normalized = user_text.lower()
        # Обработка команд завершения и управления сессией
        if normalized in {"exit", "quit", "/exit", "/quit"}:
            print("Сессия завершена.")
            return 0
        if normalized == "/reset":
            session.reset()
            print("Беседа сброшена.")
            continue
        if normalized == "/help":
            print("Команды: /help, /reset, /exit")
            continue

        try:
            if session.stream_enabled:
                print("Ассистент: ", end="", flush=True)
                session.send_message(user_text, output_stream=sys.stdout)
            else:
                answer = session.send_message(user_text)
                print(f"Ассистент: {answer}")
        except KeyboardInterrupt:
            print("\nЗапрос прерван. Состояние беседы не изменилось.", file=sys.stderr)
        except OpenAIError as exc:
            print(f"Ошибка: {format_openai_error(exc)}", file=sys.stderr)
        except RuntimeError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)


def run_one_shot(session: ChatSession, message: str) -> int:
    """Выполнение одного запроса к модели."""
    if session.stream_enabled:
        session.send_message(message, output_stream=sys.stdout)
        return 0

    answer = session.send_message(message)
    print(answer)
    return 0


def extract_response_text(response: Any) -> str:
    """Извлечение текстового содержимого из объекта ответа OpenAI API."""
    text = getattr(response, "output_text", None)
    if text:
        return text.rstrip()

    parts: list[str] = []
    # Обход структуры ответа для сбора фрагментов текста
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", None)
            if content_type == "output_text":
                value = getattr(content, "text", "")
            elif content_type == "refusal":
                value = getattr(content, "refusal", "")
            else:
                value = ""

            if value:
                parts.append(value)

    if parts:
        return "".join(parts).rstrip()

    # Обработка потенциальных ошибок в ответе
    response_error = getattr(response, "error", None)
    if response_error is not None:
        code = getattr(response_error, "code", "unknown")
        message = getattr(response_error, "message", "Unknown response error.")
        raise RuntimeError(f"Ошибка ответа модели ({code}): {message}")

    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is not None:
        reason = getattr(incomplete_details, "reason", "unknown")
        raise RuntimeError(f"Модель не вернула текстовый результат. Причина незавершенности: {reason}")

    raise RuntimeError("Модель не вернула текстовый результат.")


def format_openai_error(exc: OpenAIError) -> str:
    """Форматирование исключений OpenAI SDK в понятные человеку сообщения об ошибках."""
    if isinstance(exc, AuthenticationError):
        return "Ошибка аутентификации. Проверьте OPENAI_API_KEY."
    if isinstance(exc, RateLimitError):
        return "Превышен лимит запросов. Уменьшите частоту запросов или попробуйте позже."
    if isinstance(exc, APITimeoutError):
        return "Время ожидания запроса истекло до получения ответа от API."
    if isinstance(exc, APIConnectionError):
        return "Не удалось связаться с OpenAI API. Проверьте настройки сети, прокси или DNS."
    if isinstance(exc, APIStatusError):
        return f"OpenAI API вернул HTTP {exc.status_code}: {exc}"
    return str(exc)


def setup_logging(debug: bool) -> None:
    """Настройка параметров логирования."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_env_str(name: str, default: str) -> str:
    """Получение строковой переменной окружения с поддержкой значения по умолчанию."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_env_int(name: str, default: int) -> int:
    """Получение целочисленной переменной окружения."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} должно быть целым числом.") from exc


def _get_env_float(name: str, default: float) -> float:
    """Получение переменной окружения типа float."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} должно быть числом с плавающей точкой.") from exc


def _get_env_bool(name: str, default: bool) -> bool:
    """Получение логической переменной окружения из строки."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ConfigurationError(f"{name} должно быть логическим значением (true/false).")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа в приложение."""
    try:
        config = load_config(argv or sys.argv[1:])
        setup_logging(config.debug)
        session = ChatSession(config)
        # Выбор режима работы: одно сообщение или интерактивный чат
        if config.message:
            return run_one_shot(session, config.message)
        return run_interactive(session)
    except ConfigurationError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2
    except OpenAIError as exc:
        print(f"Ошибка: {format_openai_error(exc)}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        LOGGER.exception("Необработанная ошибка")
        print(f"Непредвиденная ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
