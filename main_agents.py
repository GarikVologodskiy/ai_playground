from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, function_tool, ModelSettings, WebSearchTool
from openai import OpenAIError

from dataclasses import dataclass
from typing import Sequence

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SESSION_ID = "igor_chat"
# DEFAULT_TEMPERATURE = 0.7
# DEFAULT_MAX_OUTPUT_TOKENS = 100
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Use web search whenever the user asks for news, recent events, updates, "
    "or any current factual information. "
    "If the user asks for the current time or date, use the get_current_time tool. "
    "Do not guess or approximate time-sensitive information. "
    "Format every research answer as:\n"
    "Summary:\n"
    "- ...\n"
    "Key points:\n"
    "- ...\n"
    "Why it matters:\n"
    "- ...\n"
    "Sources:\n"
    "- ...\n"
)

LOGGER = logging.getLogger("openai_agents_chat")


class ConfigurationError(ValueError):
    """Ошибка конфигурации приложения."""


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str
    system_prompt: str
    debug: bool
    message: str | None
    session_id: str
    temperature: float
    max_output_tokens: int

@function_tool
def get_current_time() -> str:
    """Возвращает текущее локальное время."""
    print("TOOL CALLED: get_current_time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(argv: Sequence[str]) -> AppConfig:
    """Загрузка конфигурации из переменных окружения и аргументов командной строки."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Terminal chat client based on OpenAI Agents SDK.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--message",
        help="Send a single message and exit.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="Model ID.",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.getenv("OPENAI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="Agent instructions.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("OPENAI_SESSION_ID", DEFAULT_SESSION_ID),
        help="Session ID to save conversation history.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7
    )

    parser.add_argument(
        "--max-output-tokens", type=int, default=100
    )

    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY not found. Add it to .env or environment variables."
        )

    model = args.model.strip()
    if not model:
        raise ConfigurationError("Model name cannot be empty.")

    system_prompt = args.system_prompt.strip()
    if not system_prompt:
        raise ConfigurationError("System prompt cannot be empty.")

    session_id = args.session_id.strip()
    if not session_id:
        raise ConfigurationError("session-id cannot be empty.")

    message = args.message.strip() if args.message is not None else None
    if args.message is not None and not message:
        raise ConfigurationError("--message cannot be empty.")

    return AppConfig(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        debug=args.debug,
        message=message,
        session_id=session_id,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens
    )


def setup_logging(debug: bool) -> None:
    """Настройка параметров логирования."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class AgentsChatApp:
    """Класс для управления сессией чата через Agents SDK."""
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._agent = Agent(
            name="Assistant",
            instructions=config.system_prompt,
            model=config.model,
            tools=[get_current_time, WebSearchTool()],
            model_settings=ModelSettings(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens
            )
        )
        self._session = SQLiteSession(config.session_id, "chat_memory.sqlite")

    @property
    def model_name(self) -> str:
        """Возвращает название используемой модели."""
        return self._config.model

    @property
    def session_id(self) -> str:
        """Возвращает ID текущей сессии."""
        return self._config.session_id

    def reset(self) -> None:
        """Сбрасывает сессию чата."""
        # Генерируем новый уникальный суффикс для session_id, чтобы начать "чистую" историю
        self._current_session_id = f"{self._config.session_id}_{uuid.uuid4().hex[:8]}"
        self._session = SQLiteSession(self._current_session_id, "chat_memory.sqlite")
        # В SQLiteSession из Agents SDK для сброса истории обычно требуется либо новый session_id,
        # либо очистка таблицы. Здесь мы используем новый ID для имитации забывания контекста.

    def ask(self, user_text: str) -> str:
        """Отправляет запрос агенту и возвращает ответ."""
        LOGGER.debug("Starting agent with session_id=%s", self.session_id)
        result = Runner.run_sync(
            self._agent,
            user_text,
            session=self._session,
        )
        return str(result.final_output).strip()


def run_interactive(app: AgentsChatApp) -> int:
    """Запуск интерактивного режима чата в терминале."""
    print(f"Chat based on Agents SDK started with model '{app.model_name}'.")
    print("Commands: /help, /reset, /exit")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0

        if not user_text:
            continue

        normalized = user_text.lower()
        if normalized in {"exit", "quit", "/exit", "/quit"}:
            print("Session ended.")
            return 0
        if normalized == "/help":
            print("Commands: /help, /reset, /exit")
            continue
        if normalized == "/reset":
            app.reset()
            print("Conversation context reset.")
            continue

        try:
            answer = app.ask(user_text)
            print(f"\nAssistant: {answer}")
        except OpenAIError as exc:
            print(f"OpenAI error: {exc}", file=sys.stderr)
        except Exception as exc:
            LOGGER.exception("Unhandled error")
            print(f"Unexpected error: {exc}", file=sys.stderr)


def run_one_shot(app: AgentsChatApp, message: str) -> int:
    """Выполнение одного запроса к агенту."""
    answer = app.ask(message)
    print(answer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа в приложение."""
    try:
        config = load_config(argv or sys.argv[1:])
        setup_logging(config.debug)
        app = AgentsChatApp(config)

        if config.message:
            return run_one_shot(app, config.message)

        return run_interactive(app)

    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except OpenAIError as exc:
        print(f"OpenAI error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        LOGGER.exception("Unhandled error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())