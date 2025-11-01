"""LLM provider configuration and initialization."""

from typing import Optional

from langchain_openai import ChatOpenAI

from .settings import settings


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Initialize and return a LangChain LLM instance.

    Args:
        model: Model name to use (defaults to settings.llm_model)
        temperature: Temperature setting (defaults to settings.llm_temperature)
        max_tokens: Maximum tokens (defaults to settings.llm_max_tokens)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        api_key=settings.openai_api_key,
    )

