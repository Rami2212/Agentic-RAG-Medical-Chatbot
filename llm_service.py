"""LLM service for API interactions."""
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL


def get_llm_response(prompt: str) -> str:
    """
    Get response from the LLM using OpenAI API.

    Args:
        prompt: The prompt to send to the LLM

    Returns:
        The LLM response
    """
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

