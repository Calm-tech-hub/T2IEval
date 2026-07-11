import concurrent.futures
from dataclasses import dataclass
from typing import Any, cast

import openai


@dataclass(slots=True)
class OpenAIRequest:
    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 2000


@dataclass(slots=True)
class OpenAIResponse:
    content: str
    error: str | None = None


class OpenAIClient:
    def __init__(
        self,
        api_base: str | None,
        api_key: str,
        max_workers: int,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.max_workers = max_workers

    def _request_one(self, request: OpenAIRequest) -> OpenAIResponse:
        try:
            response = self.client.chat.completions.create(
                model=request.model,
                messages=cast(Any, request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            return OpenAIResponse(
                content=response.choices[0].message.content or "",
            )
        except Exception as error:
            return OpenAIResponse(content="", error=str(error))

    def __call__(self, requests: list[OpenAIRequest]) -> list[OpenAIResponse]:
        if not requests:
            return []

        results: list[OpenAIResponse | None] = [None] * len(requests)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(self._request_one, request): index
                for index, request in enumerate(requests)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()

        return [result for result in results if result is not None]
