"""API 호출 공통 처리: 재시도와 인증키 가리기.

오류 메시지에 URL 이 그대로 담기면 인증키가 로그에 남기 때문에, 예외를 다시 만들어 던진다.
"""
from __future__ import annotations

import time

import requests

RETRIES = 3
BACKOFF = 2.0
SECRET_PARAMS = ("crtfc_key", "serviceKey", "api_key")


class ApiError(RuntimeError):
    pass


def _mask(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<KEY>")
    return text


def get(url: str, params: dict | None = None, timeout: int = 30, secrets: list[str] | None = None) -> requests.Response:
    """실패 시 재시도하고, 오류 메시지에서 인증키를 가린다. URL 자체에 키가 있으면 secrets 로 넘긴다."""
    params = params or {}
    hidden = list(secrets or []) + [str(v) for k, v in params.items() if k in SECRET_PARAMS]
    last = None
    for attempt in range(RETRIES):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            last = e
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF * (attempt + 1))
    raise ApiError(f"{_mask(url, hidden)} 호출 실패 ({RETRIES}회 재시도): {_mask(str(last), hidden)}") from None
