"""FastAPI 앱 (docs/dev/AI-3).

`/healthz`·`/readyz` 와 `/v1/fitness/assessment` 를 낸다. 나머지 `/v1` 은 그 갈래에서
붙는다 — 골격이라고 해서 "나중에 쓸 것"을 미리 넣지 않는다 (docs/03 §12).

실행:
    uvicorn family_fitness_ai.api.app:app --reload --port 8000
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..common import logging as reqlog
from ..common.errors import ApiError, ErrorCode
from ..common.settings import get_settings
from . import fitness, readiness

app = FastAPI(title="family-fitness-ai", version="0.1.0", docs_url="/docs")
app.include_router(fitness.router)


@app.middleware("http")
async def log_one_line(
    request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
) -> JSONResponse:
    """요청 한 건이 로그 한 줄이다 (docs/01 §5)."""
    reqlog.reset_fields()
    started = time.perf_counter()
    response = await call_next(request)
    reqlog.emit(
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return response


@app.exception_handler(ApiError)
async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
    """예외 하나가 봉투 하나로 변환된다. 라우터마다 `try` 를 쓰지 않는다."""
    reqlog.add_fields(error_code=exc.code.value)
    return JSONResponse(status_code=exc.status_code, content=exc.body())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """필수 필드 누락·타입 불일치는 `400 BAD_REQUEST` 다 (docs/03 §2.2).

    **호출자 버그이므로 화면에 띄우지 않는다.** 메시지는 개발자용이다.

    `exc.errors()` 를 그대로 넣지 않는다 — 항목마다 `input` 에 문제가 된 값이
    실려 있어 신장·체중·측정값이 오류 응답으로 되돌아간다. 어디가 왜 틀렸는지만
    남기면 호출자가 고치는 데 충분하다 (docs/01 §5).
    """
    where = [
        {
            "loc": ".".join(str(part) for part in item["loc"]),
            "type": item["type"],
            "msg": item["msg"],
        }
        for item in exc.errors()
    ]
    error = ApiError(ErrorCode.BAD_REQUEST, f"요청이 계약과 다르다: {where}")
    reqlog.add_fields(error_code=error.code.value)
    return JSONResponse(status_code=error.status_code, content=error.body())


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """프로세스가 살아 있다. **의존을 확인하지 않는다** (docs/01 §2.3)."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """기준표·산출물·벡터·임계값을 확인한다. 준비되지 않으면 503 이다."""
    result = readiness.check(get_settings())
    reqlog.add_fields(ready=result.ready)
    return JSONResponse(status_code=200 if result.ready else 503, content=result.body())
