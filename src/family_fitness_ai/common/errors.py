"""오류는 한 형태다 (docs/03 §2.2).

    {"error": {"code": "...", "message": "..."}}

성공은 payload를 그대로 돌려준다. 봉투를 씌우지 않는다.

**거부(`refused: true`)는 이 경로를 타지 않는다.** 자료에 없어 답하지 못하는 것은
정상 동작이고 HTTP 200 이다. 예외로 만들면 거부가 오류 로그에 섞여 거부율을
측정할 수 없다 (docs/01 §5).
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    BAD_REQUEST = "BAD_REQUEST"
    ITEM_NOT_ALLOWED = "ITEM_NOT_ALLOWED"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_IN_PROGRESS = "RUN_IN_PROGRESS"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"


# docs/03 §2.2 의 표. 코드 하나에 상태 하나다.
STATUS_OF: dict[ErrorCode, int] = {
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.ITEM_NOT_ALLOWED: 400,
    ErrorCode.RUN_NOT_FOUND: 404,
    ErrorCode.RUN_IN_PROGRESS: 409,
    ErrorCode.TEMPORARILY_UNAVAILABLE: 503,
}


class ApiError(Exception):
    """계약의 오류 하나. `message` 는 개발자용이고 화면에 그대로 내지 않는다."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status_code(self) -> int:
        return STATUS_OF[self.code]

    def body(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code.value, "message": self.message}}
