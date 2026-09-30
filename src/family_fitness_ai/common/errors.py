"""실패는 한 모양이다 — {"error": {"code", "message"}}.

성공은 payload 를 그대로 낸다. 봉투를 씌우지 않는다.
"""

from __future__ import annotations


class ApiError(Exception):
    """호출자에게 그대로 나가는 오류. 코드 표는 docs/인터페이스-명세.md 에 있다."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def bad_request(message: str) -> ApiError:
    return ApiError(400, "BAD_REQUEST", message)


def item_not_allowed(codes: list[str]) -> ApiError:
    return ApiError(400, "ITEM_NOT_ALLOWED", f"받지 않는 측정 항목입니다: {', '.join(codes)}")


def run_not_found(run_id: str) -> ApiError:
    return ApiError(404, "RUN_NOT_FOUND", f"실행을 찾을 수 없습니다: {run_id}")


def run_in_progress() -> ApiError:
    return ApiError(409, "RUN_IN_PROGRESS", "실행 중인 코치 실행이 있습니다")


def temporarily_unavailable(message: str = "잠시 후 다시 시도해 주세요") -> ApiError:
    return ApiError(503, "TEMPORARILY_UNAVAILABLE", message)
