"""`POST /v1/fitness/assessment` (docs/dev/AI-4 §5)."""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from family_fitness_ai.api.app import app
from family_fitness_ai.common.copy import contains_forbidden

PATH = "/v1/fitness/assessment"
BASE = {"profile_ref": "p_c7a91f", "age": 11, "age_unit": "세", "sex": "F"}
MEASURED = {"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30, "018": 18.7}


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not pathlib.Path("data/release/age_band_score_summary.csv").exists():
        pytest.skip("산출물이 없다 — make distribution 을 먼저 돌린다")
    return TestClient(app)


def post(client: TestClient, **overrides: object) -> dict:
    body = {**BASE, **overrides}
    response = client.post(PATH, json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ── 노출 경계 ────────────────────────────────────────────────────────


def test_child_scope_에_점수와_백분위와_체중이_없다(client: TestClient) -> None:
    """아이 화면이 실수로 노출할 값이 애초에 페이로드에 없다 (docs/03 §2.6)."""
    child = post(client, measurements=MEASURED, height_cm=148.0, weight_kg=41.0)["child_scope"]
    flat = repr(child)
    for banned in ("score", "percentile", "weight", "band", "grade"):
        assert banned not in flat
    assert set(child) == {"focus_one"}
    assert set(child["focus_one"]) == {"factor", "copy"}


def test_요인_하나만_제안한다(client: TestClient) -> None:
    """요인 간 비교·순위를 담지 않는다 (docs/03 §3.3)."""
    child = post(client, measurements=MEASURED)["child_scope"]
    assert child["focus_one"]["factor"] == "순발력"  # 점수 최저 (docs/02 §5.6)


def test_문구에_금지_어휘가_없다(client: TestClient) -> None:
    """ "부족"·"미달"·"하위"를 쓰지 않는다 (docs/03 §2.6)."""
    body = post(client, measurements=MEASURED)
    for text in (
        body["child_scope"]["focus_one"]["copy"],
        body["parent_scope"]["copy"]["strength"],
        body["parent_scope"]["copy"]["focus"],
        body["disclaimer"],
    ):
        assert contains_forbidden(text) == []


# ── 입력 수준 ────────────────────────────────────────────────────────


def test_측정값이_없어도_200_이고_factors_는_빈_배열이다(client: TestClient) -> None:
    body = post(client)
    assert body["input_level"] == "L0"
    assert body["parent_scope"]["factors"] == []
    assert body["parent_scope"]["grade"] is None
    assert body["child_scope"]["focus_one"] is not None  # 연령대 고정 제안


def test_신장_체중만_있으면_L1_이다(client: TestClient) -> None:
    body = post(client, height_cm=148.0, weight_kg=41.0)
    assert body["input_level"] == "L1"
    assert body["parent_scope"]["factors"] == []


def test_그_구간의_기준항목이_아니면_L2_가_아니다(client: TestClient) -> None:
    """`measurements` 가 왔다고 L2 가 아니다 (docs/dev/AI-4 §2)."""
    body = post(client, measurements={"041": 0.5})  # 성인 항목이다
    assert body["input_level"] == "L0"
    assert body["parent_scope"]["factors"] == []


def test_peer_distribution_은_L0_에서도_나간다(client: TestClient) -> None:
    peers = post(client)["parent_scope"]["peer_distribution"]
    assert [p["grade"] for p in peers] == ["1등급", "2등급", "3등급", "참가"]
    assert abs(sum(p["ratio"] for p in peers) - 1.0) < 0.01


# ── 값 ──────────────────────────────────────────────────────────────


def test_실측_예시가_그대로_나온다(client: TestClient) -> None:
    """docs/02 §5.6 의 회귀 검사. 산출물이 다시 만들어졌을 때 여기서 드러난다."""
    body = post(client, measurements=MEASURED)
    factors = {f["factor"]: f for f in body["parent_scope"]["factors"]}
    assert factors["근력"]["score"] == 67.6
    assert factors["유연성"]["score"] == 45.3
    assert factors["심폐지구력"]["score"] == 85.9
    assert factors["순발력"]["score"] == 41.2


def test_item_label_은_연령대의_실제_시험명이다(client: TestClient) -> None:
    """같은 코드 020 이 유소년은 15m, 청소년은 20m 다 (docs/03 §3.5)."""
    유소년 = post(client, measurements={"020": 70})["parent_scope"]["factors"][0]
    청소년 = post(client, age=15, measurements={"020": 70})["parent_scope"]["factors"][0]
    assert 유소년["item_label"] == "15m왕복오래달리기"
    assert 청소년["item_label"] == "20m왕복오래달리기"
    assert 유소년["item_name"] == 청소년["item_name"]  # 데이터 항목명은 같다


def test_신체조성은_요인에_나오지_않는다(client: TestClient) -> None:
    """점수화하지 않는다 (docs/02 §5.4). 등급 판정에만 쓴다."""
    body = post(client, measurements=MEASURED)
    assert "018" not in [f["item_code"] for f in body["parent_scope"]["factors"]]
    assert body["parent_scope"]["grade"] == "3등급"


def test_판정_항목이_빠지면_등급이_null_이다(client: TestClient) -> None:
    """재지 않은 것과 기준을 못 넘은 것은 다르다 (docs/02 §5.2)."""
    body = post(client, measurements={"028": 41.3, "012": 4.0, "020": 70, "022": 133})
    assert body["input_level"] == "L2"
    assert body["parent_scope"]["grade"] is None


# ── 연령대 ──────────────────────────────────────────────────────────


def test_만_7에서_10세는_factors_가_빈_배열이다(client: TestClient) -> None:
    """측정 0건 구간이다 (docs/02 §2.2). 오류가 아니다."""
    body = post(client, age=8, measurements={"028": 41.3})
    assert body["age_group"] == "유소년"  # 가장 가까운 구간 (docs/02 §6 ④)
    assert body["parent_scope"]["factors"] == []
    assert body["parent_scope"]["grade"] is None


def test_어르신도_200_이다(client: TestClient) -> None:
    """기준항목이 미정이라 점수가 없을 뿐이다 (docs/03 §2.4)."""
    body = post(client, age=70)
    assert body["age_group"] == "어르신"
    assert body["parent_scope"]["factors"] == []


def test_유아기는_개월로_조회된다(client: TestClient) -> None:
    """세로 환산은 내림이다 — 반올림하면 공백을 거짓으로 메운다 (docs/02 §2.4)."""
    body = post(client, age=60, age_unit="개월")
    assert body["age_group"] == "유아기"


# ── 오류 ────────────────────────────────────────────────────────────


def test_혈압을_보내면_400_이다(client: TestClient) -> None:
    """AI는 이 항목을 다루지 않는다 (docs/03 §9)."""
    response = client.post(PATH, json={**BASE, "measurements": {"005": 80, "028": 41.3}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ITEM_NOT_ALLOWED"


def test_필수_필드가_빠지면_400_BAD_REQUEST_다(client: TestClient) -> None:
    response = client.post(PATH, json={"profile_ref": "p_1"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_연령대를_정할_수_없으면_400_이다(client: TestClient) -> None:
    response = client.post(PATH, json={**BASE, "age": 2})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_신장_범위를_벗어나면_400_이다(client: TestClient) -> None:
    response = client.post(PATH, json={**BASE, "height_cm": 300.0})
    assert response.status_code == 400


def test_저표본_칸은_점수를_null_로_내린다(client: TestClient) -> None:
    """`n < 30` 이면 ECDF를 믿지 않는다 (docs/02 §6 ③ · docs/dev/AI-4 §4).

    청소년 여 13세 `035` 는 n=24 다. 같은 응답의 다른 요인은 그대로 점수가 나온다 —
    칸 단위로 내리지 응답 전체를 버리지 않는다.
    """
    body = post(client, age=13, measurements={"035": 40.0, "028": 40.0})
    assert body["low_sample"] is True
    factors = {f["factor"]: f for f in body["parent_scope"]["factors"]}
    저표본 = factors["심폐지구력"]
    assert 저표본["n"] < 30
    assert 저표본["score"] is None
    assert 저표본["percentile"] is None
    assert 저표본["band"] is None
    assert 저표본["value"] == 40.0  # 측정값 자체는 남는다
    assert factors["근력"]["score"] is not None
