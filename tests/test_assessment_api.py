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


def test_측정값이_없으면_요인을_지목하지_않는다(client: TestClient) -> None:
    """연령만으로 요인을 고르면 그 값에는 인용할 것이 없다 (docs/dev/AI-4 §3.1).

    빈 자리를 문구로 채우지 않는다 — `focus_one: null` 이면 호출자는 제안이 없음을
    안다. 제안은 코퍼스가 선 뒤(AI-8) 근거와 함께 붙는다.
    """
    for kwargs in ({}, {"height_cm": 148.0, "weight_kg": 41.0}):
        child = post(client, **kwargs)["child_scope"]
        assert child == {"focus_one": None}


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
    # 점수가 없으면 요인을 지목하지 않는다 — 권할 근거가 없다
    assert body["child_scope"]["focus_one"] is None


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
    # 점수만 보면 백분위·밴드·단위가 어긋나도 통과한다. 한 행을 통째로 못박는다.
    assert factors["심폐지구력"] == {
        "factor": "심폐지구력",
        "item_code": "020",
        "item_name": "왕복오래달리기",
        "item_label": "15m왕복오래달리기",
        "unit": "회",
        "value": 70.0,
        "score": 85.9,
        "percentile": 80,
        "band": "strength",
        "n": 28215,
    }
    # 점수 내림차순이다 — 대상 요인(최저)과 strength(최고) 선택이 이 정렬에 기댄다
    scores = [f["score"] for f in body["parent_scope"]["factors"]]
    assert scores == sorted(scores, reverse=True)


def test_부모_문구가_최고와_최저_요인을_고른다(client: TestClient) -> None:
    """이 엔드포인트가 새로 하는 두 일 중 하나다 (docs/dev/AI-4 §3.3)."""
    copy = post(client, measurements=MEASURED)["parent_scope"]["copy"]
    assert copy["strength"] == "심폐지구력은 잘하고 있는 영역입니다"
    assert copy["focus"] == "순발력은 꾸준히 하고 있는 영역입니다"


def test_요인이_하나면_같은_문장을_두_번_내지_않는다(client: TestClient) -> None:
    """하나만 재고 두 가지를 말할 수는 없다 (docs/dev/AI-4 §3.3)."""
    copy = post(client, measurements={"028": 41.3})["parent_scope"]["copy"]
    assert copy["strength"] != copy["focus"]
    assert "근력" in copy["focus"]


def test_측정값이_없으면_문구도_그렇게_말한다(client: TestClient) -> None:
    copy = post(client)["parent_scope"]["copy"]
    assert copy["strength"] != copy["focus"]
    assert all("넣으면" in line for line in copy.values())


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
    """개월 키로 산출물 칸에 실제로 닿는지 본다 — 연령대 문자열만 보면 조회가
    비어도 통과한다."""
    body = post(client, age=60, age_unit="개월", measurements={"020": 20, "028": 30})
    assert body["age_group"] == "유아기"
    assert body["input_level"] == "L2"
    assert len(body["parent_scope"]["factors"]) == 2
    assert body["parent_scope"]["peer_distribution"] != []


def test_유아기를_세로_보내면_400_이다(client: TestClient) -> None:
    """계약이 개월로 받기로 했다 (docs/03 §3.1). 만 5세는 60~71개월이라 세→개월
    환산이 유일하지 않다 — 조회되지 않을 것을 200 으로 내보내지 않는다."""
    response = client.post(PATH, json={**BASE, "age": 5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


# ── 오류 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", ["005", "006"])
def test_혈압을_보내면_400_이다(client: TestClient, code: str) -> None:
    """AI는 이 항목을 다루지 않는다 (docs/03 §9). 둘 다 리터럴로 적는다 — 상수를
    돌리면 상수에서 빠진 코드를 검사도 함께 놓친다."""
    response = client.post(PATH, json={**BASE, "measurements": {code: 80, "028": 41.3}})
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


@pytest.mark.parametrize(
    ("field", "value"),
    [("height_cm", 300.0), ("height_cm", 10.0), ("weight_kg", 300.0), ("weight_kg", 1.0)],
)
def test_신장_체중_범위를_벗어나면_400_이다(client: TestClient, field: str, value: float) -> None:
    """docs/03 §3.1 — 신장 30~230, 체중 5~250."""
    response = client.post(PATH, json={**BASE, field: value})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_오류_응답이_보낸_값을_되돌려_보내지_않는다(client: TestClient) -> None:
    """검증 오류에 신장·체중·측정값이 실려 나가지 않는다 (docs/01 §5)."""
    response = client.post(PATH, json={**BASE, "height_cm": 300.0})
    message = response.json()["error"]["message"]
    assert "300" not in message
    assert "height_cm" in message  # 어디가 틀렸는지는 남는다


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


def test_저표본_칸만_재도_행과_측정값이_남는다(client: TestClient) -> None:
    """저표본 강등과 L2 판정은 다른 물음이다.

    docs/03 §3.5 는 `n < 30` 이면 점수를 내리라고 했지 행을 지우라 하지 않았다 —
    `value` 와 `n` 은 필수 필드다. 지우면 부모가 넣은 측정값이 응답에서 사라지고,
    `low_sample: true` 인데 그 표본이 어느 칸 것인지 알 수 없는 응답이 나간다.
    """
    body = post(client, age=14, measurements={"035": 40.0})  # 청소년 여 14세 n=22
    assert body["input_level"] == "L2"
    row = body["parent_scope"]["factors"][0]
    assert row["value"] == 40.0
    assert row["n"] < 30
    assert row["score"] is None
    assert body["low_sample"] is True
