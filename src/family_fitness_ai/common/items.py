"""측정 항목 표.

docs/인터페이스-명세.md 「측정 항목 코드」와 짝이다. 이 파일이 항목의 유일한
출처다 — 이름·단위·요인·방향을 다른 곳에 다시 적지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

AGE_GROUPS = ("유아기", "유소년", "청소년", "성인", "어르신")

FACTORS = (
    "심폐지구력",
    "근력",
    "근지구력",
    "유연성",
    "민첩성",
    "순발력",
    "협응력",
    "평형성",
)

GRADES = ("1등급", "2등급", "3등급", "참가")


@dataclass(frozen=True)
class Item:
    code: str
    name: str
    unit: str
    factor: str
    #: 작을수록 우수한 항목. 백분위를 뒤집어 센다.
    lower_is_better: bool = False


ITEMS: dict[str, Item] = {
    it.code: it
    for it in (
        Item("009", "윗몸말아올리기", "회", "근지구력"),
        Item("010", "반복점프", "회", "근지구력"),
        Item("012", "앉아윗몸앞으로굽히기", "cm", "유연성"),
        Item("013", "일리노이", "초", "민첩성", lower_is_better=True),
        Item("014", "체공시간", "초", "순발력"),
        # 협응력은 연령대마다 다른 시험을 쓴다. 017 은 청소년의 계산결과값(초),
        # 044 는 유소년의 벽패스 성공 횟수(회)다. 원자료에서 둘은 겹치지 않는다.
        Item("017", "눈-손협응력", "초", "협응력", lower_is_better=True),
        Item("019", "교차윗몸일으키기", "회", "근지구력"),
        Item("020", "왕복오래달리기", "회", "심폐지구력"),
        Item("021", "10m 4회 왕복달리기", "초", "민첩성", lower_is_better=True),
        Item("022", "제자리멀리뛰기", "cm", "순발력"),
        Item("028", "상대악력", "%", "근력"),
        Item("035", "VO₂max(트레드밀)", "ml/kg/min", "심폐지구력"),
        Item("037", "VO₂max(스텝)", "ml/kg/min", "심폐지구력"),
        Item("040", "반응시간", "초", "민첩성", lower_is_better=True),
        Item("041", "성인체공시간", "초", "순발력"),
        Item("043", "반복옆뛰기", "회", "민첩성"),
        # 컬럼정의서는 044 를 「(초)」로 적어 두었지만 값은 0~40 의 정수고, 기준표의
        # 유소년 「눈-손 협응력 검사 (회)」와 단위가 맞는다 — 횟수로 읽는다.
        Item("044", "눈-손협응력(벽패스)", "회", "협응력"),
        Item("050", "5m 4회 왕복달리기", "초", "민첩성", lower_is_better=True),
        Item("051", "3×3 버튼누르기", "초", "협응력", lower_is_better=True),
    )
}

#: 신체조성. 체력 요인이 아니라 등급의 별도 관문이다.
#:
#: 클수록 좋다도 작을수록 좋다도 아니고 적정 구간이 있을 뿐이라, 백분위를 점수로
#: 바꾸지 않는다 — FitnessFactor 여덟 개 중 여기 해당하는 것도 없다. 대신 3등급
#: 판정에서 「민첩성·순발력·협응력」 자리를 대신 맡는다. 기준표 3등급 줄을 보면
#: 운동 체력 칸이 비고 BMI·체지방률·WHtR 칸이 차 있다.
BODY_ITEMS: dict[str, tuple[str, str]] = {
    "003": ("체지방률", "%"),
    "004": ("허리둘레", "cm"),
    "018": ("BMI", "kg/㎡"),
    "042": ("허리둘레-신장비", "WHtR"),
}

#: 받지 않는다 — 이완기·수축기 혈압. 400 ITEM_NOT_ALLOWED 로 돌려보낸다.
BLOOD_PRESSURE = ("005", "006")

#: 연령대별 기준항목 — 요인 점수를 내는 항목이다. 기준표 1·2등급 줄이 보는 것과
#: 같다. 어르신은 시험 자체가 다른 표(2분 제자리 걷기·8자보행…)를 쓰는데 이번
#: 범위 밖이라, 성인과 겹치는 둘만 둔다. 등급은 내지 않는다.
AGE_GROUP_ITEMS: dict[str, tuple[str, ...]] = {
    "유아기": ("020", "028", "009", "012", "050", "022", "051"),
    "유소년": ("020", "028", "009", "012", "043", "022", "044"),
    "청소년": ("020", "035", "037", "028", "009", "010", "012", "013", "014", "017"),
    "성인": ("020", "035", "037", "028", "019", "012", "021", "040", "022", "041"),
    "어르신": ("012", "028"),
}

#: 020 은 연령대마다 거리가 다르다. 화면에 나가는 이름은 이쪽이다.
_ITEM_LABELS: dict[tuple[str, str], str] = {
    ("020", "유아기"): "10m왕복오래달리기",
    ("020", "유소년"): "15m왕복오래달리기",
    ("020", "청소년"): "20m왕복오래달리기",
    ("020", "성인"): "20m왕복오래달리기",
    ("020", "어르신"): "20m왕복오래달리기",
}


def item_label(code: str, age_group: str) -> str:
    """그 연령대의 실제 시험명."""
    label = _ITEM_LABELS.get((code, age_group))
    if label:
        return label
    item = ITEMS.get(code)
    if item:
        return item.name
    body = BODY_ITEMS.get(code)
    return body[0] if body else code


def age_group_of(age: int, age_unit: str) -> str:
    """나이를 연령대로 바꾼다. 유아기만 개월로 들어온다."""
    years = age // 12 if age_unit == "개월" else age
    if years < 7:
        return "유아기"
    if years < 13:
        return "유소년"
    if years < 19:
        return "청소년"
    if years < 65:
        return "성인"
    return "어르신"


def scored_items(age_group: str, measurements: dict[str, float]) -> list[str]:
    """그 연령대에서 요인 점수로 셀 수 있는 항목 코드. 들어온 순서를 지킨다."""
    allowed = AGE_GROUP_ITEMS.get(age_group, ())
    return [code for code in measurements if code in allowed]
