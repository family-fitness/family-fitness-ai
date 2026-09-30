"""클립 목록과 한 회분 짜기.

영상 하나에 운동이 여럿이라 영상을 통째로 내보내면 아이가 10분을 기다린다.
클립(시작·끝이 있는 한 동작)으로 끊어 두고, 필요한 분량만큼 골라 **준비 → 본 →
정리** 순서로 쌓는다. 클립 하나는 대개 1분 안쪽이라, 15분 한 회는 여러 클립이
모여 만들어진다.

클립은 두 곳에서 온다. 유튜브 영상을 화면 글자로 끊은 것(video_clips.csv)과,
처음부터 한 편이 한 동작인 공단 영상(kspo_videos.csv)이다. `clips()` 가 한 목록으로
합치고, 그 뒤로는 어느 쪽인지 가리지 않고 똑같이 고른다.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import lru_cache

from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.index import Chunk, corpus
from family_fitness_ai.video.vocabulary import exercises_in

PHASES = ("준비운동", "본운동", "정리운동")
#: 한 회를 단계에 나누는 몫. 몸을 덥히고, 하고, 푼다.
#: 한 회에 몇 편을 담나 — (준비운동, 본운동, 정리운동).
#:
#: 영상 길이를 다 합쳐 요청 시간을 채우지 않는다. 화면에서 한 편을 여러 세트
#: 반복해 시간을 채우기 때문이다. 시간으로 고르면 30~40초짜리 클립이 스무 편씩
#: 붙어 따라 할 수 없는 목록이 된다 — 가짓수로 고른다.
SESSION_CLIPS: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (10, (1, 3, 1)),  # 10분까지   5편
    (20, (2, 4, 1)),  # 20분까지   7편
    (35, (2, 5, 2)),  # 35분까지   9편
    (999, (3, 6, 3)),  # 그 위     12편
)

#: 한 세트로 삼기 좋은 길이. 같은 순위면 이 근처를 먼저 고른다.
SET_SECONDS = 60

#: 그 연령대가 또래 영상으로 치는 연령대. 첫째가 제 연령대다. 어르신 전용 영상을
#: 따로 만들지 않고 성인 영상(공단 「공통」 포함)을 똑같이 쓴다(팀 결정). 공단 어르신
#: 영상은 싣지 않아서(video.kspo), 성인을 또래로 치지 않으면 65세 이상은 유아기·
#: 유소년 영상까지 섞인 후보를 받고 규칙 편성은 한 편도 못 고른다.
SHARED_AGES: dict[str, tuple[str, ...]] = {"어르신": ("어르신", "성인")}


def ages_for(age_group: str) -> tuple[str, ...]:
    """그 연령대에게 또래로 치는 영상 연령대."""
    return SHARED_AGES.get(age_group, (age_group,))


def clip_counts(minutes: int) -> dict[str, int]:
    """그 시간에 몇 편을 담을지 단계별로."""
    counts = next(
        (counts for limit, counts in SESSION_CLIPS if minutes <= limit), SESSION_CLIPS[-1][1]
    )
    return dict(zip(PHASES, counts, strict=True))


def focus_quota(slots: int) -> int:
    """보호자가 키워 주고 싶은 역량으로 채울 본운동 칸 수. 4분의 3, 올림이다.

    네 칸이면 세 칸이다(사용자 결정). 전에는 처방에 나온 동작이 요인보다 앞서
    네 칸 가운데 두 칸만 그 역량이었다.
    """
    return -(-slots * 3 // 4)


@dataclass(frozen=True)
class Clip:
    video_id: str
    name: str
    #: 처방 어휘로 옮긴 이름. 못 옮긴 클립은 빈 문자열이다.
    exercise_name: str
    fitness_factor: str
    phase: str
    start_sec: int
    end_sec: int
    age_group: str
    quiet: bool
    home_ok: bool
    needs_props: bool
    #: youtube · kspo. 고를 때는 가리지 않는다 — 트는 쪽만 다르다.
    source: str = "youtube"
    #: 공단 영상 파일. 유튜브는 video_id 로 튼다.
    url: str = ""
    #: 알맞은 체력수준(1 낮음 ~ 5 높음). 적혀 있지 않으면 누구나다.
    level_lo: int = 1
    level_hi: int = 5
    #: 공단 영상은 코퍼스에 없어 인용 이름을 여기 들고 다닌다.
    citation_label: str = ""

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec

    @property
    def title(self) -> str:
        """화면에 나가는 이름. 처방 어휘가 있으면 그쪽이 또렷하다."""
        return self.exercise_name or self.name

    def as_video(self) -> dict[str, object]:
        url = self.url or (f"https://www.youtube.com/watch?v={self.video_id}&t={self.start_sec}s")
        return {
            "source": self.source,
            "video_id": self.video_id,
            "url": url,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }


@lru_cache
def clips() -> tuple[Clip, ...]:
    release = settings().release_dir
    labels: dict[str, dict[str, str]] = {}
    with (release / "clip_labels.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels[row["name_on_video"]] = row

    videos = {
        chunk.chunk_id.split(":", 1)[1]: chunk
        for chunk in corpus().chunks
        if chunk.source == "video"
    }

    out: list[Clip] = []
    with (release / "video_clips.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = labels.get(row["name_on_video"], {})
            if label.get("is_exercise") != "True":
                continue
            video = videos.get(row["video_id"])
            out.append(
                Clip(
                    video_id=row["video_id"],
                    name=row["name_on_video"],
                    exercise_name=label.get("exercise_name", ""),
                    fitness_factor=label.get("fitness_factor", ""),
                    # 영상이 화면에 띄운 단계를 먼저 믿는다. 같은 스트레칭이 준비운동에도
                    # 정리운동에도 나오는데, 라벨 표는 이름당 한 단계라 그걸 못 담는다.
                    phase=row["phase_on_video"] or label.get("phase") or "본운동",
                    start_sec=int(row["start_sec"]),
                    end_sec=int(row["end_sec"]),
                    age_group=video.age_group if video else "",
                    quiet=label.get("quiet") == "True",
                    home_ok=label.get("home_ok") == "True",
                    needs_props=label.get("needs_props") == "True",
                )
            )
    # 공단 영상을 같은 목록에 붙인다. 여기서부터는 어느 쪽인지 가리지 않고 고른다.
    return tuple(out) + _kspo()


def _kspo() -> tuple[Clip, ...]:
    """data/release/kspo_videos.csv (video.kspo 가 만든다).

    서버는 이 표가 없으면 뜨지 않는다(api.app 의 lifespan). 띄우지 않고 부를 때만
    없을 수 있고, 그때는 유튜브만 쓴다.
    """
    path = settings().release_dir / "kspo_videos.csv"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8", newline="") as fh:
        return tuple(
            Clip(
                video_id=row["video_id"],
                name=row["name_on_video"],
                exercise_name=row["exercise_name"],
                fitness_factor=row["fitness_factor"],
                # 유튜브와 같다 — 영상(여기서는 API)이 준 단계를 먼저 믿는다.
                phase=row["phase_on_video"] or row["phase"] or "본운동",
                start_sec=int(row["start_sec"]),
                end_sec=int(row["end_sec"]),
                age_group=row["age_group"],
                quiet=row["quiet"] == "True",
                home_ok=row["home_ok"] == "True",
                needs_props=row["needs_props"] == "True",
                source="kspo",
                url=row["url"],
                level_lo=int(row["level_lo"]),
                level_hi=int(row["level_hi"]),
                citation_label=row["citation_label"],
            )
            for row in csv.DictReader(fh)
            if row["is_exercise"] == "True"
        )


@lru_cache
def _kspo_citations() -> dict[str, Chunk]:
    """공단 영상의 인용. 코퍼스에 없으니 표에서 만든다 — 없으면 검증이 제안을 버린다."""
    out: dict[str, Chunk] = {}
    for clip in _kspo():
        out.setdefault(
            clip.video_id,
            Chunk(
                chunk_id=f"kspo:{clip.video_id}",
                source="video",
                text=clip.citation_label,
                citation_label=clip.citation_label,
                citation_url=clip.url,
                age_group=clip.age_group,
                factors=(clip.fitness_factor,) if clip.fitness_factor else (),
                grade="",
            ),
        )
    return out


def citation_for(video_id: str) -> Chunk | None:
    return _kspo_citations().get(video_id) or corpus().by_id(f"video:{video_id}")


def level_of(percentile: int | None) -> int | None:
    """백분위 → 공단의 체력수준(1~5). 다섯으로 나눈다. 측정이 없으면 None."""
    if percentile is None:
        return None
    return 1 + min(4, max(0, percentile) // 20)


@dataclass(frozen=True)
class Conditions:
    """보호자가 고른 조건. 고른 것만 좁히고 나머지는 건드리지 않는다."""

    quiet: bool = False
    small_space: bool = False
    no_props: bool = False


def _fits(clip: Clip, conditions: Conditions) -> bool:
    if conditions.quiet and not clip.quiet:
        return False
    if conditions.small_space and not clip.home_ok:
        return False
    return not (conditions.no_props and clip.needs_props)


def _rank(
    clip: Clip, factor: str, prescribed: set[str], level: int | None = None
) -> tuple[int, int]:
    """앞에 설 차례. 처방에 실제로 나온 동작이 먼저다.

    같은 순위면 한 세트로 삼기 좋은 길이를 먼저 고른다. 네 분짜리 한 편을 여러
    세트 반복하라고 낼 수는 없다. 체력수준이 맞지 않는 영상은 뒤로 미룬다 —
    빼지는 않는다. 수준이 적힌 것은 공단 영상뿐이라 유튜브 클립은 늘 맞는다.
    """
    score = 0
    if clip.exercise_name and clip.exercise_name in prescribed:
        score -= 4
    if factor and clip.fitness_factor == factor:
        score -= 2
    if clip.exercise_name:
        score -= 1
    if level is not None and not clip.level_lo <= level <= clip.level_hi:
        score += 3
    return score, abs(clip.duration_sec - SET_SECONDS)


def shuffle_key(clip: Clip, seed: str) -> str:
    """같은 순위끼리의 차례. 시드(편성 시작일)가 바뀌면 바뀌고, 같으면 늘 같다.

    BE 는 하루씩 편성을 부른다. 순위만으로 고르면 같은 아이 · 같은 조건에 날마다 같은
    영상이 나온다. 파이썬 hash() 는 프로세스마다 달라 되짚을 수 없어 쓰지 않는다.
    """
    if not seed:
        return ""
    key = f"{seed}|{clip.video_id}|{clip.start_sec}".encode()
    return hashlib.blake2b(key, digest_size=8).hexdigest()


def _age_rank(
    age_group: str, factor: str, prescribed: set[str], level: int | None, seed: str = ""
) -> Callable[[Clip], tuple[int, int, int, str, int]]:
    """_rank 에 「제 연령대 먼저」를 끼운다. 같은 순위면 제 연령대 라벨이 붙은 영상이
    함께 쓰는 연령대(어르신에게 성인) 영상보다 앞선다.

    그다음은 길이를 30초 단위로 거칠게 보고, 같은 칸끼리는 시드로 섞는다. 한 세트로
    삼기 좋은 길이를 앞세우는 것은 그대로 두면서 날마다 다른 것이 앞에 오게 한다.
    """

    def rank(clip: Clip) -> tuple[int, int, int, str, int]:
        score, distance = _rank(clip, factor, prescribed, level)
        return (
            score,
            int(clip.age_group != age_group),
            distance // 30,
            shuffle_key(clip, seed),
            distance,
        )

    return rank


#: recent 목록에서 한 날로 묶어 볼 id 수. BE 가 부르는 하루 편성(20분) 한 회의 칸 수다.
DAY_IDS = 7


def least_recent_first(
    recent: Collection[str], seed: str = "", per_day: int = DAY_IDS
) -> Callable[[Clip], tuple[int, int, str]]:
    """최근에 받은 영상끼리의 차례. 오래전에 받은 것이 앞에 선다.

    recent 는 BE 가 보내는 차례 그대로다 — 최근 미션부터, 한 미션 안은 칸 차례, 겹치면
    한 번만. 뒤쪽일수록 오래전에 받았다. 예전에는 집합으로만 봐서 최근 영상끼리는
    순위 그대로 섰고, 요인이 맞는 새 후보가 떨어지면 날마다 같은 상위 몇 편이 다시
    나왔다(12세 순발력 본운동이 9일 내내 같은 넷).

    다만 한 미션 안의 칸 차례는 받은 때가 아니다. 여기서 새 영상 → 오래전에 받은
    영상 차례로 고르므로, 같은 날 받은 것 중 앞 칸이 그 전에 더 오래 쉰 영상이다.
    목록 차례를 그대로 믿으면 뒤 칸 — 그 전날에도 받은 영상 — 이 가장 오래된 것처럼
    보여 날마다 다시 뽑혔다(5세 민첩성 본운동 한 편이 13일 잇달아). 그래서 per_day
    (한 회 칸 수)개씩을 한 날로 묶고, 그 안에서는 앞 칸을 더 오래된 것으로 본다.
    받은 때가 같으면(한 영상의 여러 클립) seed 로 섞는다.

    차례가 없는 집합(set · frozenset)을 넘기면 모두 같은 때 받은 것으로 본다. 목록에
    없는 영상은 가장 오래된 것으로 친다.
    """
    if isinstance(recent, (set, frozenset)):
        places = dict.fromkeys(recent, 0)
    else:
        places = {}
        for place, video_id in enumerate(recent):
            places.setdefault(video_id, place)
    oldest = len(recent)
    per_day = max(1, per_day)

    def key(clip: Clip) -> tuple[int, int, str]:
        place = places.get(clip.video_id, oldest)
        return -(place // per_day), place, shuffle_key(clip, seed)

    return key


def _defer_recent(
    ranked: list[Clip],
    factor: str,
    recent: Collection[str],
    seed: str = "",
    per_day: int = DAY_IDS,
) -> list[Clip]:
    """최근에 받은 영상을 뒤로 미룬다. 빼지는 않는다.

    단계마다 따로 본다. 대상 요인에 맞는 최근 영상은 같은 요인의 새 후보가 다 선
    뒤에, 요인이 다른 최근 영상은 그 단계 맨 뒤에 선다. 새 후보가 모자라면 그대로
    다시 쓰인다. 단계가 서 있던 자리는 바꾸지 않는다 — 목록을 자르는 곳(pool 의
    limit)에서 단계별 몫이 달라지지 않게.

    최근 영상끼리는 순위 대신 오래전에 받은 것부터 세운다(least_recent_first). 같은
    때 받은 것끼리는 seed 로 섞는다. 유아기 정리운동 후보 일곱은 모두 한 영상에서
    나와 늘 최근이다. 순위대로 두면 점수가 가장 높은 클립이 날마다 맨 앞에
    섰다(14일 중 13일).
    """
    if not recent:
        return ranked
    older_first = least_recent_first(recent, seed, per_day)
    out = list(ranked)
    for phase in {clip.phase for clip in ranked}:
        places = [i for i, clip in enumerate(ranked) if clip.phase == phase]
        group = [ranked[i] for i in places]
        matches = [not factor or clip.fitness_factor == factor for clip in group]
        fresh = [c for c in group if c.video_id not in recent]
        stale_match = sorted(
            (c for c, m in zip(group, matches, strict=True) if m and c.video_id in recent),
            key=older_first,
        )
        stale_other = sorted(
            (c for c, m in zip(group, matches, strict=True) if not m and c.video_id in recent),
            key=older_first,
        )
        last_match = max(
            (i for i, c in enumerate(fresh) if not factor or c.fitness_factor == factor),
            default=-1,
        )
        reordered = fresh[: last_match + 1] + stale_match + fresh[last_match + 1 :] + stale_other
        for place, clip in zip(places, reordered, strict=True):
            out[place] = clip
    return out


def prescribed_names(chunks: list[Chunk]) -> set[str]:
    names: set[str] = set()
    for chunk in chunks:
        names |= {name for name, _ in exercises_in(chunk.text)}
    return names


def routine(
    age_group: str,
    minutes: int,
    *,
    factor: str = "",
    prescribed: set[str] | None = None,
    conditions: Conditions | None = None,
    exclude: set[str] | None = None,
    level: int | None = None,
    seed: str = "",
    recent: Collection[str] = (),
    focus: str = "",
) -> dict[str, list[Clip]]:
    """한 회분 클립을 단계별로 고른다.

    시간이 아니라 **가짓수**를 맞춘다(SESSION_CLIPS). 화면에서 한 편을 여러 세트
    반복하므로 영상 길이의 합이 요청 시간과 같을 필요가 없다.

    같은 동작을 두 번 넣지 않는다. 조건에 걸려 남는 클립이 없으면 그 단계는 빈
    채로 둔다 — 조건을 몰래 풀어 아무거나 채우지 않는다.

    seed(편성 시작일)로 같은 순위끼리 섞고, recent(최근에 받은 영상 id, 최근 것부터)는
    뒤로 미룬다(_defer_recent). 최근 영상끼리는 오래전에 받은 것이 먼저다. 둘 다
    비우면 예전과 같은 차례다.

    한 회 안에서는 같은 영상을 두 번 쓰지 않는다 — 그 단계에 아직 안 쓴 영상의
    후보가 남아 있으면. 모자라면 그때 이미 쓴 영상의 다른 클립을 쓴다. 유아기는
    후보 영상이 몇 편 안 돼 한 회에 같은 영상이 두 번씩 나왔다(14일 중 13일).

    focus(보호자가 키워 주고 싶은 역량)가 있으면 본운동 칸의 4분의 3(focus_quota)을
    먼저 그 역량 클립으로 채운다. 그 역량 후보가 모자라면 최근에 받은 영상 → 앞선
    날에 쓴 동작 차례로 그 역량을 다시 쓰고, 그래도 모자라면 나머지 칸과 함께 지금
    차례대로(다른 요인) 채운다. 한 회 안에서 같은 동작은 여전히 두 번 넣지 않는다.
    준비 · 정리운동은 focus 로 바뀌지 않는다.
    """
    conditions = conditions or Conditions()
    prescribed = prescribed or set()
    ages = ages_for(age_group)
    pool = [clip for clip in clips() if clip.age_group in ages and _fits(clip, conditions)]
    rank = _age_rank(age_group, factor, prescribed, level, seed)

    want = clip_counts(minutes)
    picked: dict[str, list[Clip]] = {phase: [] for phase in PHASES}
    # 날마다 같은 차림을 내지 않으려고, 앞선 날에 쓴 동작을 빼고 고른다.
    used: set[str] = set(exclude or ())
    videos: set[str] = set()
    for phase in PHASES:
        candidates = [clip for clip in pool if clip.phase == phase]
        if candidates and all(clip.title in used for clip in candidates):
            # 뺄 것을 빼고 나니 남는 게 없다. 그 단계만 처음으로 되돌린다.
            used -= {clip.title for clip in candidates}
        ranked = _defer_recent(
            sorted(candidates, key=rank), factor, recent, seed, sum(want.values())
        )
        if focus and phase == "본운동":
            quota = focus_quota(want[phase])
            _pick_focus(picked[phase], ranked, focus, quota, used, recent, videos)
        _take(picked[phase], ranked, want[phase], used, videos)
    return picked


def _take(
    picked: list[Clip], order: list[Clip], want: int, used: set[str], videos: set[str]
) -> None:
    """order 차례로 picked 를 want 편까지 채운다. picked · used · videos 를 고친다.

    쓴 동작(used)은 건너뛴다. 먼저 이 회에 아직 없는 영상만 보고, 모자라면 이미 나온
    영상의 다른 클립도 쓴다.
    """
    for other_videos_only in (True, False):
        for clip in order:
            if len(picked) >= want:
                return
            if clip.title in used or (other_videos_only and clip.video_id in videos):
                continue
            picked.append(clip)
            used.add(clip.title)
            videos.add(clip.video_id)


def _pick_focus(
    picked: list[Clip],
    ranked: list[Clip],
    focus: str,
    quota: int,
    used: set[str],
    recent: Collection[str],
    videos: set[str],
) -> None:
    """본운동 앞 quota 칸을 focus 클립으로 채운다. picked · used · videos 를 고친다.

    차례: 새 동작의 새 영상 → 새 동작의 최근 영상 → 앞선 날에 쓴 동작. 순위(ranked)
    안에서의 차례는 그대로 둔다 — 최근 영상끼리는 _defer_recent 가 이미 오래전에
    받은 것부터 세워 두었다. 오늘 이미 넣은 동작은 다시 넣지 않고, 이 회에 아직 없는
    영상을 먼저 쓴다(_take).
    """
    earlier = set(used)
    mine = [clip for clip in ranked if clip.fitness_factor == focus]
    order = sorted(mine, key=lambda clip: (clip.title in earlier, clip.video_id in recent))
    today = used - earlier
    # 앞선 날에 쓴 동작도 그 역량이면 다시 쓸 수 있다. 오늘 넣은 것만 막는다.
    _take(picked, order, quota, today, videos)
    used |= today


def pool(
    age_group: str,
    *,
    conditions: Conditions | None = None,
    factor: str = "",
    prescribed: set[str] | None = None,
    limit: int = 120,
    level: int | None = None,
    seed: str = "",
    recent: Collection[str] = (),
) -> tuple[list[Clip], str]:
    """고를 만한 클립과, 또래 밖까지 갔는지 알리는 말.

    또래 라벨이 맞는 클립을 앞세우되, 거기서 끊지 않는다. 어르신은 성인 영상도
    또래로 친다(ages_for) — 섞였다고 알리지 않는다. 영상에 연령 라벨이
    붙은 편이 많지 않아 또래만 고집하면 여덟 살에게 아무것도 못 준다. 라벨이
    다른 클립도 뒤에 세워 두고, 그런 것이 섞였으면 그 사실을 말로 돌려준다 —
    쓸지 말지는 화면 저쪽에서 정한다.

    seed · recent 는 routine 과 같다. 최근에 받은 영상은 이름이 같은 다른 클립보다도
    뒤로 가므로, 이름마다 하나만 남길 때(_distinct) 새 영상 쪽이 남는다.
    """
    conditions = conditions or Conditions()
    prescribed = prescribed or set()
    fitting = [clip for clip in clips() if _fits(clip, conditions)]

    ages = ages_for(age_group)
    same = [clip for clip in fitting if clip.age_group in ages]
    other = [clip for clip in fitting if clip.age_group not in ages]
    rank = _age_rank(age_group, factor, prescribed, level, seed)
    same = _defer_recent(sorted(same, key=rank), factor, recent, seed)
    other = _defer_recent(sorted(other, key=rank), factor, recent, seed)
    # 또래가 모자란지는 합치기 전 클립 수로 본다. 합친 뒤의 수로 보면 조건을 켠
    # 유소년·성인이 60 밑으로 내려가, 전에 없던 다른 연령대가 섞이고 알림이 뜬다.
    short = len(same) < limit // 2

    # 같은 동작이 영상마다 따로 잘려 이름이 같은 클립이 많다(「엉덩이 스트레칭」 20개).
    # 그대로 넘기면 LLM 이 다른 것인 줄 알고 같은 날 둘을 고르거나 다른 날 또 고른다.
    # 이름·단계마다 순위가 가장 높은 하나만 남긴다.
    same = _distinct(same)
    picked = same[:limit]
    notice = ""
    if short:
        taken = {_key(clip) for clip in picked}
        room = limit - len(picked)
        picked += [clip for clip in _distinct(other) if _key(clip) not in taken][:room]
        if room and other:
            notice = (
                f"{age_group}에 맞춘 영상이 적어 다른 연령대 영상도 함께 골랐어요. "
                "그대로 쓸지는 보고 정해 주세요."
            )
    return picked, notice


def _key(clip: Clip) -> tuple[str, str]:
    return clip.title, clip.phase


def _distinct(ranked: list[Clip]) -> list[Clip]:
    """이름·단계가 같은 클립 중 앞선 하나만. 순위대로 들어와야 한다."""
    kept: dict[tuple[str, str], Clip] = {}
    for clip in ranked:
        kept.setdefault(_key(clip), clip)
    return list(kept.values())
