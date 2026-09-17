from family_fitness_ai.coach import verify

CITATIONS = [{"index": 1, "label": "근거", "chunk_id": "prescription:x", "url": None}]


def test_answer_needs_a_citation_mark():
    assert verify.check_answer("근거 없이 답한다.", CITATIONS) == ["본문에 근거 번호가 없다"]


def test_answer_marks_must_stay_in_range():
    problems = verify.check_answer("이렇습니다 [2].", CITATIONS)
    assert "인용 범위 밖 [2]" in problems


def test_banned_words_block_the_answer():
    problems = verify.check_answer("유연성이 부족합니다 [1].", CITATIONS)
    assert any("금지 어휘" in problem for problem in problems)


def test_proposal_without_citations_fails():
    assert "인용 0건" in verify.check_proposal({"missions": [], "citations": []}, set())
