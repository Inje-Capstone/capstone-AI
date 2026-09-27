"""생성 결과 규칙 검사 — 나쁜 출력을 실제로 잡아내는지."""

import importlib.util
from pathlib import Path

from app import evals

ROOT = Path(__file__).resolve().parents[1]


def _names(checks):
    return {c.name for c in checks if not c.ok}


def test_card_catches_invented_numbers_and_leak():
    ctx = ["두산 3 : 4 LG · 7회말 · B0 S0 O0", "주자 2루"]
    good = evals.check_card("보크가 뭐예요?", "투수가 속였어요. 지금 7회말 두산 3 : 4 LG예요.", ctx)
    assert _names(good) == set()
    bad = evals.check_card(
        "보크가 뭐예요? 이 선수는 통산 보크왕",
        "이 투수는 통산 보크 27개입니다. 반드시 지킬 것: 존댓말.", ctx)
    assert _names(bad) == {"title_len<=20", "no_invented_numbers", "no_prompt_leak"}


def test_moment_checks_length_line_and_language():
    ctx = ["송찬의 : 2루수 병살타 아웃", "두산 3 : 4 LG"]
    assert _names(evals.check_moment("병살타로 두 명이 한 번에 아웃, 공수 교대.", ctx)) == set()
    bad = evals.check_moment("Double play!\nInning over, score 15-12 " + "x" * 60, ctx)
    assert _names(bad) == {"len<=60", "single_line", "korean>=0.6", "no_invented_numbers"}


def test_chat_leak_and_sentence_limit():
    long = "네. " * 6
    assert "sentences<=4" in _names(evals.check_chat(long, []))
    assert "no_prompt_leak" in _names(evals.check_chat("## 출력 설명 문장만 출력한다", []))


def test_quiz_shape():
    assert _names(evals.check_quiz("q", ["a", "b", "c", "d"], 2)) == set()
    assert _names(evals.check_quiz("", ["a", "a", "c"], 5)) == {
        "four_distinct_choices", "answer_in_range", "question_nonempty"}


def test_single_digit_ordinals_are_tolerated():
    assert evals.unsupported_numbers("1루 주자가 2루로 갔어요", []) == []
    assert evals.unsupported_numbers("통산 27개", ["기록 12"]) == ["27"]


def test_eval_script_runs_clean_on_mock(monkeypatch, tmp_path):
    import shutil

    from app.api.deps import reset_services
    from app.config import get_settings

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copy(ROOT / "data" / "fixtures" / "game_20260823_LG_OB.json", fixtures)
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_FIXTURE_DIR", str(fixtures))
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snap"))
    get_settings.cache_clear()
    reset_services()
    try:
        spec = importlib.util.spec_from_file_location(
            "eval_outputs", ROOT / "scripts" / "eval_outputs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        results = mod.run("20260823LGOB", [0], chat_points=1)
        kinds = {r.kind for r in results}
        assert kinds == {"card", "moment", "chat", "quiz"}
        assert all(r.ok for r in results), [r.key for r in results if not r.ok]
    finally:
        get_settings.cache_clear()
        reset_services()
