from pathlib import Path

from triple_barrier_chart import find_first_barrier_break, render_triple_barrier_chart


def test_render_without_future_candles_marks_open_barrier(tmp_path: Path) -> None:
    candles = [
        {"time": 1, "open": 98, "high": 101, "low": 97, "close": 100},
        {"time": 2, "open": 100, "high": 103, "low": 99, "close": 102},
    ]

    result = render_triple_barrier_chart(
        candles,
        entry=102,
        stop=99,
        take=108,
        output_path=tmp_path / "open_barrier.png",
    )

    assert result.entry_index == 1
    assert result.break_event is None
    assert result.output_path is not None
    assert result.output_path.exists()


def test_render_with_future_candles_marks_first_take_break(tmp_path: Path) -> None:
    candles = [
        {"time": 1, "open": 98, "high": 101, "low": 97, "close": 100},
        {"time": 2, "open": 100, "high": 103, "low": 99, "close": 102},
        {"time": 3, "open": 102, "high": 106, "low": 101, "close": 105},
        {"time": 4, "open": 105, "high": 109, "low": 104, "close": 108},
    ]

    result = render_triple_barrier_chart(
        candles,
        entry=102,
        stop=99,
        take=108,
        entry_index=1,
        output_path=tmp_path / "take_break.png",
    )

    assert result.break_event is not None
    assert result.break_event.kind == "take"
    assert result.break_event.candle_index == 3
    assert result.break_event.price == 108
    assert result.output_path is not None
    assert result.output_path.exists()


def test_find_first_barrier_break_supports_short_direction() -> None:
    candles = [
        {"time": 1, "open": 105, "high": 106, "low": 104, "close": 105},
        {"time": 2, "open": 105, "high": 106, "low": 101, "close": 102},
    ]
    normalized = render_triple_barrier_chart(
        candles,
        entry=105,
        stop=108,
        take=101,
        entry_index=0,
    )

    assert normalized.break_event is not None
    assert normalized.break_event.kind == "take"

