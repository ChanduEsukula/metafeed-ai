import csv

import pytest

from metafeed.data_pipeline.kuairec import (
    KuaiRecDataError,
    iter_kuairec,
    parse_interaction,
)


def valid_row():
    return {
        "user_id": "0",
        "video_id": "3649",
        "play_duration": "13838",
        "video_duration": "10867",
        "timestamp": "1593878903.438",
        "watch_ratio": "1.2733965215790926",
    }


def test_parse_real_example():
    event = parse_interaction(valid_row(), 2)

    assert event.user_id == 0
    assert event.video_id == 3649
    assert event.watch_ratio > 1


def test_negative_identifier_rejected():
    row = valid_row()
    row["video_id"] = "-1"

    with pytest.raises(KuaiRecDataError):
        parse_interaction(row, 2)


def test_zero_duration_rejected():
    row = valid_row()
    row["video_duration"] = "0"

    with pytest.raises(KuaiRecDataError):
        parse_interaction(row, 2)


def test_nonfinite_ratio_rejected():
    row = valid_row()
    row["watch_ratio"] = "nan"

    with pytest.raises(KuaiRecDataError):
        parse_interaction(row, 2)


def test_bad_identifier_rejected():
    row = valid_row()
    row["user_id"] = "not-an-integer"

    with pytest.raises(KuaiRecDataError):
        parse_interaction(row, 2)


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,video_id\n0,1\n")

    with pytest.raises(KuaiRecDataError):
        list(iter_kuairec(path))


def test_stream_limit(tmp_path):
    path = tmp_path / "sample.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(valid_row()),
        )
        writer.writeheader()
        writer.writerow(valid_row())
        writer.writerow(valid_row())

    events = list(iter_kuairec(path, limit=1))

    assert len(events) == 1
    assert events[0].video_id == 3649


def test_zero_limit(tmp_path):
    path = tmp_path / "sample.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(valid_row()),
        )
        writer.writeheader()
        writer.writerow(valid_row())

    assert list(iter_kuairec(path, limit=0)) == []
