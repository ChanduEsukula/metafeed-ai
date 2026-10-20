import pytest

from metafeed.data_pipeline.sampling import sample_by_user


def test_per_user_limit():
    records = [
        (1, 10),
        (1, 11),
        (1, 12),
        (2, 20),
        (2, 21),
    ]

    sampled = sample_by_user(
        records,
        user_id=lambda record: record[0],
        max_per_user=2,
    )

    assert len(sampled) == 4
    assert sum(row[0] == 1 for row in sampled) == 2
    assert sum(row[0] == 2 for row in sampled) == 2


def test_reproducible_sampling():
    records = [(1, i) for i in range(100)]

    first = sample_by_user(
        records, lambda record: record[0], 10, seed=42
    )
    second = sample_by_user(
        records, lambda record: record[0], 10, seed=42
    )

    assert first == second


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        sample_by_user(
            [(1, 10)],
            lambda record: record[0],
            max_per_user=0,
        )


from metafeed.data_pipeline.sampling import reservoir_sample_by_user


def test_reservoir_sample_limits_each_user():
    records = [
        (user, item)
        for user in range(3)
        for item in range(100)
    ]

    sampled = reservoir_sample_by_user(
        iter(records),
        user_id=lambda row: row[0],
        max_per_user=10,
        seed=42,
    )

    assert set(sampled) == {0, 1, 2}
    assert all(len(rows) == 10 for rows in sampled.values())


def test_reservoir_sampling_reproducible():
    records = [(1, i) for i in range(200)]

    first = reservoir_sample_by_user(
        iter(records), lambda row: row[0], 15, seed=42
    )
    second = reservoir_sample_by_user(
        iter(records), lambda row: row[0], 15, seed=42
    )

    assert first == second


def test_reservoir_rejects_invalid_cap():
    with pytest.raises(ValueError):
        reservoir_sample_by_user(
            [(1, 10)],
            lambda row: row[0],
            max_per_user=0,
        )
