import numpy as np
import pytest

from slurm_sweeps.sampler import Choice, Grid, LogUniform, Sampler, Uniform


def test_uniform():
    parameter = Uniform(0, 2, seed=42)
    samples = np.array([parameter() for _ in range(100)])

    np.testing.assert_allclose(
        samples[:3], np.array([1.547912, 0.877757, 1.717196]), rtol=1e-5
    )
    np.testing.assert_array_less(samples, 2)
    np.testing.assert_array_less(0, samples)


def test_loguniform():
    parameter = LogUniform(1, 10, seed=42)
    samples = np.array([parameter() for _ in range(100)])

    np.testing.assert_allclose(
        samples[:3], np.array([5.94232, 2.747125, 7.22101]), rtol=1e-5
    )
    np.testing.assert_array_less(samples, 10)
    np.testing.assert_array_less(1, samples)


def test_choice():
    with pytest.raises(AssertionError):
        Choice([1, "str"])

    parameter = Choice(list(range(5)), seed=43)
    samples = [parameter() for _ in range(100)]
    assert samples[:3] == [2, 3, 2]
    assert all(sample in [0, 1, 2, 3, 4] for sample in samples)


def test_grid():
    cfg = {"a": Grid([1, 2, 3])}
    assert cfg["a"]() == [1, 2, 3]


class TestSampler:
    def test_grid(self):
        cfg = {"a": Grid([1, 2, 3])}
        samples = list(Sampler(cfg))
        assert samples == [{"a": 1}, {"a": 2}, {"a": 3}]

        cfg = {"a": Grid([1, 2]), "b": {"aa": Grid([3, 4])}}
        samples = list(Sampler(cfg))
        assert samples == [
            {"a": 1, "b": {"aa": 3}},
            {"a": 1, "b": {"aa": 4}},
            {"a": 2, "b": {"aa": 3}},
            {"a": 2, "b": {"aa": 4}},
        ]
