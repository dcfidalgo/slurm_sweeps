from slurm_sweeps.sampler import Grid, Sampler


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
