import numpy as np
import pytest

from triples_sigfast.core.pipeline import SigPipeline
from triples_sigfast.io.raw import RawReader


def test_pipeline_basic():
    # Test SigPipeline map and filter
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    pipeline = SigPipeline(data)

    # map
    mapped = pipeline.map(lambda x: x * 2)
    res = mapped.compute()
    assert np.array_equal(res[0], [2, 4, 6])
    assert np.array_equal(res[1], [8, 10, 12])

    # filter
    filtered = pipeline.filter(lambda x: x > 3)
    res2 = filtered.compute()
    assert np.array_equal(res2[0], [])
    assert np.array_equal(res2[1], [4, 5, 6])


def test_raw_reader_iterate(tmp_path):
    # Test RawReader iterate
    f = tmp_path / "test.csv"
    f.write_text("A,B\n1,2\n3,4\n5,6\n7,8\n")
    reader = RawReader(str(f))

    pipeline = reader.iterate(chunksize=2)
    chunks = pipeline.compute()
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert chunks[0].iloc[0]["A"] == 1
    assert chunks[0].iloc[0]["B"] == 2
