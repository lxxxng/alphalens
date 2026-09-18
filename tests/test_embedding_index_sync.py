"""Tests for incremental FAISS and PostgreSQL ID reconciliation."""

import unittest

import faiss
import numpy as np

from pipelines.sec.embedder import reconcile_index_ids as reconcile_sec_ids
from pipelines.transcripts.embedder import (
    reconcile_index_ids as reconcile_transcript_ids,
)


def _index_with_ids(chunk_ids: list[int]):
    index = faiss.IndexIDMap2(
        faiss.IndexFlatIP(3)
    )
    vectors = np.eye(
        len(chunk_ids),
        3,
        dtype="float32",
    )
    index.add_with_ids(
        vectors,
        np.array(chunk_ids, dtype="int64"),
    )
    return index


class EmbeddingIndexSyncTests(unittest.TestCase):
    def test_sec_reconciliation_prunes_only_orphaned_vectors(self):
        index = _index_with_ids([1, 2, 99])

        remaining, removed = reconcile_sec_ids(
            [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}],
            index,
            {1, 2, 99},
        )

        self.assertEqual(removed, 1)
        self.assertEqual(remaining, {1, 2})
        self.assertEqual(index.ntotal, 2)

    def test_transcript_reconciliation_is_a_noop_when_ids_match(self):
        index = _index_with_ids([7, 8])

        remaining, removed = reconcile_transcript_ids(
            [{"chunk_id": 7}, {"chunk_id": 8}],
            index,
            {7, 8},
        )

        self.assertEqual(removed, 0)
        self.assertEqual(remaining, {7, 8})
        self.assertEqual(index.ntotal, 2)


if __name__ == "__main__":
    unittest.main()
