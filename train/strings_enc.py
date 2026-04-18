"""prefix-shared ngram string encoding for binary weight files.

each bucket (unigrams, bigrams, ...) is sorted lexicographically so consecutive
entries share long byte-level prefixes. each entry is emitted as a single
nibble-packed header byte `(shared_len << 4) | suffix_len` followed by the
suffix bytes. the loader resets the "previous bytes" buffer at each bucket
boundary so bucket-level slicing via ngramCounts[] still works.

shared and suffix lengths are each 4 bits, capping at 15 bytes. the encoder
asserts this invariant.
"""

from __future__ import annotations

# nibble cap — shared_len and suffix_len must both fit in 4 bits
_NIBBLE_MAX = 15


def bucket_sort_perms(buckets: list[list[str]]) -> list[list[int]]:
    """compute the canonical per-bucket sort order used by prefix encoding.

    shared with `train.canonicalize_vocabs` so training-time canonicalization
    and export-time encoding agree on the column order.

    @param buckets list of ngram buckets (one list per ngram type)
    @returns permutations[i][j] = original index within bucket i of the j-th
      entry in sorted order
    """
    return [
        sorted(range(len(bucket)), key=lambda i, b=bucket: b[i].encode("utf-8"))
        for bucket in buckets
    ]


def encode_prefix_buckets(buckets: list[list[str]]) -> tuple[bytes, list[list[int]]]:
    """encode ngram buckets using nibble-packed prefix sharing.

    each bucket is sorted lexicographically by UTF-8 bytes; within a bucket,
    each entry is emitted as `(shared_len << 4 | suffix_len)` + suffix bytes.
    the previous-bytes state resets at bucket boundaries.

    @param buckets list of ngram buckets (one list per ngram type)
    @returns (encoded_bytes, permutations) where permutations[i][j] is the
      original (pre-sort) index within bucket i of the j-th encoded entry —
      callers use this to permute the corresponding weight-matrix columns
    @throws ValueError if any shared or suffix length exceeds 15 bytes
    """
    out = bytearray()
    perms = bucket_sort_perms(buckets)
    for bucket, order in zip(buckets, perms, strict=True):
        prev = b""
        for i in order:
            cur = bucket[i].encode("utf-8")
            m = min(len(cur), len(prev))
            shared = 0
            while shared < m and cur[shared] == prev[shared]:
                shared += 1
            suffix = cur[shared:]
            if shared > _NIBBLE_MAX or len(suffix) > _NIBBLE_MAX:
                raise ValueError(
                    f"prefix encoding overflow: shared={shared}, suffix_len={len(suffix)} "
                    f"(max {_NIBBLE_MAX}) for ngram {cur!r}",
                )
            out.append((shared << 4) | len(suffix))
            out.extend(suffix)
            prev = cur
    return bytes(out), perms
