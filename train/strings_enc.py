"""string-storage encodings for binary weight files.

format v5 adds a "prefix" encoding for the ngram string payload. each bucket
(unigrams, bigrams, ...) is sorted lexicographically so consecutive entries
share long byte-level prefixes. each entry is emitted as a single nibble-packed
length byte `(shared_len << 4) | suffix_len` followed by the suffix bytes.

the loader resets the "previous bytes" buffer at each bucket boundary so
bucket-level slicing via ngramCounts[] still works.

the max observed shared/suffix length across all groups is 12 bytes, well
within the 4-bit range (0..15). the encoder asserts this invariant.
"""

from __future__ import annotations

# nibble cap — shared_len and suffix_len must both fit in 4 bits
_NIBBLE_MAX = 15


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
    perms: list[list[int]] = []
    for bucket in buckets:
        # sort indices by utf-8-encoded string bytes so we can permute weights
        encoded = [s.encode("utf-8") for s in bucket]
        order = sorted(range(len(bucket)), key=lambda i: encoded[i])
        perms.append(order)
        prev = b""
        for i in order:
            cur = encoded[i]
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
