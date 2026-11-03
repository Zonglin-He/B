"""Bounded CPU decoding for separately admitted post-replay scoring."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import struct
import zipfile
import zlib

import numpy as np

from .bootstrap import require


def stable_bytes(path, max_bytes, digest=None, expected_identity=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
            and path.is_file(), "Noncanonical score payload")
    before = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if expected_identity is not None:
        require(identity(before)[:4] == expected_identity, "Frozen opened-file identity differs")
    require(0 < before.st_size <= max_bytes, "Score payload exceeds byte bound")
    with path.open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before)
                and path.resolve(strict=True) == path, "Opened payload descriptor identity differs")
        content = stream.read(max_bytes+1)
        require(identity(os.fstat(stream.fileno())) == identity(before), "Opened payload changed during read")
    require(identity(before) == identity(path.stat()) and len(content) == before.st_size,
            "Score payload changed while reading")
    actual = hashlib.sha256(content).hexdigest()
    require(digest is None or actual == digest, "Score payload digest differs")
    return content, actual


def npy_header(stream, shape, dtype, total):
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "Unsupported scored NPY version")
    start = stream.tell()
    length_bytes = stream.read(2 if version == (1, 0) else 4)
    require(len(length_bytes) == (2 if version == (1, 0) else 4), "Truncated NPY header length")
    header_length = struct.unpack("<H" if version == (1, 0) else "<I", length_bytes)[0]
    require(0 < header_length <= 4096, "NPY header length exceeds pre-read bound")
    stream.seek(start)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    dimensions, fortran, actual = reader(stream, max_header_size=4096)
    require(dimensions == tuple(shape) and not fortran and actual == np.dtype(dtype)
            and not actual.hasobject and total == stream.tell()+int(np.prod(shape))*actual.itemsize,
            "Scored NPY geometry/dtype/size differs")


def support_array(content, reference, classes, shape):
    expected = (classes, *shape)
    require(type(classes) is int and 1 <= classes <= 81 and len(shape) == 2
            and all(type(n) is int and 0 < n <= 224 for n in shape), "Scored support shape exceeds bound")
    dtype = reference["probabilities_dtype"]
    require(dtype in ("float16", "float32") and reference["shape"] == list(expected), "Scored support metadata differs")
    raw = int(np.prod(expected))*np.dtype(dtype).itemsize
    require(isinstance(content, bytes) and 0 < len(content) <= 32*1024**2, "Support archive exceeds byte bound")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        require(len(entries) == 1 and entries[0].filename == "probabilities.npy"
                and entries[0].flag_bits & (1 | 8) == 0 and entries[0].file_size <= raw+4096
                and entries[0].compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                and entries[0].header_offset == 0 and not archive.comment,
                "Scored support ZIP inventory/resource bound differs")
        entry = entries[0]
        require(len(content) >= 30, "Truncated ZIP local header")
        local = struct.unpack("<4s5H3I2H", content[:30])
        require(local[0] == b"PK\x03\x04" and local[2] == entry.flag_bits and local[3] == entry.compress_type
                and local[-2] == len(b"probabilities.npy") and local[-1] <= 4096,
                "Unexpected support ZIP local header")
        start = 30+local[-2]+local[-1]
        require(content[30:30+local[-2]] == b"probabilities.npy"
                and start+entry.compress_size == archive.start_dir, "ZIP compressed extent differs")
        compressed = content[start:start+entry.compress_size]
        if entry.compress_type == zipfile.ZIP_STORED:
            require(len(compressed) <= raw+4096, "Stored support exceeds actual-output bound")
            data = compressed
        else:
            # ZipExtFile can trust forged central-directory sizes and decompress
            # more than read(n). Bound zlib's actual output independently.
            inflater = zlib.decompressobj(-15)
            data = inflater.decompress(compressed, raw+4096+1)
            require(len(data) <= raw+4096 and inflater.eof and not inflater.unconsumed_tail
                    and not inflater.unused_data, "Actual decompression exceeds support bound")
        require(len(data) == entry.file_size and zlib.crc32(data) & 0xffffffff == entry.CRC,
                "Decompressed support size/CRC differs")
    npy_header(io.BytesIO(data), expected, dtype, len(data))
    array = np.load(io.BytesIO(data), allow_pickle=False)
    require(bool(np.isfinite(array).all()) and bool(((array >= 0) & (array <= 1)).all()),
            "Invalid mapped support values")
    # MAX-collapsed query channels intentionally need not sum to one.
    return array


def hard_array(content, classes, shape):
    require(len(content) <= 2*1024**2, "Scored hard-label payload too large")
    npy_header(io.BytesIO(content), shape, "uint16", len(content))
    array = np.load(io.BytesIO(content), allow_pickle=False)
    require(bool((array < classes).all()), "Scored hard labels outside vocabulary")
    return array


def annotation_bytes(row):
    details = row["label_metadata_details"]
    path = Path(row["label"])
    def check():
        state = path.stat()
        require([state.st_size, state.st_mtime_ns] == row["label_metadata"]
                and dict(size_bytes=state.st_size, mtime_ns=state.st_mtime_ns,
                         device=state.st_dev, inode=state.st_ino, contents_read=False) == details,
                "Frozen annotation identity changed")
    check()
    frozen = (details["device"], details["inode"], details["size_bytes"], details["mtime_ns"])
    content, digest = stable_bytes(path, 16*1024**2, expected_identity=frozen)
    check()
    return content, digest
