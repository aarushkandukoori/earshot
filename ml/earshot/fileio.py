"""The `.esw` / `.est` container: one file = 16-byte preamble + JSON header + 64-byte aligned arrays (contract 3.1).

The weights file (`earshot.esw`, magic ESW1) and the test-vector files (`*.est`, magic EST1) share this layout:

    offset 0    4 bytes   magic, ASCII "ESW1" or "EST1"
    offset 4    uint32    format_version = 1
    offset 8    uint32    header_len = number of bytes of UTF-8 JSON that follow (no NUL at the end)
    offset 12   uint32    reserved = 0
    offset 16   header_len bytes of JSON
    zeros up to data_start = round_up(16 + header_len, 64)
    data_start  the arrays, each starting on a 64-byte boundary (relative to data_start), zeros between them

All numbers are little-endian. The JSON header lists every array as {name, dtype, shape, offset, nbytes}, with the
offset counted from data_start, plus `data_bytes` and `data_sha256` (SHA-256 of everything from data_start to the
end of the file). For a weights file that hash is the model_hash (contract 3.1): it names the trained numbers and does
not change when only `meta` changes.

The C engine reads the same files (engine/src/model.c), so this writer keeps to what its small JSON parser accepts:
plain ASCII text with no \\uXXXX escapes.
"""
import hashlib
import json
import os
import struct

import numpy as np

FORMAT_VERSION = 1
ALIGN = 64
MAGICS = (b"ESW1", b"EST1")
FORMAT_OF_MAGIC = {b"ESW1": "earshot-weights", b"EST1": "earshot-testvec"}

# Contract dtype names and the exact numpy types they mean (little-endian, so the file is the same on any machine).
DTYPES = {"f32": np.dtype("<f4"), "u8": np.dtype("u1"), "i32": np.dtype("<i4")}


def round_up(n, multiple=ALIGN):
    """Smallest multiple of `multiple` that is >= n."""
    return ((n + multiple - 1) // multiple) * multiple


def dtype_name(array):
    """The contract name of a numpy array's dtype ("f32", "u8" or "i32"). Anything else is an error, so a float64 or
    bool array is never written by accident."""
    kind = array.dtype
    if kind == np.float32:
        return "f32"
    if kind == np.uint8:
        return "u8"
    if kind == np.int32:
        return "i32"
    raise ValueError(f"unsupported dtype {kind}: the container holds only float32, uint8 and int32 arrays")


def sha256_hex(data):
    """Lowercase hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------------
def build_data(arrays):
    """Lay out the arrays: returns (data bytes, list of array entries for the header).

    `arrays` is an ordered list of (name, numpy array). Each array starts at the next 64-byte boundary of the data
    region (zero padding between arrays), and there is no padding after the last array.
    """
    blob = bytearray()
    entries = []
    names = set()
    for name, array in arrays:
        if not isinstance(name, str) or not name.isascii() or name in names:
            raise ValueError(f"array name {name!r} must be a new ASCII string")
        names.add(name)
        array = np.asarray(array)
        dname = dtype_name(array)
        # C order and little-endian, then raw bytes.
        raw = np.ascontiguousarray(array, dtype=DTYPES[dname]).tobytes()
        offset = round_up(len(blob))
        blob.extend(b"\x00" * (offset - len(blob)))
        blob.extend(raw)
        entries.append({"name": name, "dtype": dname, "shape": [int(s) for s in array.shape],
                        "offset": offset, "nbytes": len(raw)})
    return bytes(blob), entries


def header_text(header):
    """The JSON text written to the file: compact separators, plain ASCII, no \\u escapes (the engine's parser
    rejects them), no NaN or Infinity (not valid JSON)."""
    text = json.dumps(header, separators=(",", ":"), allow_nan=False)
    if not text.isascii() or "\\u" in text:
        raise ValueError("container header must be plain ASCII without \\u escapes")
    return text


def write_container(path, magic, header, arrays):
    """Write one container file and return the header that was written (with arrays, data_bytes, data_sha256).

    path   - output file. The file is written to path + ".tmp" first and then renamed, so a crash never leaves a
             half-written file under the real name.
    magic  - b"ESW1" (weights) or b"EST1" (test vectors), or the same as a str.
    header - dict with the caller's fields (format, format_version, model, model_version, config, config_hash, meta).
             The keys "arrays", "data_bytes" and "data_sha256" are filled in here.
    arrays - ordered list of (name, numpy array) with dtype float32, uint8 or int32.
    """
    if isinstance(magic, str):
        magic = magic.encode("ascii")
    if magic not in MAGICS:
        raise ValueError(f"magic must be one of {MAGICS}, got {magic!r}")

    data, entries = build_data(arrays)
    full_header = dict(header)
    full_header["arrays"] = entries
    full_header["data_bytes"] = len(data)
    full_header["data_sha256"] = sha256_hex(data)
    text = header_text(full_header).encode("utf-8")

    preamble = magic + struct.pack("<III", FORMAT_VERSION, len(text), 0)
    data_start = round_up(16 + len(text))
    padding = b"\x00" * (data_start - 16 - len(text))

    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(preamble)
        f.write(text)
        f.write(padding)
        f.write(data)
    os.replace(tmp, path)
    return full_header


# ---------------------------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------------------------
def parse_container(buf):
    """Parse and check a whole container held in memory -> (magic bytes, header dict, {name: numpy array}).

    Checks, in order, and raises ValueError naming the first bad field: size of the preamble, magic,
    format_version, reserved, header length, JSON, header format and format_version, data_bytes, data_sha256, and
    for every array its dtype, shape, nbytes, 64-byte aligned offset, bounds and list order.
    The returned arrays are copies (writable, independent of `buf`).
    """
    n = len(buf)
    if n < 16:
        raise ValueError(f"file: {n} bytes is shorter than the 16-byte preamble")
    magic = bytes(buf[0:4])
    if magic not in MAGICS:
        raise ValueError(f"magic: {magic!r}, expected one of {MAGICS}")
    version, header_len, reserved = struct.unpack("<III", bytes(buf[4:16]))
    if version != FORMAT_VERSION:
        raise ValueError(f"format_version: {version}, expected {FORMAT_VERSION}")
    if reserved != 0:
        raise ValueError(f"reserved: {reserved}, expected 0")
    if 16 + header_len > n:
        raise ValueError(f"header_len: {header_len} runs past the end of the {n}-byte file")
    try:
        header = json.loads(bytes(buf[16:16 + header_len]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"header: not valid JSON ({err})") from None
    if not isinstance(header, dict):
        raise ValueError("header: not a JSON object")

    if header.get("format") != FORMAT_OF_MAGIC[magic]:
        raise ValueError(f"format: {header.get('format')!r}, expected {FORMAT_OF_MAGIC[magic]!r} for {magic!r}")
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"format_version (header): {header.get('format_version')!r}, expected {FORMAT_VERSION}")

    data_start = round_up(16 + header_len)
    data_bytes = n - data_start
    if data_bytes < 0 or header.get("data_bytes") != data_bytes:
        raise ValueError(f"data_bytes: header says {header.get('data_bytes')!r}, file holds {data_bytes}")
    data = bytes(buf[data_start:])
    if header.get("data_sha256") != sha256_hex(data):
        raise ValueError("data_sha256: does not match the data region (file damaged or edited)")

    entries = header.get("arrays")
    if not isinstance(entries, list):
        raise ValueError("arrays: missing or not a list")
    arrays = {}
    previous_end = 0
    for entry in entries:
        name = entry.get("name")
        label = f"array {name}"
        if not isinstance(name, str) or name in arrays:
            raise ValueError(f"{label}: missing or repeated name")
        dname = entry.get("dtype")
        if dname not in DTYPES:
            raise ValueError(f"{label}: dtype {dname!r}, expected f32, u8 or i32")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(isinstance(s, int) and s >= 0 for s in shape):
            raise ValueError(f"{label}: shape {shape!r} is not a list of sizes")
        count = 1
        for s in shape:
            count *= s
        nbytes = entry.get("nbytes")
        if nbytes != count * DTYPES[dname].itemsize:
            raise ValueError(f"{label}: nbytes {nbytes!r}, shape {shape} {dname} needs {count * DTYPES[dname].itemsize}")
        offset = entry.get("offset")
        if not isinstance(offset, int) or offset < 0 or offset % ALIGN != 0:
            raise ValueError(f"{label}: offset {offset!r} is not a non-negative multiple of {ALIGN}")
        if offset < previous_end:
            raise ValueError(f"{label}: offset {offset} overlaps the previous array (arrays must be in list order)")
        if offset + nbytes > data_bytes:
            raise ValueError(f"{label}: offset {offset} + nbytes {nbytes} is past the end of the data ({data_bytes})")
        values = np.frombuffer(data, dtype=DTYPES[dname], count=count, offset=offset)
        arrays[name] = values.reshape(shape).copy()
        previous_end = offset + nbytes
    return magic, header, arrays


def read_container(path, expect_magic=None):
    """Read and check a container file -> (magic as str, header dict, {name: numpy array}).

    expect_magic - optional "ESW1" or "EST1"; a different magic raises ValueError.
    """
    with open(path, "rb") as f:
        buf = f.read()
    magic, header, arrays = parse_container(buf)
    magic = magic.decode("ascii")
    if expect_magic is not None and magic != expect_magic:
        raise ValueError(f"magic: {magic!r}, expected {expect_magic!r}")
    return magic, header, arrays


def model_hash(path):
    """The contract 3.1 model_hash of a weights file: its verified header data_sha256 (64 lowercase hex chars)."""
    _, header, _ = read_container(path, expect_magic="ESW1")
    return header["data_sha256"]
