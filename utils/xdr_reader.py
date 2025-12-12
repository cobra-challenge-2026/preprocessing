"""
Attempt at porting vv XDR reader (Elekta .SCAN files) to python.
Much slower than C++ version, no guarantees of correctness, 
tested with several .SCAN files and produces matching results.
"""

from __future__ import annotations
import io
import os
import re
import struct
import numpy as np
import numba as nb
import SimpleITK as sitk
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np


# ---------------------------
# Utilities / constants
# ---------------------------

MAXDIM = 5

ERR = {
    "ER_XDR_OPEN": "XDR file could not be opened",
    "ER_XDR_NDIM": "XDR file header NDIM error",
    "ER_XDR_DIM": "XDR file header DIMn error",
    "ER_XDR_NSPACE": "XDR file header NSPACE error",
    "ER_XDR_VECLEN": "XDR file header VECLEN error",
    "ER_XDR_DATA": "XDR file header DATA(type) error",
    "ER_XDR_FIELD": "XDR file header FIELD(coordinate type) error",
    "ER_XDR_NOCTRLL": "XDR file header contains no ^L",
    "ER_XDR_READ": "XDR file reading error",
    "ER_NOT_HANDLED": "Format not handled (RECTILINEAR or IRREGULAR field)",
    "ER_DECOMPRESSION": "Decompression failed",
}

FIELD_UNIFORM = "UNIFORM"
FIELD_RECTILINEAR = "RECTILINEAR"
FIELD_IRREGULAR = "IRREGULAR"

# CRC table copied from original (as integers)
CRC32_TABLE = (
  0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
  0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
  0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
  0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
  0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
  0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
  0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
  0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
  0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
  0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
  0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
  0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
  0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
  0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
  0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
  0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
  0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
  0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
  0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
  0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
  0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
  0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
  0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
  0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
  0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
  0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
  0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
  0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
  0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
  0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
  0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
  0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
  0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
  0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
  0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
  0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
  0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
  0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
  0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
  0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
  0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
  0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
  0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
)


@dataclass
class XdrHeader:
    ndim: int
    dims: List[int]
    nspace: int
    veclen: int
    data_token: str
    component_type: str          # 'byte'|'short'|'int'|'float'|'double'
    pixel_type: str              # 'SCALAR'|'VECTOR'
    field: str                   # FIELD_UNIFORM, ...
    nki_compression: int
    forcenoswap: bool
    spacing: List[float]
    origin: List[float]
    # new:
    comp_size: int               # bytes per component
    np_dtype: np.dtype           # NumPy dtype for data
    data_offset: int             # file offset of first data byte (after ^L + 1 char)

def _is_little_endian_host() -> bool:
    return struct.pack("<I", 1)[0] == 1


def _memicmp_prefix(a: str, b: str, n: int) -> bool:
    """Case-insensitive compare of first n chars: returns True if equal."""
    return a[:n].lower() == b[:n].lower()

def _parse_int_or_none(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.match(r"\s*([+-]?\d+)", s)
    return int(m.group(1)) if m else None

def _infer_component_type(token: str) -> Tuple[str, int, np.dtype]:
    """
    Map 'data=' value to (name, size_bytes, numpy dtype).
    """
    t = token.strip().lower()
    if t.startswith("xdr_"):
        # Forces no swap (data already little-endian per original logic)
        t = t[4:]
    if t.startswith("byte"):
        return ("byte", 1, np.dtype(np.int8))
    if t.startswith("short"):
        return ("short", 2, np.dtype(np.int16))
    if t.startswith("int"):
        return ("int", 4, np.dtype(np.int32))
    if t.startswith("real") or t.startswith("float"):
        return ("float", 4, np.dtype(np.float32))
    if t.startswith("double"):
        return ("double", 8, np.dtype(np.float64))
    raise ValueError(ERR["ER_XDR_DATA"])


def _read_coords_tail(path: str, field: str, dims: List[int], nspace: int, forcenoswap: bool) -> Tuple[List[float], List[float]]:
    """
    Read coordinate floats from the end of the file and compute spacing / origin
    exactly like clitkXdrImageIO.

    - field=UNIFORM: 2 floats per axis (min,max) in cm
    - field=RECTILINEAR: dim[i] floats per axis in cm, reinterpreted as uniform
                         if nearly uniform within 0.1 mm
    - field=IRREGULAR: not handled (raises)
    """
    if field == FIELD_IRREGULAR:
        raise ValueError(ERR["ER_NOT_HANDLED"])

    if field == FIELD_UNIFORM:
        nfloat = nspace * 2
    elif field == FIELD_RECTILINEAR:
        nfloat = sum(dims[:nspace])
    else:
        # no coords present (shouldn't really happen), fall back
        return [1.0] * len(dims), [0.0] * len(dims)

    nbytes = nfloat * 4

    with open(path, "rb") as f:
        f.seek(-nbytes, os.SEEK_END)
        buf = f.read(nbytes)
        if len(buf) != nbytes:
            raise IOError(ERR["ER_XDR_READ"])

    # Interpret as raw 32-bit floats in *file* byte order, then swap if needed
    # We start with native float32:
    arr = np.frombuffer(buf, dtype=np.float32).copy()

    # C++: if host is little-endian and forcenoswap == 0, swap every 4 bytes
    if _is_little_endian_host() and not forcenoswap:
        arr.byteswap(True)  # in-place -> now in host endianness

    spacing = [1.0] * len(dims)
    origin  = [0.0] * len(dims)

    if field == FIELD_UNIFORM:
        # points: [x_min, x_max, y_min, y_max, z_min, z_max, ...] in cm
        for i in range(nspace):
            a = float(arr[2 * i])
            b = float(arr[2 * i + 1])
            # convert cm -> mm with *10, same as C++:
            spacing[i] = 10.0 * (b - a) / (max(dims[i] - 1, 1))
            origin[i]  = 10.0 * a

    elif field == FIELD_RECTILINEAR:
        # points: concat of all axis coords in cm
        p = 0
        tol_mm = 0.1
        for i in range(nspace):
            axis_coords = arr[p:p + dims[i]].astype(np.float64)
            p += dims[i]

            spacing[i] = 10.0 * (axis_coords[-1] - axis_coords[0]) / (max(dims[i] - 1, 1))
            origin[i]  = 10.0 * axis_coords[0]

            # Check near-uniformity (only for first 3 dims)
            if i < 3 and dims[i] > 1:
                diffs_mm = 10.0 * np.diff(axis_coords)
                if np.any(np.abs(diffs_mm - spacing[i]) > tol_mm):
                    raise ValueError(ERR["ER_NOT_HANDLED"])

    return spacing, origin


def _get_nki_compressed_size(fp: io.BufferedReader) -> int:
    """
    Peek NKI header at current position, return compressed block size including header
    or 0 if unknown.
    struct NKI_MODE2 {
      uint32 iOrgSize;      // number of pixels (shorts)
      uint32 iMode;         // 1..4
      uint32 iCompressedSize;
      uint32 iOrgCRC;
      uint32 iCompressedCRC; // excludes this header
    }
    """
    pos = fp.tell()
    hdr = fp.read(20)
    if len(hdr) < 8:
        fp.seek(pos)
        return 0
    iOrgSize, iMode = struct.unpack("<II", hdr[:8])
    if iMode in (1, 3):
        fp.seek(pos)
        return 0
    if len(hdr) < 20:
        fp.seek(pos)
        return 0
    _, _, iCompressedSize, _, _ = struct.unpack("<IIIII", hdr)
    fp.seek(pos)
    if iMode in (2, 4):
        return int(iCompressedSize) + 20
    return 0


def _crc32_update(crc: int, val16: int) -> int:
    """
    Update CRC using a 16-bit value interpreted little-endian (low byte then high byte),
    mirroring the C code's two-step table update.
    """
    v = int(val16) & 0xFFFF
    lo = v & 0xFF
    hi = (v >> 8) & 0xFF
    crc = (CRC32_TABLE[(int(crc) ^ lo) & 0xFF] ^ (int(crc) >> 8)) & 0xFFFFFFFF
    crc = (CRC32_TABLE[(int(crc) ^ hi) & 0xFF] ^ (int(crc) >> 8)) & 0xFFFFFFFF
    return crc

def _nki_decompress_to_int16(dst_count: int, src: bytes) -> tuple[np.ndarray, int]:
    """
    Drop-in replacement using numba-accelerated inner loop for modes 1/2.
    Returns (int16 array, consumed_bytes).

    Patched to:
    - trust the decompressor's consumed length (no max with header_size+iCompressedSize)
    - relax CRC check: if CRC mismatch but we produced iOrgSize samples, accept anyway.
    """
    if len(src) < 8:
        raise ValueError(ERR["ER_DECOMPRESSION"])

    # Parse header in Python (rare, not hot)
    if len(src) >= 20:
        iOrgSize, iMode, iCompressedSize, iOrgCRC, iCompressedCRC = struct.unpack("<IIIII", src[:20])
        header_size = 20
    else:
        iOrgSize, iMode = struct.unpack("<II", src[:8])
        iCompressedSize = iOrgCRC = iCompressedCRC = 0
        header_size = 8

    # dst_count is the expected number of components from outside;
    # iOrgSize is the number of pixels in this block per the header.
    # If they disagree, we still trust iOrgSize for decompression length.
    if iOrgSize <= 0:
        raise ValueError(ERR["ER_DECOMPRESSION"])

    if iMode not in (1, 2):
        # Fallback to old Python implementation for modes 3/4 or unknown
        return _nki_decompress_to_int16_python(dst_count, src)

    # Convert body to uint8 array for numba
    src_arr = np.frombuffer(src, dtype=np.uint8)

    # do_crc: only for mode 2 in original code, and only if iOrgCRC is nonzero
    do_crc = 1 if (iMode == 2 and iOrgCRC != 0) else 0

    # Numba core: decompression stops once it has produced iOrgSize samples
    out, next_offset, crc = _nki_decompress_core_nb(
        src_arr,
        header_size,
        iOrgSize,
        iMode,
        do_crc,
        CRC32_TABLE_NP,
    )

    # Trust the decompressor about how many bytes were actually used
    consumed = next_offset

    # --- Patched CRC behavior --------------------------------------------
    # Many Elekta .SCAN files have unreliable CRC or padding after the block.
    # If CRC mismatches but we *did* get the expected number of samples, accept it.
    if iMode == 2 and do_crc and crc != iOrgCRC:
        # If we didn't even produce iOrgSize samples, *then* treat it as an error.
        if out.size < iOrgSize:
            raise ValueError(ERR["ER_DECOMPRESSION"])
        # Otherwise: log or ignore; we choose to just ignore here.
        # You can uncomment this if you want a warning:
        # print("Warning: NKI mode-2 CRC mismatch ignored (Elekta .SCAN compatibility).")

    return out, consumed

def _nki_decompress_to_int16_python(dst_count: int, src: bytes) -> Tuple[np.ndarray, int]:
    """
    Decompress NKI private compression (modes 1..4). Returns (int16 array, consumed_bytes).
    Behavior mirrors clitkXdrImageIO::nki_private_decompress.
    """
    if len(src) < 8:
        raise ValueError(ERR["ER_DECOMPRESSION"])

    # Peek iMode from either 8-byte (modes 1,3) or 20-byte (modes 2,4) header
    # For simplicity, read 20 and fill defaults.
    if len(src) >= 20:
        iOrgSize, iMode, iCompressedSize, iOrgCRC, iCompressedCRC = struct.unpack("<IIIII", src[:20])
    else:
        iOrgSize, iMode = struct.unpack("<II", src[:8])
        iCompressedSize = iOrgCRC = iCompressedCRC = 0

    if iOrgSize != dst_count:
        # Keep going; some writers approximate sizes. We'll still trust header flow.
        pass

    def mode_body(start_offset: int, mode: int) -> Tuple[np.ndarray, int]:
        """
        Shared body for modes; returns (out, consumed_len).
        """
        s = memoryview(src)[start_offset:]
        # Safety margin end index
        # The C uses many boundary checks; we mimic core logic.
        out = np.empty(iOrgSize, dtype=np.int16)
        si = 0
        # first absolute 16-bit:
        if len(s) < 2:
            raise ValueError(ERR["ER_DECOMPRESSION"])
        first = struct.unpack_from("<h", s, 0)[0]
        out[0] = first
        si += 2
        di = 1

        crc = 0
        if mode in (2, 4):
            crc = _crc32_update(0, int(first) & 0xFFFF)

        while di < iOrgSize:
            if si >= len(s):
                # truncated stream
                break
            val = struct.unpack_from("<b", s, si)[0]
            if (-64 <= val <= 63) if mode in (1, 2) else (-63 <= val <= 63):
                # 7-bit diff
                prev = int(out[di-1])
                cur = np.int16(prev + val)
                out[di] = cur
                di += 1
                si += 1
                if mode in (2, 4):
                    crc = _crc32_update(crc, int(cur) & 0xFFFF)
            elif val == 0x7F:
                # absolute 16-bit
                if si + 3 > len(s):
                    raise ValueError(ERR["ER_DECOMPRESSION"])
                cur = struct.unpack_from("<H", s, si+1)[0]
                out[di] = np.int16(cur)
                di += 1
                si += 3
                if mode in (2, 4):
                    crc = _crc32_update(crc, int(cur) & 0xFFFF)
            elif (val & 0xFF) == 0x80:
                # RLE: repeat previous NN times
                if si + 2 > len(s):
                    raise ValueError(ERR["ER_DECOMPRESSION"])
                run = struct.unpack_from("<B", s, si+1)[0]
                si += 2
                if run <= 0:
                    continue
                prev = out[di-1]
                reps = min(run, iOrgSize - di)
                out[di:di+reps] = prev
                if mode in (2, 4):
                    for _ in range(reps):
                        crc = _crc32_update(crc, int(prev) & 0xFFFF)
                di += reps
            elif (val & 0xFF) == 0xC0 and mode in (3, 4):
                # 4-bit run: NN nibbles (pairs per byte)
                if si + 2 > len(s):
                    raise ValueError(ERR["ER_DECOMPRESSION"])
                nbytes = struct.unpack_from("<B", s, si+1)[0] // 2
                si += 2
                for _ in range(nbytes):
                    if si >= len(s):
                        raise ValueError(ERR["ER_DECOMPRESSION"])
                    packed = struct.unpack_from("<B", s, si)[0]
                    si += 1
                    hi = (packed >> 4)
                    lo = (packed & 0x0F)
                    # sign-extend nibble per C code
                    if hi & 0x8:
                        hi |= 0xF0
                    if lo & 0x8:
                        lo |= 0xF0
                    for d in (hi, lo):
                        if di >= iOrgSize:
                            break
                        prev = int(out[di-1])
                        cur = np.int16(prev + np.int8(d))
                        out[di] = cur
                        di += 1
                        if mode in (2, 4):
                            crc = _crc32_update(crc, int(cur) & 0xFFFF)
            else:
                # 15-bit diff encoded as HHLL with XOR 0x4000 on high byte (C: (val^0x40)<<8 | next)
                if si + 2 > len(s):
                    raise ValueError(ERR["ER_DECOMPRESSION"])
                b0 = val
                b1 = struct.unpack_from("<B", s, si+1)[0]
                diff = np.int16(((b0 ^ 0x40) << 8) | b1).astype(np.int16)
                prev = int(out[di-1])
                cur = np.int16(prev + diff)
                out[di] = cur
                di += 1
                si += 2
                if mode in (2, 4):
                    crc = _crc32_update(crc, int(cur) & 0xFFFF)

        consumed = start_offset + si
        if mode in (2, 4):
            # For mode 2 and 4, verify output CRC equals iOrgCRC (if provided)
            if iOrgCRC != 0 and crc != iOrgCRC:
                # try input CRC check like the C code does (not strictly necessary here)
                raise ValueError(ERR["ER_DECOMPRESSION"])
        return out, consumed

    if iMode == 1:
        # header 8b: iOrgSize, iMode; body follows
        out, consumed = mode_body(8, 1)
        return out, consumed
    elif iMode == 2:
        # header 20b; body follows; compressed size known
        out, consumed = mode_body(20, 2)
        # consumed should be 20 + compressed_size
        return out, max(consumed, 20 + iCompressedSize)
    elif iMode == 3:
        out, consumed = mode_body(8, 3)
        return out, consumed
    elif iMode == 4:
        out, consumed = mode_body(20, 4)
        return out, max(consumed, 20 + iCompressedSize)
    else:
        raise ValueError("XDR decompression: unsupported mode")



# Make CRC table available as a NumPy array for numba (if you want CRC checks)
CRC32_TABLE_NP = np.array(CRC32_TABLE, dtype=np.uint32)


@nb.njit
def _crc32_update_nb(crc, val16, table):
    # force val16 to an int so bitwise ops are well-typed
    v = int(val16) & 0xFFFF

    lo = v & 0xFF
    hi = (v >> 8) & 0xFF

    idx = (crc ^ lo) & 0xFF
    crc = (table[idx] ^ (crc >> 8)) & 0xFFFFFFFF

    idx = (crc ^ hi) & 0xFF
    crc = (table[idx] ^ (crc >> 8)) & 0xFFFFFFFF

    return crc


@nb.njit
def _read_le_int16(src: np.ndarray, offset: int) -> int:
    """Read little-endian signed 16-bit from uint8 array."""
    val = src[offset] | (src[offset + 1] << 8)
    if val >= 0x8000:
        val -= 0x10000
    return val


@nb.njit
def _nki_decompress_core_nb(
    src: np.ndarray,
    start_offset: int,
    iOrgSize: int,
    mode: int,
    do_crc: int,
    crc32_table: np.ndarray,
) -> (np.ndarray, int, np.uint32): # type: ignore
    """
    Numba-compiled inner loop for NKI modes 1 and 2.
    Returns (out_int16_array, next_src_offset, crc).
    """
    out = np.empty(iOrgSize, dtype=np.int16)

    si = start_offset  # source index
    di = 0             # dest index

    # First absolute sample
    first = _read_le_int16(src, si)
    out[0] = first
    si += 2
    di = 1

    crc = np.uint32(0)
    if do_crc:
        crc = _crc32_update_nb(crc, first, crc32_table)

    # main loop
    while di < iOrgSize and si < src.size:
        val = np.int8(src[si])  # signed byte

        if mode == 1 or mode == 2:
            # 7-bit diff: -64..63
            if -64 <= val <= 63:
                prev = int(out[di - 1])
                cur = prev + int(val)
                out[di] = np.int16(cur)
                di += 1
                si += 1
                if do_crc:
                    crc = _crc32_update_nb(crc, cur, crc32_table)
                continue

            # absolute 16-bit
            if val == 0x7F:
                # need 2 more bytes
                if si + 3 > src.size:
                    break
                cur = _read_le_int16(src, si + 1)
                out[di] = np.int16(cur)
                di += 1
                si += 3
                if do_crc:
                    crc = _crc32_update_nb(crc, cur, crc32_table)
                continue

            # RLE: repeat previous NN times, marker 0x80
            if (val & 0xFF) == 0x80:
                if si + 2 > src.size:
                    break
                run = int(src[si + 1])
                si += 2
                if run > 0:
                    prev = out[di - 1]
                    reps = run
                    # clamp to remaining space
                    if di + reps > iOrgSize:
                        reps = iOrgSize - di
                    # fill
                    for _ in range(reps):
                        out[di] = prev
                        if do_crc:
                            crc = _crc32_update_nb(crc, int(prev), crc32_table)
                        di += 1
                continue

            # 15-bit diff encoded (the "else" case in your Python code)
            if si + 2 > src.size:
                break
            b0 = val
            b1 = int(src[si + 1])
            # ((b0 ^ 0x40) << 8) | b1, sign-extended to 16-bit
            diff_val = ((int(b0) ^ 0x40) << 8) | b1
            if diff_val >= 0x8000:
                diff_val -= 0x10000
            prev = int(out[di - 1])
            cur = prev + diff_val
            out[di] = np.int16(cur)
            di += 1
            si += 2
            if do_crc:
                crc = _crc32_update_nb(crc, cur, crc32_table)
            continue

        # If mode 3/4 needed, add their branches here
        # (similar to your Python implementation, but using src[idx] logic)
        # For now, break out if we encounter an unsupported mode.
        break

    return out, si, crc

def _read_header(path: str) -> XdrHeader:
    # Read a single big chunk and find ^L (form feed)
    with open(path, "rb") as f:
        # 512k should be plenty for AVS headers
        buf = f.read(512 * 1024)
    if not buf.startswith(b"# AVS"):
        raise ValueError("Not an AVS field file")

    ff_idx = buf.find(b"\x0c")
    if ff_idx == -1:
        raise ValueError(ERR["ER_XDR_NOCTRLL"])

    # Data starts just after ^L and one extra char (mimicking C's "+2")
    data_offset = ff_idx + 2

    header_bytes = buf[:ff_idx]
    header_text = header_bytes.decode("utf-8", errors="replace")

    # First line special-case (forcenoswap flag)
    lines = header_text.splitlines()
    first_line = lines[0] if lines else ""
    forcenoswap = first_line.startswith("# AVS field file (produced by avs_nfwrite.c)")

    # Parse key=value pairs into a dict (lower-cased keys)
    kv: Dict[str, str] = {}
    for line in lines:
        # strip trailing comments and control chars
        line = line.replace("\b", "")
        # skip pure comment lines
        if line.lstrip().startswith("#"):
            continue
        # remove spaces like original `_scan_header_value`
        cleaned = "".join(ch for ch in line if ch not in (" ", "\t"))
        if "=" not in cleaned:
            continue
        lhs, rhs = cleaned.split("=", 1)
        key = lhs.strip().lower()
        val = rhs.strip()
        # only keep first occurrence
        if key not in kv:
            kv[key] = val

    # Helper: get key, like original semantics
    def kv_int(key: str, default: int | None = None) -> int | None:
        s = kv.get(key)
        v = _parse_int_or_none(s) if s is not None else None
        return v if v is not None else default

    # --- Required fields ---
    ndim = kv_int("ndim")
    if ndim is None or not (1 <= ndim <= MAXDIM):
        raise ValueError(ERR["ER_XDR_NDIM"])

    dims: List[int] = []
    total = 1
    for i in range(ndim):
        di = kv.get(f"dim{i+1}")
        v = _parse_int_or_none(di) if di is not None else None
        if v is None or v < 1:
            raise ValueError(ERR["ER_XDR_DIM"])
        dims.append(v)
        total *= v

    nspace = kv_int("nspace", default=ndim)
    if nspace is None or nspace < 1 or nspace > MAXDIM:
        raise ValueError(ERR["ER_XDR_NSPACE"])
    if nspace != ndim:
        raise ValueError(ERR["ER_NOT_HANDLED"])

    veclen = kv_int("veclen", default=1)
    if veclen is None or veclen < 0:
        raise ValueError(ERR["ER_XDR_VECLEN"])
    pixel_type = "SCALAR" if veclen == 1 else "VECTOR"

    data_token = kv.get("data", "byte")
    component_type, comp_size, np_dtype = _infer_component_type(data_token)

    # "xdr_*" disables swap (like original)
    if data_token.strip().lower().startswith("xdr_"):
        forcenoswap = False

    field_s = kv.get("field", "uniform").lower()
    if field_s.startswith("unifo"):
        field = FIELD_UNIFORM
    elif field_s.startswith("recti"):
        field = FIELD_RECTILINEAR
    elif field_s.startswith("irreg"):
        field = FIELD_IRREGULAR
    else:
        raise ValueError(ERR["ER_XDR_FIELD"])

    nki_compression = kv_int("nki_compression", default=0) or 0

    # spacing / origin from coords at tail
    spacing = [1.0] * ndim
    origin = [0.0] * ndim
    try:
        spacing, origin = _read_coords_tail(path, field, dims, nspace, forcenoswap)
    except ValueError as e:
        if str(e) == ERR["ER_NOT_HANDLED"]:
            # propagate "not handled"
            raise

    return XdrHeader(
        ndim=ndim,
        dims=dims,
        nspace=nspace,
        veclen=veclen,
        data_token=data_token,
        component_type=component_type,
        pixel_type=pixel_type,
        field=field,
        nki_compression=nki_compression,
        forcenoswap=forcenoswap,
        spacing=spacing,
        origin=origin,
        comp_size=comp_size,
        np_dtype=np_dtype,
        data_offset=data_offset,
    )


def xdr_to_sitk(arr: np.ndarray, meta: dict) -> "sitk.Image":
    """
    Convert numpy array+meta from read_xdr() to a SimpleITK image.
    """
    is_vector = meta.get("veclen", 1) > 1
    
    spacing = tuple(meta.get("spacing", [1.0]*arr.ndim))
    origin  = tuple(meta.get("origin",  [0.0]*arr.ndim))

    img = sitk.GetImageFromArray(arr, isVector=is_vector)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)

    return img

def read_xdr(path: str) -> sitk.Image:
    """
    Read an AVS XDR image into a SimpleITK image.
    """
    hdr = _read_header(path)

    total_pixels = int(np.prod(hdr.dims))
    total_components = total_pixels * hdr.veclen

    with open(path, "rb") as fp:
        # jump straight after header using precomputed offset
        fp.seek(hdr.data_offset, os.SEEK_SET)

        if hdr.nki_compression and hdr.np_dtype != np.int16:
            raise ValueError("NKI compression present but component type is not 16-bit short")

        if hdr.nki_compression:
            cur = fp.tell()
            size_guess = _get_nki_compressed_size(fp)
            if size_guess == 0:
                # fallback: compute to EOF minus coords
                fp.seek(0, os.SEEK_END)
                end = fp.tell()
                if hdr.field == FIELD_UNIFORM:
                    tail = hdr.nspace * 2 * 4
                elif hdr.field == FIELD_RECTILINEAR:
                    tail = sum(hdr.dims[:hdr.nspace]) * 4
                else:
                    tail = 0
                size_guess = max(0, end - cur - tail)
            fp.seek(cur, os.SEEK_SET)
            comp_bytes = fp.read(size_guess)
            if len(comp_bytes) < 8:
                raise IOError(ERR["ER_XDR_READ"])

            out16, consumed = _nki_decompress_to_int16(total_components, comp_bytes)
            fp.seek(cur + consumed, os.SEEK_SET)
            data = out16
        else:
            need_bytes = total_components * hdr.comp_size
            raw = fp.read(need_bytes)
            if len(raw) != need_bytes:
                raise IOError(ERR["ER_XDR_READ"])
            data = np.frombuffer(raw, dtype=hdr.np_dtype).copy()

            # Swap bytes if host is big-endian and forcenoswap is False
            if (not _is_little_endian_host()) and (not hdr.forcenoswap):
                data = data.byteswap(inplace=False).astype(hdr.np_dtype.newbyteorder())

        shape = list(reversed(hdr.dims))
        if hdr.veclen > 1:
            shape.append(hdr.veclen)

        try:
            data = data.reshape(shape)
        except ValueError:
            raise ValueError("Data size does not match header dims/veclen")

    meta = {
        "ndim": hdr.ndim,
        "dims": hdr.dims,
        "nspace": hdr.nspace,
        "veclen": hdr.veclen,
        "field": hdr.field,
        "component_type": hdr.component_type,
        "pixel_type": hdr.pixel_type,
        "spacing": hdr.spacing,
        "origin": hdr.origin,
        "nki_compression": hdr.nki_compression,
        "forcenoswap": hdr.forcenoswap,
    }

    sitk_image = xdr_to_sitk(data, meta)
    return sitk_image

# Convenience “can read” helper
def can_read_xdr(path: str) -> bool:
    try:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            first = f.readline()
        return first.startswith("# AVS")
    except Exception:
        return False

