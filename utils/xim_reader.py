#!/usr/bin/env python3
"""
Python XIM reader implementing the same logic as rtk::XimImageIO.

Requirements:
    pip install numpy
"""

import os
import struct
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple, Any


import numpy as np
import SimpleITK as sitk

PROPERTY_NAME_MAX_LENGTH = 256


@dataclass
class XimHeader:
    # Basic file info
    sFileType: bytes = b""
    FileVersion: int = 0
    SizeX: int = 0
    SizeY: int = 0

    dBitsPerPixel: int = 0
    dBytesPerPixel: int = 0
    dCompressionIndicator: int = 0

    # Compression related (optional)
    lookUpTableSize: int = 0
    compressedPixelBufferSize: int = 0
    unCompressedPixelBufferSize: int = 0

    binsInHistogram: int = 0
    numberOfProperties: int = 0

    # Properties we actually care about (same as in C++ code)
    dCouchLat: float = 0.0
    dCouchLng: float = 0.0
    dCouchVrt: float = 0.0
    nPixelOffset: int = 0
    dCTProjectionAngle: float = 0.0
    dDetectorOffsetX: float = 0.0
    dDetectorOffsetY: float = 0.0
    dCollX1: float = 0.0
    dCollX2: float = 0.0
    dCollY1: float = 0.0
    dCollY2: float = 0.0
    dXRayKV: float = 0.0
    dXRayMA: float = 0.0
    dCTNormChamber: float = 0.0
    dGating4DInfoX: float = 0.0
    dGating4DInfoY: float = 0.0
    dGating4DInfoZ: float = 0.0
    dCollRtn: float = 0.0
    dDoseRate: float = 0.0
    dEnergy: float = 0.0
    dIDUResolutionX: float = 0.0  # PixelWidth * 10.0
    dIDUResolutionY: float = 0.0  # PixelHeight * 10.0


# -------------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------------


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise IOError(f"Unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _read_int32(f) -> int:
    # Int4 in the C++ code
    return struct.unpack("<i", _read_exact(f, 4))[0]


def _read_double(f) -> float:
    return struct.unpack("<d", _read_exact(f, 8))[0]


def _load_binary_char_as_int(bin_vals: bytes, n_bytes: int) -> int:
    if n_bytes == 1:
        return struct.unpack("<b", bin_vals)[0]
    elif n_bytes == 2:
        return struct.unpack("<h", bin_vals)[0]
    elif n_bytes == 4:
        return struct.unpack("<i", bin_vals)[0]
    else:
        raise ValueError(f"Unsupported number of bytes for diff: {n_bytes}")


def _cast_binary_char_to_int(bin_vals: bytes, n_bytes: int) -> int:
    # This mimics cast_binary_char_to<long long>
    return int(_load_binary_char_as_int(bin_vals, n_bytes))


def _lut_to_bytes(val: int) -> int:
    # Same as lut_to_bytes in C++
    if val == 0:
        return 1
    elif val == 1:
        return 2
    elif val == 2:
        return 4
    else:  # only 0, 1 & 2 should be possible
        return 8


# -------------------------------------------------------------------------
# Properties helper
# -------------------------------------------------------------------------


# Map property name prefixes to (XimHeader attribute, scale_factor)
_PROPERTY_MAP: Dict[str, Tuple[str, float]] = {
    "CouchLat": ("dCouchLat", 1.0),
    "CouchLng": ("dCouchLng", 1.0),
    "CouchVrt": ("dCouchVrt", 1.0),
    "DataOffset": ("nPixelOffset", 1.0),
    "KVSourceRtn": ("dCTProjectionAngle", 1.0),
    "KVDetectorLat": ("dDetectorOffsetX", 1.0),
    "KVDetectorLng": ("dDetectorOffsetY", 1.0),
    "KVCollimatorX1": ("dCollX1", 1.0),
    "KVCollimatorX2": ("dCollX2", 1.0),
    "KVCollimatorY1": ("dCollY1", 1.0),
    "KVCollimatorY2": ("dCollY2", 1.0),
    "KVKiloVolts": ("dXRayKV", 1.0),
    "KVMilliAmperes": ("dXRayMA", 1.0),
    "KVNormChamber": ("dCTNormChamber", 1.0),
    "MMTrackingRemainderX": ("dGating4DInfoX", 1.0),
    "MMTrackingRemainderY": ("dGating4DInfoY", 1.0),
    "MMTrackingRemainderZ": ("dGating4DInfoZ", 1.0),
    "MVCollimatorRtn": ("dCollRtn", 1.0),
    "MVCollimatorX1": ("dCollX1", 1.0),
    "MVCollimatorX2": ("dCollX2", 1.0),
    "MVCollimatorY1": ("dCollY1", 1.0),
    "MVCollimatorY2": ("dCollY2", 1.0),
    "MVDoseRate": ("dDoseRate", 1.0),
    "MVEnergy": ("dEnergy", 1.0),
    # multiplied by 10.0 in C++ (cm -> mm)
    "PixelHeight": ("dIDUResolutionY", 10.0),
    "PixelWidth": ("dIDUResolutionX", 10.0),
}


def _apply_property(header: XimHeader, name: str, value: float) -> None:
    for prefix, (attr, scale) in _PROPERTY_MAP.items():
        if name.startswith(prefix):
            setattr(header, attr, float(value) * scale)
            break


# -------------------------------------------------------------------------
# Header reading (ReadImageInformation)
# -------------------------------------------------------------------------

PropertyValue = Union[int, float, str, List[int], List[float], bytes]

def read_xim_header(path: str) -> Tuple[XimHeader, int, Dict[str, PropertyValue]]:
    """
    Parse XIM header and properties.

    Returns
    -------
    header : XimHeader
    image_data_start : int
        File offset where the image data (LUT + compressed buffer) starts.
    properties : dict
        Mapping property_name -> decoded value(s),
        including ones not mapped to XimHeader fields.
    """
    header = XimHeader()
    properties: Dict[str, PropertyValue] = {}

    with open(path, "rb") as f:
        # Initial header fields
        header.sFileType = _read_exact(f, 8)
        header.FileVersion = _read_int32(f)
        header.SizeX = _read_int32(f)
        header.SizeY = _read_int32(f)
        header.dBitsPerPixel = _read_int32(f)
        header.dBytesPerPixel = _read_int32(f)
        header.dCompressionIndicator = _read_int32(f)

        image_data_start = f.tell()

        if header.dCompressionIndicator == 1:
            header.lookUpTableSize = _read_int32(f)
            f.seek(header.lookUpTableSize, os.SEEK_CUR)
            header.compressedPixelBufferSize = _read_int32(f)
            f.seek(header.compressedPixelBufferSize, os.SEEK_CUR)
            header.unCompressedPixelBufferSize = _read_int32(f)
        else:
            header.unCompressedPixelBufferSize = _read_int32(f)
            f.seek(header.unCompressedPixelBufferSize, os.SEEK_CUR)

        # Histogram
        header.binsInHistogram = _read_int32(f)
        f.seek(header.binsInHistogram * 4, os.SEEK_CUR)

        # Properties
        header.numberOfProperties = _read_int32(f)

        for _ in range(header.numberOfProperties):
            name_len = _read_int32(f)
            if name_len > PROPERTY_NAME_MAX_LENGTH:
                raise ValueError(f"Property name too long: {name_len}")

            name_bytes = _read_exact(f, name_len)
            prop_name = name_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")

            prop_type = _read_int32(f)

            # Decode according to type
            if prop_type == 0:
                # uint32 / Int4 (single)
                value = _read_int32(f)
                properties[prop_name] = value
                _apply_property(header, prop_name, value)

            elif prop_type == 1:
                # double (single)
                value = _read_double(f)
                properties[prop_name] = value
                _apply_property(header, prop_name, value)

            elif prop_type == 2:
                # length * char (string / raw bytes)
                val_len = _read_int32(f)
                raw = _read_exact(f, val_len)
                # Try to decode as ASCII, but keep bytes if that fails
                try:
                    value = raw.split(b"\x00", 1)[0].decode("ascii")
                except UnicodeDecodeError:
                    value = raw
                properties[prop_name] = value

            elif prop_type == 4:
                # length * double array
                val_len = _read_int32(f)      # number of BYTES
                n_doubles = val_len // 8
                raw = _read_exact(f, val_len)
                doubles = list(struct.unpack("<" + "d" * n_doubles, raw))
                properties[prop_name] = doubles
                # optional: if you wanted to map some array-based property,
                # you could call _apply_property on doubles[0] here.

            elif prop_type == 5:
                # length * uint32 array
                val_len = _read_int32(f)      # number of BYTES
                n_ints = val_len // 4
                raw = _read_exact(f, val_len)
                ints = list(struct.unpack("<" + "I" * n_ints, raw))
                properties[prop_name] = ints

            else:
                raise ValueError(
                    f"Unsupported property type {prop_type} for property '{prop_name}'"
                )

    return header, image_data_start, properties



# -------------------------------------------------------------------------
# can_read_xim (CanReadFile)
# -------------------------------------------------------------------------


def can_read_xim(path: str) -> bool:
    """
    Rough equivalent of rtk::XimImageIO::CanReadFile.
    Checks extension and basic header fields.
    """
    if not path.lower().endswith(".xim"):
        return False

    if not os.path.exists(path):
        return False

    try:
        with open(path, "rb") as f:
            sfiletype = _read_exact(f, 8)
            fileversion = _read_int32(f)
            sizex = _read_int32(f)
            sizey = _read_int32(f)
    except Exception:
        return False

    if sizex * sizey <= 0:
        return False

    # sfiletype and fileversion are not deeply checked in the original;
    # they just care that we can read them and that sizes are > 0.
    return True


# -------------------------------------------------------------------------
# Image reading (Read)
# -------------------------------------------------------------------------


def read_xim_image(path: str) -> Tuple[np.ndarray, XimHeader, Dict[str, Any]]:
    """
    Read XIM image using the same LUT/differential decoding as rtk::XimImageIO::Read.

    Returns
    -------
    image : np.ndarray, shape (SizeY, SizeX), dtype=np.int32
        Decompressed image data.
    header : XimHeader
    meta : dict
        Dictionary with a few important meta values
        ('dCTProjectionAngle', 'dDetectorOffsetX_mm', 'dDetectorOffsetY_mm').
    """
    header, image_data_start, properties = read_xim_header(path)

    xdim = header.SizeX
    ydim = header.SizeY

    if xdim * ydim == 0:
        # Match the "empty image" behaviour conceptually
        raise ValueError(f"Image dimensions are zero in {path}")

    if header.dCompressionIndicator != 1:
        # The C++ Read() implementation really only handles the compressed case.
        # Here we mimic that and explicitly fail on uncompressed for clarity.
        raise NotImplementedError(
            "Only compressed XIM files (dCompressionIndicator==1) are supported."
        )

    with open(path, "rb") as f:
        # Seek to the same place as C++: m_ImageDataStart
        f.seek(image_data_start, os.SEEK_SET)

        lookUpTableSize = _read_int32(f)
        lookup_table = np.frombuffer(
            _read_exact(f, lookUpTableSize), dtype=np.uint8
        ).astype(np.uint8)

        compressedPixelBufferSize = _read_int32(f)
        # NOTE: the C++ code does NOT use compressedPixelBufferSize directly,
        # it recomputes how many bytes should be there from the LUT. We mimic that.

        # First row + 1 (Int4)
        buf = np.empty(xdim * ydim + 1, dtype=np.int32)
        n_first = xdim + 1
        data_first = _read_exact(f, n_first * 4)
        buf[:n_first] = np.frombuffer(data_first, dtype="<i4")

        # Transform LUT into byte counts per 4-diff pack
        byte_table = []
        for v in lookup_table:
            b0 = _lut_to_bytes(v & 0b00000011)       # 0x03
            b1 = _lut_to_bytes((v & 0b00001100) >> 2)  # 0x0C
            b2 = _lut_to_bytes((v & 0b00110000) >> 4)  # 0x30
            b3 = _lut_to_bytes((v & 0b11000000) >> 6)  # 0xC0
            byte_table.extend([b0, b1, b2, b3])

        total_bytes = sum(byte_table)

        # total_bytes - 3 because last two bits can be redundant (per C++ comment)
        compr_img_buffer = _read_exact(f, total_bytes)
        if len(compr_img_buffer) < total_bytes - 3:
            raise IOError(
                f"Could not read image buffer of XIM file: {path} "
                f"(expected at least {total_bytes-3}, got {len(compr_img_buffer)})"
            )

        # Now do the differential decoding similar to C++ Read()
        j = 0  # index into compr_img_buffer
        i = xdim  # current image index (we already have first row+1)
        iminxdim = 0  # index i - xdim (row above), stepping by 4

        # We iterate on the original lookup_table; every byte encodes 4 diffs
        for lut_idx, v in enumerate(lookup_table):
            # 4 diffs per table entry
            diffs = []
            for shift in (0, 2, 4, 6):
                n_bytes = _lut_to_bytes((v >> shift) & 0b00000011)
                if n_bytes not in (1, 2, 4):
                    raise ValueError(
                        f"Unsupported byte count ({n_bytes}) in LUT at index {lut_idx}"
                    )
                chunk = compr_img_buffer[j : j + n_bytes]
                if len(chunk) != n_bytes:
                    raise IOError(
                        f"Truncated compressed buffer at j={j}, "
                        f"wanted {n_bytes}, got {len(chunk)}"
                    )
                j += n_bytes
                diffs.append(_cast_binary_char_to_int(chunk, n_bytes))

            diff1, diff2, diff3, diff4 = diffs

            # Same recurrence as in C++
            buf[i + 1] = np.int32(
                diff1 + buf[i] + buf[iminxdim + 1] - buf[iminxdim]
            )
            buf[i + 2] = np.int32(
                diff2 + buf[i + 1] + buf[iminxdim + 2] - buf[iminxdim + 1]
            )
            buf[i + 3] = np.int32(
                diff3 + buf[i + 2] + buf[iminxdim + 3] - buf[iminxdim + 2]
            )
            if i + 4 < xdim * ydim:
                buf[i + 4] = np.int32(
                    diff4 + buf[i + 3] + buf[iminxdim + 4] - buf[iminxdim + 3]
                )

            i += 4
            iminxdim += 4

        # Sanity-ish checks (mirroring the C++ asserts)
        if j > len(compr_img_buffer):
            raise AssertionError("Read past end of compressed buffer")
        if i != xdim * ydim:
            raise AssertionError(
                f"Decoded pixels mismatch: i={i}, expected {xdim*ydim}"
            )

    # Drop the sentinel at index 0; reshape to (Y, X)
    img = buf[1 : xdim * ydim + 1].reshape(ydim, xdim)

    # Convert numpy array to SimpleITK image
    sitk_img = sitk.GetImageFromArray(img)

    # Optionally set spacing if available in properties or header
    spacing_x = float(header.dIDUResolutionX) if hasattr(header, 'dIDUResolutionX') else 1.0
    spacing_y = float(header.dIDUResolutionY) if hasattr(header, 'dIDUResolutionY') else 1.0
    sitk_img.SetSpacing((spacing_x, spacing_y))

    # Return SimpleITK image, header, and all metadata tags (properties)
    return sitk_img, header, properties
