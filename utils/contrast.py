"""Standalone contrast-media correction.

This is a copy of the contrast-media correction implemented inside
``simcbctgenerator.simulation.standard.StandardCBCTSimulator`` (the
``_apply_contrast_media_correction`` method and the surrounding mask-handling
logic in ``run``). It is extracted here so the correction can be tested on a
regular CT image without running the full projection simulation. The
simcbctgenerator library itself is left untouched.

The original implementation requires cupy. Here cupy is used when available and
the code transparently falls back to a numpy/scipy CPU implementation, so it can
be exercised on machines without a GPU.
"""

import logging
from typing import Optional

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def apply_contrast_media_correction(
    ct_image: sitk.Image,
    cm_mask_image: sitk.Image,
) -> sitk.Image:
    """Reduce contrast-media enhancement inside ``cm_mask_image``.

    Faithful port of
    ``StandardCBCTSimulator._apply_contrast_media_correction``: within the mask
    (and only where HU > 50), a blurred, lightly noised fraction (0.92) of the
    CT intensity is subtracted to suppress the bright contrast agent.

    Parameters
    ----------
    ct_image:
        The CT to correct (HU).
    cm_mask_image:
        Binary mask marking the region containing contrast media (e.g. bowel).
        Resampled onto the CT grid with nearest-neighbour if sizes differ.

    Returns
    -------
    sitk.Image
        The corrected CT, sharing the input CT's geometry.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

        xp = cp
        gaussian_filter = cp_gaussian_filter
        to_numpy = cp.asnumpy
    except ImportError:
        from scipy.ndimage import gaussian_filter as np_gaussian_filter

        xp = np
        gaussian_filter = np_gaussian_filter
        to_numpy = np.asarray
        logger.info("cupy not available; using numpy/scipy CPU fallback for "
                    "contrast-media correction.")

    if cm_mask_image.GetSize() != ct_image.GetSize():
        cm_mask_image = sitk.Resample(
            cm_mask_image, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            cm_mask_image.GetPixelID()
        )

    ct_arr = xp.asarray(sitk.GetArrayFromImage(ct_image).astype(np.float32))
    mask_arr = xp.asarray(sitk.GetArrayFromImage(cm_mask_image).astype(np.uint8))
    cm = mask_arr.astype(bool) & (ct_arr > 50)
    blurred = gaussian_filter(cm.astype(xp.float32), (0.0, 1.0, 1.0), mode="constant", cval=0)
    noise = xp.random.normal(loc=1.0, scale=0.02, size=cm.shape)
    corrected = sitk.GetImageFromArray(to_numpy(ct_arr - blurred * ct_arr * noise * 0.92))
    corrected.CopyInformation(ct_image)
    return corrected


def correct_ct_contrast_media(
    ct_image: sitk.Image,
    cm_mask_image: Optional[sitk.Image] = None,
    segment_bowel: bool = True,
) -> sitk.Image:
    """Correct a CT for contrast media, segmenting the bowel if no mask is given.

    Mirrors the contrast-media branch of ``StandardCBCTSimulator.run``: if no
    ``cm_mask_image`` is supplied and ``segment_bowel`` is True, a bowel mask is
    generated with TotalSegmentator (via the library's ``OrganMaskGenerator``)
    and used as the contrast-media region.

    Parameters
    ----------
    ct_image:
        The CT to correct (HU).
    cm_mask_image:
        Optional binary contrast-media mask. If None, the bowel is segmented.
    segment_bowel:
        When True and no mask is given, run TotalSegmentator to obtain a bowel
        mask. When False and no mask is given, a ValueError is raised.

    Returns
    -------
    sitk.Image
        The corrected CT.
    """
    if cm_mask_image is None:
        if not segment_bowel:
            raise ValueError("No cm_mask_image provided and segment_bowel=False.")
        logger.info("Segmenting bowel for contrast-media correction via TotalSegmentator")
        from simcbctgenerator.organ_mask_generator import OrganMaskGenerator
        gen = OrganMaskGenerator(fast_mode=True, device="gpu")
        cm_mask_image = gen.generate_multi_organ_masks(ct_image, ["bowel"])["bowel"]

    return apply_contrast_media_correction(ct_image, cm_mask_image)
