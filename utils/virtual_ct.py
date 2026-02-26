import SimpleITK as sitk
import numpy as np
from typing import Optional
from pathlib import Path
import sys

class VirtualCTCreator:
    def __init__(
        self,
        air_threshold_hu: float = -200,
        blend_margin_mm: float = 3.0,
        body_threshold_hu: float = -300,
        correct_cbct_before_virtual_ct: bool = False,
        sct_model_path: Optional[str] = None,
        sct_device: str = "cuda",
        sct_apply_normalization: bool = True,
        sct_apply_denormalization: bool = True,
    ):
        self.air_threshold_hu = air_threshold_hu
        self.blend_margin_mm = blend_margin_mm
        self.body_threshold_hu = body_threshold_hu
        self.correct_cbct_before_virtual_ct = correct_cbct_before_virtual_ct
        self.sct_apply_normalization = sct_apply_normalization
        self.sct_apply_denormalization = sct_apply_denormalization

        self._sct_predictor = None
        if self.correct_cbct_before_virtual_ct:
            if sct_model_path is None:
                raise ValueError(
                    "sct_model_path must be provided when "
                    "correct_cbct_before_virtual_ct=True"
                )
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.sct_generator import StandaloneRegressionInference

            self._sct_predictor = StandaloneRegressionInference(
                model_path=sct_model_path,
                device=sct_device,
            )

    def _predict_corrected_cbct(self, cbct: sitk.Image) -> sitk.Image:
        if self._sct_predictor is None:
            raise RuntimeError("sCT predictor is not initialized")

        cbct_arr = sitk.GetArrayFromImage(cbct)
        corrected_arr = self._sct_predictor.predict(
            input_array=cbct_arr,
            apply_normalization=self.sct_apply_normalization,
            apply_denormalization=self.sct_apply_denormalization,
        )
        corrected_cbct = sitk.GetImageFromArray(corrected_arr.astype(np.int16))
        corrected_cbct.CopyInformation(cbct)
        return corrected_cbct

    def create(
        self,
        deformed_ct: sitk.Image,
        cbct: sitk.Image
    ) -> sitk.Image:
        """Build a virtual CT by blending CT and CBCT in mismatch low-density regions."""
        deformed_ct_arr = sitk.GetArrayFromImage(deformed_ct)

        if self.correct_cbct_before_virtual_ct:
            cbct_for_blending = self._predict_corrected_cbct(cbct)
        else:
            cbct_for_blending = cbct

        cbct_arr = sitk.GetArrayFromImage(cbct_for_blending)


        # Compute body masks for both CT and CBCT
        body_mask_ct = deformed_ct_arr > self.body_threshold_hu

        # Fill holes in the intersection mask (optional, for robustness)
        body_mask_ct_sitk = sitk.GetImageFromArray(body_mask_ct.astype(np.uint8))
        body_mask_ct_sitk.CopyInformation(cbct_for_blending)
        body_filled_ct_sitk = sitk.BinaryFillhole(body_mask_ct_sitk)
        body_filled_ct_sitk = sitk.BinaryErode(body_filled_ct_sitk, [5, 5, 0])

        body_region_arr = sitk.GetArrayFromImage(body_filled_ct_sitk) > 0

        low_density_cbct = cbct_arr < self.air_threshold_hu
        low_density_deformed = deformed_ct_arr < self.air_threshold_hu
        mismatch_mask = (low_density_cbct != low_density_deformed) & body_region_arr

        mismatch_image = sitk.GetImageFromArray(mismatch_mask.astype(np.uint8))
        mismatch_image.CopyInformation(cbct_for_blending)
        dilate_radius = int(self.blend_margin_mm / cbct_for_blending.GetSpacing()[0])
        dilated = sitk.BinaryDilate(mismatch_image, [dilate_radius] * 3)

        blend_mask = sitk.Cast(dilated, sitk.sitkFloat32)
        blend_mask = sitk.SmoothingRecursiveGaussian(blend_mask, self.blend_margin_mm)
        blend_mask_arr = np.clip(sitk.GetArrayFromImage(blend_mask), 0, 1)

        blend_mask_arr = blend_mask_arr * body_region_arr
        virtual_ct_arr = (1 - blend_mask_arr) * deformed_ct_arr + blend_mask_arr * cbct_arr

        virtual_ct = sitk.GetImageFromArray(virtual_ct_arr.astype(np.int16))
        virtual_ct.CopyInformation(cbct_for_blending)
        return virtual_ct, cbct_for_blending


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run VirtualCTCreator on one CT/CBCT pair.")
    parser.add_argument("--cbct", required=True, type=str, help="Path to CBCT image")
    parser.add_argument("--ct", required=True, type=str, help="Path to (deformed) CT image")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to sCT model bundle directory containing model.pt and metadata.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for virtual CT image (default: <cbct_dir>/virtual_ct.mha)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for sCT model (e.g. cuda or cpu)")
    parser.add_argument("--air-threshold-hu", type=float, default=-200)
    parser.add_argument("--blend-margin-mm", type=float, default=3.0)
    parser.add_argument("--body-threshold-hu", type=float, default=-300)
    return parser.parse_args()



if __name__ == "__main__":
    
    import os
    args = parse_args()

    cbct = sitk.ReadImage(args.cbct)
    deformed_ct = sitk.ReadImage(args.ct)

    creator = VirtualCTCreator(
        air_threshold_hu=args.air_threshold_hu,
        blend_margin_mm=args.blend_margin_mm,
        body_threshold_hu=args.body_threshold_hu,
        correct_cbct_before_virtual_ct=True,
        sct_model_path=args.checkpoint,
        sct_device=args.device,
    )

    virtual_ct, cbct_for_blending = creator.create(deformed_ct=deformed_ct, cbct=cbct)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.cbct), "virtual_ct.mha")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sitk.WriteImage(virtual_ct, output_path, useCompression=True)
    sitk.WriteImage(cbct_for_blending, output_path.replace(".mha", "_cbct_for_blending.mha"), useCompression=True)
    print(f"Saved virtual CT to: {output_path}")

