import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import SimpleITK as sitk

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
_DEFAULT_MODEL_PATH = _CONFIGS_DIR / "M730_2_Layers.pt"
_DEFAULT_PARAMETER_MAP_PATH = _CONFIGS_DIR / "ParameterMap_deformable.txt"


def _clip_and_standardize_ct(
    image: sitk.Image,
    clip_min: float,
    clip_max: float,
    mean: float,
    std: float,
) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    data = data.clip(clip_min, clip_max)
    data = (data - mean) / std
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result


def _standardize_z_score(image: sitk.Image) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    data = (data - data.mean()) / data.std()
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result


def _setup_data_dir(
    data_dir: Path,
    model_path: Path,
    models_dest_dir: str,
    parameter_map_path: Path,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)

    if model_path is not None and model_path.exists():
        dest_dir = data_dir / models_dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(model_path), str(dest_dir / model_path.name))

    if parameter_map_path is not None and parameter_map_path.exists():
        shutil.copy(str(parameter_map_path), str(data_dir / "ParameterMap.txt"))

    return data_dir


def _run_docker_elastix(
    data_dir: Path,
    out_dir: Path,
    docker_image: str,
    elastix_binary: str,
    threads: int,
    fixed_mask_path: Optional[Path],
    moving_mask_path: Optional[Path],
    verbose: bool,
) -> None:
    cmd_parts = [
        "docker run --rm --gpus all",
        f"-v {data_dir.absolute()}:/Data",
        f"-v {out_dir.absolute()}:/Out",
        docker_image,
        elastix_binary,
        "-f /Data/Fixed_image.mha",
        "-m /Data/Moving_image.mha",
        "-p /Data/ParameterMap.txt",
        "-out /Out",
        f"-threads {threads}",
    ]

    if fixed_mask_path is not None:
        cmd_parts.append("-fMask /Data/FixedMask.mha")
    if moving_mask_path is not None:
        cmd_parts.append("-mMask /Data/MovingMask.mha")

    cmd = " ".join(cmd_parts)
    if verbose:
        print(f"Running: {cmd}")

    result = os.system(cmd)
    if result != 0:
        raise RuntimeError(f"Elastix Docker command failed with return code {result}")


def _apply_deformation(image: sitk.Image, transform_file: Path) -> sitk.Image:
    transform = sitk.TransformixImageFilter()
    param_map = transform.ReadParameterFile(str(transform_file.absolute()))
    transform.SetTransformParameterMap(param_map)
    transform.SetMovingImage(image)
    transform.SetLogToConsole(False)
    transform.SetLogToFile(False)
    transform.Execute()
    result = transform.GetResultImage()
    return sitk.Cast(result, sitk.sitkInt16)


def elastix_impact_pt(
    img_fixed: sitk.Image,
    img_moving: sitk.Image,
    model_path: Path = _DEFAULT_MODEL_PATH,
    parameter_map_path: Path = _DEFAULT_PARAMETER_MAP_PATH,
    models_dest_dir: str = "Models/TS",
    persistent_data_dir: Path = Path("/tmp/elastix_impact_data"),
    threads: int = 24,
    docker_image: str = "elastix_impact:latest",
    elastix_binary: str = "/opt/elastix-install/bin/elastix",
    clip_min: float = -1024.0,
    clip_max: float = 276.0,
    standardize_mean: float = -370.0,
    standardize_std: float = 436.6,
    use_fixed_mask: bool = False,
    fixed_mask_path: Optional[Path] = None,
    use_moving_mask: bool = False,
    moving_mask_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[sitk.Image, Path]:
    """Run IMPACT+Elastix deformable registration on sitk.Image inputs.

    img_fixed is treated as CBCT (z-score normalised before registration).
    img_moving is treated as CT (clip+standardise before registration).

    Returns:
        (deformed_moving, transform_file_path)
    """
    data_dir = _setup_data_dir(
        persistent_data_dir, model_path, models_dest_dir, parameter_map_path
    )

    # Preprocess and write images
    fixed_std = _standardize_z_score(img_fixed)
    moving_std = _clip_and_standardize_ct(
        img_moving, clip_min, clip_max, standardize_mean, standardize_std
    )
    sitk.WriteImage(fixed_std, str(data_dir / "Fixed_image.mha"))
    sitk.WriteImage(moving_std, str(data_dir / "Moving_image.mha"))

    # Copy masks into data_dir so Docker can mount them
    docker_fixed_mask = None
    docker_moving_mask = None
    if use_fixed_mask and fixed_mask_path is not None and Path(fixed_mask_path).exists():
        shutil.copy(str(fixed_mask_path), str(data_dir / "FixedMask.mha"))
        docker_fixed_mask = data_dir / "FixedMask.mha"
    if use_moving_mask and moving_mask_path is not None and Path(moving_mask_path).exists():
        shutil.copy(str(moving_mask_path), str(data_dir / "MovingMask.mha"))
        docker_moving_mask = data_dir / "MovingMask.mha"

    out_dir = Path(tempfile.mkdtemp(prefix="elastix_out_"))
    try:
        _run_docker_elastix(
            data_dir=data_dir,
            out_dir=out_dir,
            docker_image=docker_image,
            elastix_binary=elastix_binary,
            threads=threads,
            fixed_mask_path=docker_fixed_mask,
            moving_mask_path=docker_moving_mask,
            verbose=verbose,
        )

        transform_file = out_dir / "TransformParameters.0.txt"
        if not transform_file.exists():
            raise FileNotFoundError(
                "TransformParameters.0.txt not generated by elastix"
            )

        # Copy transform to persistent location so it survives cleanup
        saved_transform = data_dir / "TransformParameters.0.txt"
        shutil.copy(str(transform_file), str(saved_transform))

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    deformed = _apply_deformation(img_moving, saved_transform)
    return deformed, saved_transform


def elastix_impact(
    path_img_fixed: Union[Path, str],
    path_img_moving: Union[Path, str],
    model_path: Path = _DEFAULT_MODEL_PATH,
    parameter_map_path: Path = _DEFAULT_PARAMETER_MAP_PATH,
    models_dest_dir: str = "Models/TS",
    persistent_data_dir: Path = Path("/tmp/elastix_impact_data"),
    threads: int = 24,
    docker_image: str = "elastix_impact:latest",
    elastix_binary: str = "/opt/elastix-install/bin/elastix",
    clip_min: float = -1024.0,
    clip_max: float = 276.0,
    standardize_mean: float = -370.0,
    standardize_std: float = 436.6,
    use_fixed_mask: bool = False,
    fixed_mask_path: Optional[Path] = None,
    use_moving_mask: bool = False,
    moving_mask_path: Optional[Path] = None,
    result_path: Union[Path, str] = "./",
    verbose: bool = False,
) -> None:
    """File-path-based wrapper around elastix_impact_pt.

    Loads images, runs registration, saves ct_deformed.mha and
    TransformParameters.0.txt to result_path.
    """
    img_fixed = sitk.ReadImage(str(path_img_fixed))
    img_moving = sitk.ReadImage(str(path_img_moving))

    deformed, transform_file = elastix_impact_pt(
        img_fixed=img_fixed,
        img_moving=img_moving,
        model_path=model_path,
        parameter_map_path=parameter_map_path,
        models_dest_dir=models_dest_dir,
        persistent_data_dir=persistent_data_dir,
        threads=threads,
        docker_image=docker_image,
        elastix_binary=elastix_binary,
        clip_min=clip_min,
        clip_max=clip_max,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
        use_fixed_mask=use_fixed_mask,
        fixed_mask_path=fixed_mask_path,
        use_moving_mask=use_moving_mask,
        moving_mask_path=moving_mask_path,
        verbose=verbose,
    )

    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(deformed, str(result_path / "ct_deformed.mha"))
    shutil.copy(str(transform_file), str(result_path / "TransformParameters.0.txt"))

    if verbose:
        print(f"Saved ct_deformed.mha and TransformParameters.0.txt to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IMPACT+Elastix deformable registration"
    )
    parser.add_argument("-f", "--path_img_fixed", type=str, required=True)
    parser.add_argument("-m", "--path_img_moving", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=str(_DEFAULT_MODEL_PATH))
    parser.add_argument("--parameter_map_path", type=str, default=str(_DEFAULT_PARAMETER_MAP_PATH))
    parser.add_argument("--models_dest_dir", type=str, default="Models/TS")
    parser.add_argument(
        "--persistent_data_dir", type=str, default="/tmp/elastix_impact_data"
    )
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--docker_image", type=str, default="nvcr.io/muwsc/radonc/elastix_impact:latest")
    parser.add_argument(
        "--elastix_binary", type=str, default="/opt/elastix-install/bin/elastix"
    )
    parser.add_argument("--result_path", type=str, default="./")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    elastix_impact(
        path_img_fixed=args.path_img_fixed,
        path_img_moving=args.path_img_moving,
        model_path=Path(args.model_path),
        parameter_map_path=Path(args.parameter_map_path),
        models_dest_dir=args.models_dest_dir,
        persistent_data_dir=Path(args.persistent_data_dir),
        threads=args.threads,
        docker_image=args.docker_image,
        elastix_binary=args.elastix_binary,
        result_path=args.result_path,
        verbose=args.verbose,
    )
