import os
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend
from matplotlib import pyplot as plt
import utils.img as img
from scipy import ndimage
from typing import List, Literal

logger = logging.getLogger(__name__)

def format_metadata(md: dict, patient_ID: str) -> str:
    lines = []
    lines.append(f"Patient ID: {patient_ID}")
    lines.append(f"Center: A")
    lines.append("")
    lines.append("CT:")
    lines.append(f"  Manufacturer: {md.get('ct', {}).get('Manufacturer', 'N/A')}")
    lines.append(f"  Model: {md.get('ct', {}).get('ManufacturerModelName', 'N/A')}")
    lines.append(f"  kVp: {md.get("ct", {}).get("KVP", None)} kV")
    lines.append(f"  mA: {md.get("ct", {}).get("XRayTubeCurrent", None)} mA")
    lines.append(f"  Exposure Time: {md.get("ct", {}).get("ExposureTime", None)} ms")
    lines.append(f"  Exposure: {md.get("ct", {}).get("Exposure", None)} mAs")
    lines.append(f"  CTDIvol: {md.get("ct", {}).get("CTDIvol", None):.2f}")
    lines.append(f"  DataCollectionDiameter: {md.get("ct", {}).get("DataCollectionDiameter", None)} mm")
    lines.append(f"  Reconstruction Diameter: {md.get("ct", {}).get("ReconstructionDiameter", None)} mm")
    lines.append(f"  Slice Thickness: {md.get("ct", {}).get("SliceThickness", None)} mm")
    lines.append(f"  Pixel Spacing: {md.get("ct", {}).get("PixelSpacing", None)} mm")
    lines.append(f"  Rows x Columns: {md.get("ct", {}).get("Rows", None)} x {md.get("ct", {}).get("Columns", None)}")
    lines.append("")
    lines.append("CBCT:")
    lines.append(f"  Manufacturer: {md.get('cbct', {}).get('Manufacturer', 'N/A')}")
    lines.append(f"  Model: ")
    lines.append(f"  kVp: {md.get("cbct", {}).get("TubeVoltage", None)} kV")
    lines.append(f"  mA: {md.get("cbct", {}).get("TubeCurrent", None)} mA")
    lines.append(f"  Exposure Time: {md.get("cbct", {}).get("PulseLength", None)} ms")
    lines.append(f"  Exposure: {md.get("cbct", {}).get("TubeCurrent", None)*md.get("cbct", {}).get("PulseLength", None)/1000} mAs")
    lines.append(f"  Frames: {md.get("cbct", {}).get("Frames", None)}")
    lines.append(f"  Projection Spacing: {md.get("cbct", {}).get("ImagerResX", None)} mm x {md.get("cbct", {}).get("ImagerResY", None)} mm")
    lines.append(f"  Projection Size: {md.get("cbct", {}).get("ImagerSizeX", None)} x {md.get("cbct", {}).get("ImagerSizeY", None)}")
    lines.append(f"  Trajectory: {md.get("cbct", {}).get("Trajectory", 'N/A')}")
    lines.append(f"  Fan: {md.get("cbct", {}).get("Fan", None)}")
    lines.append(f"  Start Angle: {md.get("cbct", {}).get("StartAngle", None):.2f} degrees")
    lines.append(f"  Stop Angle: {md.get("cbct", {}).get("StopAngle", None):.2f} degrees")
    lines.append(f"  Recon. Spacing: {md.get("cbct", {}).get("ReconstructionSpacingX", None):.2f} x {md.get("cbct", {}).get("ReconstructionSpacingY", None):.2f} x {md.get("cbct", {}).get("ReconstructionSpacingZ", None):.2f} mm")
    lines.append(f"  Recon. Size: {md.get("cbct", {}).get("ReconstructionSizeX", None)} x {md.get("cbct", {}).get("ReconstructionSizeY", None)} x {md.get("cbct", {}).get("ReconstructionSizeZ", None)}")
    lines.append(f"  Antiscatter Grid: {md.get("cbct", {}).get("ScatterGrid", 'N/A')}")
    lines.append(f"  Filter Type: {md.get("cbct", {}).get("KVFilter", 'N/A')}")
    lines.append(f"  Collimator: {md.get("cbct", {}).get("Collimator", 'N/A')}")
    lines.append(f"  FOV Type: {md.get("cbct", {}).get("FOV", 'N/A')}")
    lines.append(f"")
    return "\n".join(lines)

def generate_overview(
    cbct_clinical: sitk.Image | None,
    cbct_rtk: sitk.Image | None,
    ct: sitk.Image | None,
    output_path: str,
    patient_ID: str,
    metadata: dict,
) -> None:
    """
    Generate an overview PNG image showing slices from different orientations of CT, CBCT clinical, and CBCT RTK.
    """
    ct = sitk.Resample(ct, cbct_clinical)
    cbct_rtk = sitk.Resample(cbct_rtk, cbct_clinical)

    ct[ct < -1024] = -1024
    cbct_clinical[cbct_clinical < -1024] = -1024
    cbct_rtk[cbct_rtk < -1024] = -1024
    
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)

    background_clinical = np.percentile(sitk.GetArrayFromImage(cbct_clinical), 0.1)
    high_clinical = np.percentile(sitk.GetArrayFromImage(cbct_clinical), 99.9)

    background_rtk = np.percentile(sitk.GetArrayFromImage(cbct_rtk), 0.1)
    high_rtk = np.percentile(sitk.GetArrayFromImage(cbct_rtk), 99.9)

    arr = sitk.GetArrayFromImage(ct)
    z, y, x = arr.shape
    sx, sy, sz = ct.GetSpacing()

    Lx = x * sx
    Ly = y * sy
    Lz = z * sz

    slice_sag = x // 2
    slice_cor = y // 2
    slice_ax  = z // 2

    cor_len = Ly
    sag_len = Lx
    ax_len  = Lz

    # Calculate Aspect Ratios
    height_ratios = [cor_len / cor_len, (ax_len / cor_len) * (sag_len / cor_len), ax_len / cor_len]

    # Update width calculation for 4 columns instead of 3
    # We treat the text column as having the same relative width as an image column (approx)
    x_len = 4 * sag_len / cor_len 
    y_len = np.sum(height_ratios) / x_len

    # Update gridspec to have 4 columns
    # The last 1.2 gives the text column slightly more breathing room
    gridspec_kw = {'width_ratios': [1, 1, 1, 1.2], 'height_ratios': height_ratios}

    size = 20
    # Change subplots to (3, 4)
    fig, ax = plt.subplots(3, 4, figsize=(size, y_len * size), gridspec_kw=gridspec_kw)

    # --- Image Plotting (Columns 0, 1, 2) ---

    # Row 0 (Axial)
    ax[0,0].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:], cmap='gray', vmin=background_ct, vmax=high_ct, aspect = sy/sx)
    ax[0,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[slice_ax,:,:], cmap='gray', vmin=background_clinical, vmax=high_clinical, aspect = sy/sx)
    ax[0,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[slice_ax,:,:], cmap='gray', vmin=background_rtk, vmax=high_rtk, aspect = sy/sx)

    # Row 1 (Sagittal)
    ax[1,0].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag], cmap='gray', vmin=background_ct, vmax=high_ct, aspect = sz/sy)
    ax[1,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[::-1,:,slice_sag], cmap='gray', vmin=background_clinical, vmax=high_clinical, aspect = sz/sy)
    ax[1,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[::-1,:,slice_sag], cmap='gray', vmin=background_rtk, vmax=high_rtk, aspect = sz/sy)

    # Row 2 (Coronal)
    ax[2,0].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:], cmap='gray', vmin=background_ct, vmax=high_ct, aspect = sz/sx)
    ax[2,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[::-1,slice_cor,:], cmap='gray', vmin=background_clinical, vmax=high_clinical, aspect = sz/sx)
    ax[2,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[::-1,slice_cor,:], cmap='gray', vmin=background_rtk, vmax=high_rtk, aspect = sz/sx)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    if metadata is not {}:
        metadata_text = format_metadata(metadata, patient_ID)

        fig.text(
            0.73, 0.95,
            metadata_text,
            fontsize=13,
            va='top',
            ha='left',
            family='monospace',   # makes it easier to read structured info
            bbox=dict(
                facecolor='white',
                alpha=0.95,
                edgecolor='lightgray',
                boxstyle='round,pad=0.6'
            )
        )
    
    # --- Helper Functions ---
    def add_text(ax, text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    def add_patient(ax, text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

    for r, ax_row in enumerate(ax):
        for c, a in enumerate(ax_row):
            # Turn off ticks for everyone
            a.set_xticks([])
            a.set_yticks([])
            
            # Columns 0-2: Standard Labels
            if c == 0:
                add_text(a, 'CT')
                add_patient(a, patient_ID)
                a.set_ylabel('Axial' if r == 0 else 'Sagittal' if r == 1 else 'Coronal', fontsize=12, fontweight='bold')
            elif c == 1:
                add_text(a, 'CBCT clinical')
                add_patient(a, patient_ID)
            elif c == 2:
                add_text(a, 'CBCT rtk')
                add_patient(a, patient_ID)
            elif c == 3:
                a.axis('off')  # No axis for text column

    plt.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)



VisMode = Literal["raw", "diff", "checkerboard", "fusion", "edges"]

VIS_DEFAULTS: List[VisMode] = ["raw", "diff", "checkerboard"]


def _sobel_edges(slice_2d: np.ndarray) -> np.ndarray:
    sx = ndimage.sobel(slice_2d.astype(float), axis=0)
    sy = ndimage.sobel(slice_2d.astype(float), axis=1)
    mag = np.hypot(sx, sy)
    if mag.max() > 0:
        mag /= mag.max()
    return mag


def _make_checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    rows, cols = a.shape
    row_idx = (np.arange(rows) // tile) % 2
    col_idx = (np.arange(cols) // tile) % 2
    mask = row_idx[:, None] ^ col_idx[None, :]
    return np.where(mask, b, a)


def _color_fusion(a: np.ndarray, b: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    a_n = np.clip((a - vmin) / (vmax - vmin + 1e-8), 0, 1)
    b_n = np.clip((b - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgb = np.zeros((*a.shape, 3))
    rgb[..., 0] = a_n
    rgb[..., 1] = b_n
    rgb[..., 2] = a_n
    return rgb


def _linear_rescale(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s_mean, s_std = src[mask].mean(), src[mask].std()
    r_mean, r_std = ref[mask].mean(), ref[mask].std()
    return (src - s_mean) / (s_std + 1e-8) * r_std + r_mean


def generate_overview_deformed(
    cbct_clinical: sitk.Image,
    ct_deformed: sitk.Image,
    output_path: str,
    patient_ID: str,
    metadata: dict | None = None,
    fov: sitk.Image | None = None,
    vis_modes: List[VisMode] | None = None,
    checkerboard_tile: int = 32,
    diff_range: float = 500,
    image2_label: str = "CT deformed",
) -> None:
    """
    Generate an overview PNG comparing CBCT clinical and deformed CT.

    Parameters
    ----------
    cbct_clinical : sitk.Image
        The CBCT clinical image (reference grid).
    ct_deformed : sitk.Image
        The deformed CT, resampled onto the CBCT grid.
    output_path : str
        Path to save the overview PNG.
    patient_ID : str
        Patient identifier shown on each panel.
    metadata : dict, optional
        Metadata to display in a text box.
    fov : sitk.Image, optional
        Field-of-view mask for the CBCT. If None, the full volume is used.
    vis_modes : list of VisMode, optional
        Which panels to include. Any combination of:
        "raw"          – CBCT clinical / CT deformed side by side (2 cols)
        "diff"         – FOV-masked difference map (1 col)
        "checkerboard" – checkerboard overlay (1 col)
        "fusion"       – magenta/green colour fusion (1 col)
        "edges"        – CBCT with CT Sobel edge overlay (1 col)
        "histogram"    – difference histogram (separate file)
        Default: all of the above.
    checkerboard_tile : int
        Tile size in pixels for the checkerboard.
    diff_range : float
        Symmetric vmin/vmax for the difference colour map.
    """
    if vis_modes is None:
        vis_modes = list(VIS_DEFAULTS)

    # ── Resample CT deformed onto CBCT grid ──────────────────────────────
    ct_res = sitk.Resample(ct_deformed, cbct_clinical)

    ct_res[ct_res < -1024] = -1024
    cbct_clinical[cbct_clinical < -1024] = -1024

    cbct_arr = sitk.GetArrayFromImage(cbct_clinical).astype(float)
    ct_arr = sitk.GetArrayFromImage(ct_res).astype(float)

    fov_arr = (
        sitk.GetArrayFromImage(sitk.Resample(fov, cbct_clinical)).astype(bool)
        if fov is not None
        else np.ones_like(cbct_arr, dtype=bool)
    )

    # ── Intensity matching (linear, inside FOV only) ─────────────────────
    # ── Intensity matching & windowed difference ─────────────────────────
    ct_matched = _linear_rescale(ct_arr, cbct_arr, fov_arr)
    diff_arr = np.where(fov_arr, cbct_arr - ct_matched, 0)

    # ── Percentile-based window ──────────────────────────────────────────
    bg_cbct, hi_cbct = np.percentile(cbct_arr, 0.1), np.percentile(cbct_arr, 99.9)
    bg_ct, hi_ct = np.percentile(ct_arr, 0.1), np.percentile(ct_arr, 99.9)

    # ── Geometry ─────────────────────────────────────────────────────────
    z, y, x = cbct_arr.shape
    sx, sy, sz = cbct_clinical.GetSpacing()
    Lx, Ly, Lz = x * sx, y * sy, z * sz

    slice_sag = x // 2
    slice_cor = y // 2
    slice_ax = z // 2

    def _get_slice(vol, view_idx):
        if view_idx == 0:
            return vol[slice_ax, :, :]
        elif view_idx == 1:
            return vol[::-1, :, slice_sag]
        else:
            return vol[::-1, slice_cor, :]

    aspect_for_view = {0: sy / sx, 1: sz / sy, 2: sz / sx}
    
    # ── Build columns dynamically ────────────────────────────────────────
    columns = []  # list of (label, draw_func)

    if "raw" in vis_modes:
        columns.append(("CBCT clinical", lambda ax, v: ax.imshow(
            _get_slice(cbct_arr, v), cmap="gray", vmin=bg_cbct, vmax=hi_cbct, aspect=aspect_for_view[v])))
        columns.append((image2_label, lambda ax, v: ax.imshow(
            _get_slice(ct_arr, v), cmap="gray", vmin=bg_ct, vmax=hi_ct, aspect=aspect_for_view[v])))

    if "diff" in vis_modes:
        columns.append(("Difference (scaled)", lambda ax, v: ax.imshow(
            _get_slice(diff_arr, v), cmap="RdBu", vmin=-diff_range, vmax=diff_range, aspect=aspect_for_view[v])))

    if "checkerboard" in vis_modes:
        def _draw_checker(ax, v):
            ax.imshow(
                _make_checkerboard(_get_slice(cbct_arr, v),
                                   _get_slice(ct_matched, v),
                                   tile=checkerboard_tile),
                cmap="gray", vmin=bg_cbct, vmax=hi_cbct, aspect=aspect_for_view[v])
        columns.append(("Checkerboard", _draw_checker))

    if "fusion" in vis_modes:
        def _draw_fusion(ax, v):
            ax.imshow(_color_fusion(
                _get_slice(cbct_arr, v),
                _get_slice(ct_matched, v),
                bg_cbct, hi_cbct), aspect=aspect_for_view[v])
        columns.append(("Fusion (CBCT=M, CT=G)", _draw_fusion))

    if "edges" in vis_modes:
        def _draw_edges(ax, v):
            ax.imshow(_get_slice(cbct_arr, v), cmap="gray", vmin=bg_cbct, vmax=hi_cbct, aspect=aspect_for_view[v])
            ax.imshow(_sobel_edges(_get_slice(ct_arr, v)), cmap="Reds", alpha=0.4, aspect=aspect_for_view[v])
        columns.append(("CBCT + CT edges", _draw_edges))

    has_metadata = metadata not in (None, {})
    if has_metadata:
        columns.append(("__metadata__", None))

    n_cols = len(columns)
    n_rows = 3

    # ── Layout ───────────────────────────────────────────────────────────
    height_ratios = [
        Ly / Ly,
        (Lz / Ly) * (Lx / Ly),
        Lz / Ly,
    ]
    width_ratios = [1.0] * n_cols
    if has_metadata:
        width_ratios[-1] = 1.2

    x_len = n_cols * Lx / Ly
    y_len = np.sum(height_ratios) / x_len
    size = 20

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(size, y_len * size),
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # ── Draw ─────────────────────────────────────────────────────────────
    row_labels = ["Axial", "Sagittal", "Coronal"]

    def _add_label(ax, text):
        props = dict(facecolor="white", alpha=0.9, edgecolor="white", boxstyle="round,pad=0.5")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=props)

    def _add_patient(ax, text):
        props = dict(facecolor="white", alpha=0.9, edgecolor="white", boxstyle="round,pad=0.5")
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="right", bbox=props)

    for col_idx, (label, draw_fn) in enumerate(columns):
        for row_idx in range(n_rows):
            a = axes[row_idx, col_idx]
            a.set_xticks([])
            a.set_yticks([])

            if label == "__metadata__":
                a.axis("off")
                continue

            draw_fn(a, row_idx)
            _add_label(a, label)
            _add_patient(a, patient_ID)

            if col_idx == 0:
                a.set_ylabel(row_labels[row_idx], fontsize=12, fontweight="bold")

    if has_metadata:
        metadata_text = format_metadata(metadata, patient_ID)
        fig.text(
            0.73, 0.95, metadata_text, fontsize=13, va="top", ha="left",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.95, edgecolor="lightgray",
                      boxstyle="round,pad=0.6"),
        )

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_overview_projections(
    proj_real: np.ndarray,
    proj_sim: np.ndarray,
    gantry_angles: list,
    output_path: str,
    patient_ID: str,
    target_angles: list = [0, 45, 90],
) -> None:
    """
    Generate an overview PNG comparing real (I0-corrected) and simulated projections
    at three gantry angles.

    Parameters
    ----------
    proj_real : np.ndarray
        I0-corrected real projections, shape [N, H, W].
    proj_sim : np.ndarray
        Simulated projections, shape [N, H, W].
    gantry_angles : list
        Gantry angle in degrees for each projection, length N.
    output_path : str
        Path to save the overview PNG.
    patient_ID : str
        Patient identifier shown on each panel.
    target_angles : list
        Gantry angles (degrees) to display. Default: [0, 45, 90].
    """
    angles_arr = np.array(gantry_angles) % 360.0

    selected = []
    for target in target_angles:
        t = target % 360.0
        diff = np.abs(angles_arr - t)
        diff = np.minimum(diff, 360.0 - diff)
        idx = int(np.argmin(diff))
        selected.append((target, idx))

    n_rows = len(selected)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    props = dict(facecolor="white", alpha=0.9, edgecolor="white", boxstyle="round,pad=0.4")

    for row, (angle, idx) in enumerate(selected):
        real_sl = proj_real[idx]
        sim_sl  = proj_sim[idx]
        diff_sl = real_sl - sim_sl

        vmin_r, vmax_r = np.percentile(real_sl, 0.5), np.percentile(real_sl, 99.5)
        vmin_s, vmax_s = np.percentile(sim_sl,  0.5), np.percentile(sim_sl,  99.5)
        diff_lim = np.percentile(np.abs(diff_sl), 99)

        for col, (sl, label, cmap, vmin, vmax) in enumerate([
            (real_sl, "Real",       "gray",  vmin_r,    vmax_r),
            (sim_sl,  "Simulated",  "gray",  vmin_s,    vmax_s),
            (diff_sl, "Difference", "RdBu",  -diff_lim, diff_lim),
        ]):
            a = axes[row, col]
            a.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_xticks([])
            a.set_yticks([])
            a.text(0.05, 0.95, label, transform=a.transAxes, fontsize=9,
                   verticalalignment="top", bbox=props)
            a.text(0.95, 0.95, patient_ID, transform=a.transAxes, fontsize=9,
                   verticalalignment="top", horizontalalignment="right", bbox=props)
            if col == 0:
                a.set_ylabel(f"{angle}°", fontsize=12, fontweight="bold")

    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_final_overview(
    cbct_clinical: sitk.Image | None,
    cbct_rtk: sitk.Image,
    cbct_simulated: sitk.Image | None,
    ct_def_masked: sitk.Image | None,
    fov_cbct: sitk.Image | None,
    proj_real: np.ndarray,
    proj_sim: np.ndarray,
    gantry_angles: list,
    output_path: str,
    patient_ID: str,
    target_angles: list = [0, 45, 90],
    checkerboard_tile: int = 32,
) -> None:
    """
    Generate a single final-overview PNG combining the volumetric reconstructions
    and the projection comparison.

    The top block shows three orientations (axial / sagittal / coronal) of five
    panels, all resampled onto the CBCT RTK grid:
        1. CBCT clinical
        2. CBCT rtk, with the FOV mask drawn as a red outline
        3. CBCT simulated
        4. CT deformed (masked)
        5. Checkerboard of CBCT rtk vs. CT deformed (intensity matched)

    The bottom block shows the real and simulated projections at ``target_angles``.

    Parameters
    ----------
    cbct_clinical, cbct_rtk, cbct_simulated, ct_def_masked : sitk.Image
        Volumes to display. ``cbct_rtk`` is used as the reference grid; the others
        are resampled onto it. ``None`` panels are left blank. Pass copies if the
        caller needs to keep the originals unmodified (values < -1024 are clamped).
    fov_cbct : sitk.Image, optional
        CBCT field-of-view mask, drawn as a red contour over the RTK panel. If
        ``None``, the full volume is treated as in-FOV.
    proj_real, proj_sim : np.ndarray
        Real (I0-corrected) and simulated projections, shape [N, H, W].
    gantry_angles : list
        Gantry angle in degrees for each projection, length N.
    output_path : str
        Path to save the overview PNG.
    patient_ID : str
        Patient identifier shown on each panel.
    target_angles : list
        Gantry angles (degrees) to display for the projections. Default [0, 45, 90].
    checkerboard_tile : int
        Tile size in pixels for the checkerboard panel.
    """
    reference = cbct_rtk

    def _res(im, interp=sitk.sitkLinear, default=-1024.0):
        if im is None:
            return None
        return sitk.Resample(im, reference, sitk.Transform(), interp, default)

    # ── Resample everything onto the RTK grid ────────────────────────────
    rtk_im = cbct_rtk
    clin_im = _res(cbct_clinical)
    sim_im = _res(cbct_simulated)
    ctd_im = _res(ct_def_masked)
    fov_im = _res(fov_cbct, sitk.sitkNearestNeighbor, 0)

    for im in (rtk_im, clin_im, sim_im, ctd_im):
        if im is not None:
            im[im < -1024] = -1024

    def _arr(im):
        return sitk.GetArrayFromImage(im).astype(float) if im is not None else None

    rtk_a = _arr(rtk_im)
    clin_a = _arr(clin_im)
    sim_a = _arr(sim_im)
    ctd_a = _arr(ctd_im)
    fov_a = (
        sitk.GetArrayFromImage(fov_im).astype(bool)
        if fov_im is not None
        else np.ones_like(rtk_a, dtype=bool)
    )

    def _win(a):
        if a is None:
            return 0.0, 1.0
        return np.percentile(a, 0.1), np.percentile(a, 99.9)

    bg_rtk, hi_rtk = _win(rtk_a)
    bg_clin, hi_clin = _win(clin_a)
    bg_sim, hi_sim = _win(sim_a)
    bg_ctd, hi_ctd = _win(ctd_a)

    # Intensity-match CT deformed to RTK (inside FOV) for the checkerboard
    ct_matched = _linear_rescale(ctd_a, rtk_a, fov_a) if ctd_a is not None else None

    # ── Geometry / slicing ───────────────────────────────────────────────
    z, y, x = rtk_a.shape
    sx, sy, sz = reference.GetSpacing()
    Lx, Ly, Lz = x * sx, y * sy, z * sz

    slice_sag = x // 2
    slice_cor = y // 2
    slice_ax = z // 2

    def _gs(vol, v):
        if vol is None:
            return None
        if v == 0:
            return vol[slice_ax, :, :]
        elif v == 1:
            return vol[::-1, :, slice_sag]
        else:
            return vol[::-1, slice_cor, :]

    asp = {0: sy / sx, 1: sz / sy, 2: sz / sx}

    props = dict(facecolor="white", alpha=0.9, edgecolor="white", boxstyle="round,pad=0.5")

    def _blank(ax):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    # ── Panel draw functions (label, fn) ─────────────────────────────────
    def _gray(a, v):
        return _gs(a, v)

    def _draw_clin(ax, v):
        sl = _gray(clin_a, v)
        if sl is None:
            _blank(ax); return
        ax.imshow(sl, cmap="gray", vmin=bg_clin, vmax=hi_clin, aspect=asp[v])

    def _draw_rtk(ax, v):
        ax.imshow(_gray(rtk_a, v), cmap="gray", vmin=bg_rtk, vmax=hi_rtk, aspect=asp[v])
        fov_sl = _gs(fov_a.astype(float), v)
        if fov_sl is not None and fov_sl.any():
            ax.contour(fov_sl, levels=[0.5], colors="red", linewidths=1.0)

    def _draw_sim(ax, v):
        sl = _gray(sim_a, v)
        if sl is None:
            _blank(ax); return
        ax.imshow(sl, cmap="gray", vmin=bg_sim, vmax=hi_sim, aspect=asp[v])

    def _draw_ctd(ax, v):
        sl = _gray(ctd_a, v)
        if sl is None:
            _blank(ax); return
        ax.imshow(sl, cmap="gray", vmin=bg_ctd, vmax=hi_ctd, aspect=asp[v])

    def _draw_check(ax, v):
        if ct_matched is None:
            _blank(ax); return
        cb = _make_checkerboard(_gs(rtk_a, v), _gs(ct_matched, v), tile=checkerboard_tile)
        ax.imshow(cb, cmap="gray", vmin=bg_rtk, vmax=hi_rtk, aspect=asp[v])

    panels = [
        ("CBCT clinical", _draw_clin),
        ("CBCT rtk (FOV outline)", _draw_rtk),
        ("CBCT simulated", _draw_sim),
        ("CT def masked", _draw_ctd),
        ("Checkerboard rtk/CT", _draw_check),
    ]

    n_cols = len(panels)
    row_labels = ["Axial", "Sagittal", "Coronal"]
    # Physical height/width of each orientation panel (column width is uniform):
    #   axial    : Lx wide × Ly tall
    #   sagittal : Ly wide × Lz tall
    #   coronal  : Lx wide × Lz tall
    row_aspect = [Ly / Lx, Lz / Ly, Lz / Lx]
    height_ratios = row_aspect

    # ── Projection panel selection ───────────────────────────────────────
    angles_arr = np.array(gantry_angles) % 360.0
    selected = []
    for target in target_angles:
        t = target % 360.0
        diff = np.abs(angles_arr - t)
        diff = np.minimum(diff, 360.0 - diff)
        selected.append((target, int(np.argmin(diff))))

    # ── Auto-size the figure from the panel aspect ratios ────────────────
    col_w = 4.0  # inches per column
    top_h = col_w * float(np.sum(row_aspect))            # 3 stacked orientation rows
    proj_aspect = proj_real.shape[1] / proj_real.shape[2]  # H / W of a projection
    bot_h = 2 * col_w * proj_aspect                      # 2 rows: real / simulated
    fig_w = max(n_cols, len(selected)) * col_w
    fig_h = top_h + bot_h

    # ── Figure: top volume grid + bottom projection grid ─────────────────
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[top_h, bot_h])

    axes_top = sf_top.subplots(3, n_cols, gridspec_kw={"height_ratios": height_ratios})
    for col, (label, draw_fn) in enumerate(panels):
        for v in range(3):
            a = axes_top[v, col]
            a.set_xticks([]); a.set_yticks([])
            draw_fn(a, v)
            a.text(0.05, 0.95, label, transform=a.transAxes, fontsize=10,
                   verticalalignment="top", bbox=props)
            a.text(0.95, 0.95, patient_ID, transform=a.transAxes, fontsize=10,
                   verticalalignment="top", horizontalalignment="right", bbox=props)
            if col == 0:
                a.set_ylabel(row_labels[v], fontsize=12, fontweight="bold")

    # Bottom: rows = real / simulated, cols = angles
    axes_bot = sf_bot.subplots(2, len(selected), squeeze=False)
    for col, (angle, idx) in enumerate(selected):
        for row, (proj, label) in enumerate([(proj_real, "Real"), (proj_sim, "Simulated")]):
            a = axes_bot[row, col]
            sl = proj[idx]
            vmin, vmax = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
            a.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            a.text(0.05, 0.95, label, transform=a.transAxes, fontsize=9,
                   verticalalignment="top", bbox=props)
            a.text(0.95, 0.95, patient_ID, transform=a.transAxes, fontsize=9,
                   verticalalignment="top", horizontalalignment="right", bbox=props)
            if row == 0:
                a.set_title(f"{angle}°", fontsize=12, fontweight="bold")
            if col == 0:
                a.set_ylabel(label, fontsize=12, fontweight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
