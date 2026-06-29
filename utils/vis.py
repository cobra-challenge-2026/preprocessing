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
    lines.append(f"Center: {patient_ID[0]}")
    lines.append("CT:")
    lines.append(f"  Manufacturer: {md.get('ct', {}).get('Manufacturer', 'N/A')}")
    lines.append(f"  Model: {md.get('ct', {}).get('ManufacturerModelName', 'N/A')}")
    lines.append(f"  kVp: {md.get("ct", {}).get("KVP", None)} kV")
    lines.append(f"  mA: {md.get("ct", {}).get("XRayTubeCurrent", None)} mA")
    lines.append(f"  Exposure Time: {md.get("ct", {}).get("ExposureTime", None)} ms")
    lines.append(f"  Slice Thickness: {md.get("ct", {}).get("SliceThickness", None)} mm")
    lines.append(f"  Pixel Spacing: {md.get("ct", {}).get("PixelSpacing", None)} mm")
    lines.append("")
    lines.append("CBCT:")
    lines.append(f"  Manufacturer: {md.get('cbct', {}).get('Manufacturer', 'N/A')}")
    lines.append(f"  kVp: {md.get("cbct", {}).get("TubeVoltage", None)} kV")
    lines.append(f"  mA: {md.get("cbct", {}).get("TubeCurrent", None)} mA")
    lines.append(f"  Exposure Time: {md.get("cbct", {}).get("PulseLength", None)} ms")
    lines.append(f"  Frames: {md.get("cbct", {}).get("Frames", None)}")
    lines.append(f"  Projection Spacing: {md.get("cbct", {}).get("ImagerResX", None)} mm x {md.get("cbct", {}).get("ImagerResY", None)} mm")
    lines.append(f"  Projection Size: {md.get("cbct", {}).get("ImagerSizeX", None)} x {md.get("cbct", {}).get("ImagerSizeY", None)}")
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


def _spread_targets_over_arc(angles_arr: np.ndarray, n: int) -> list:
    """Return ``n`` angles (deg, in [0, 360)) spread evenly across the acquired arc.

    The acquired arc is located by finding the largest angular gap between
    consecutive sorted angles; the rest of the circle is the covered span. A
    partial arc (e.g. a Varian ~210° scan) is sampled inclusive of both
    endpoints, while a near-full 360° scan is spaced around the whole circle.
    This avoids collapsing several targets onto the same projection when the
    scan does not cover hard-coded targets like 0/45/90.
    """
    s = np.unique(np.sort(angles_arr))
    if len(s) <= n:
        return list(s)
    gaps = np.append(np.diff(s), (s[0] + 360.0) - s[-1])
    k = int(np.argmax(gaps))
    max_gap = float(gaps[k])
    start = float(s[(k + 1) % len(s)])     # first angle after the largest gap
    span = 360.0 - max_gap
    if max_gap < 10.0:                      # effectively a full circle
        return [(start + 360.0 * i / n) % 360.0 for i in range(n)]
    if n == 1:
        return [(start + span / 2.0) % 360.0]
    return [(start + span * i / (n - 1)) % 360.0 for i in range(n)]


def _select_projection_indices(
    gantry_angles: list, n: int = 3, target_angles: list | None = None
) -> list:
    """Pick projection indices to display, labelled by their true gantry angle.

    If ``target_angles`` is given, each target is matched to its nearest acquired
    angle (legacy behaviour). Otherwise ``n`` angles are spread evenly across the
    actually-acquired arc (see :func:`_spread_targets_over_arc`).

    Returns a list of ``(true_angle_deg, index)`` where ``true_angle_deg`` is the
    rounded gantry angle of the chosen projection, so each panel is labelled with
    what it actually shows rather than the requested target.
    """
    angles_arr = np.array(gantry_angles, dtype=float) % 360.0
    if target_angles is not None:
        targets = [t % 360.0 for t in target_angles]
    else:
        targets = _spread_targets_over_arc(angles_arr, n)

    selected = []
    for t in targets:
        diff = np.abs(angles_arr - t)
        diff = np.minimum(diff, 360.0 - diff)
        idx = int(np.argmin(diff))
        selected.append((int(round(angles_arr[idx])), idx))
    return selected


def generate_overview_projections(
    proj_real: np.ndarray,
    proj_sim: np.ndarray,
    gantry_angles: list,
    output_path: str,
    patient_ID: str,
    target_angles: list | None = None,
    n_proj: int = 3,
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
    target_angles : list, optional
        Gantry angles (degrees) to display. If ``None`` (default), ``n_proj``
        angles are spread evenly across the actually-acquired arc instead, which
        avoids picking duplicate projections on partial-arc (e.g. Varian) scans.
    n_proj : int
        Number of projection rows to show when ``target_angles`` is ``None``.
    """
    selected = _select_projection_indices(gantry_angles, n=n_proj, target_angles=target_angles)

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
    cbct_rtk: sitk.Image,
    cbct_simulated: sitk.Image | None,
    ct_def_masked: sitk.Image | None,
    fov_cbct: sitk.Image | None,
    proj_real: np.ndarray,
    proj_sim: np.ndarray,
    gantry_angles: list,
    output_path: str,
    patient_ID: str,
    metadata: dict | None = None,
    target_angles: list | None = None,
    n_proj: int = 3,
    checkerboard_tile: int = 32,
    proj_spacing: tuple[float, float] | None = None,
    proj_window_percentiles: tuple[float, float] = (5, 95.0),
    proj_roi_frac: float = 0.4,
) -> None:
    """
    Generate a single final-overview PNG combining the volumetric reconstructions
    and the projection comparison.

    The top block shows three orientations (axial / sagittal / coronal) of four
    panels, all resampled onto the CBCT RTK grid:
        1. CBCT rtk, with the FOV mask drawn as a red outline
        2. CBCT simulated
        3. CT deformed (masked)
        4. Checkerboard of CBCT rtk vs. CT deformed (intensity matched)

    The bottom block shows the real and simulated projections at ``target_angles``.

    Parameters
    ----------
    cbct_rtk, cbct_simulated, ct_def_masked : sitk.Image
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
    metadata : dict, optional
        Acquisition metadata to display as a text box. If ``None`` or empty, no
        metadata box is drawn.
    target_angles : list, optional
        Gantry angles (degrees) to display for the projections. If ``None``
        (default), ``n_proj`` angles are spread evenly across the actually-
        acquired arc instead, which avoids picking duplicate projections on
        partial-arc (e.g. Varian) scans that don't cover hard-coded targets.
    n_proj : int
        Number of projection columns to show when ``target_angles`` is ``None``.
    checkerboard_tile : int
        Tile size in pixels for the checkerboard panel.
    proj_spacing : tuple of float, optional
        Detector pixel spacing ``(spacing_x, spacing_y)`` in mm for the projections,
        used to display them with the correct physical aspect ratio. If ``None``,
        square pixels are assumed.
    proj_window_percentiles : tuple of float, optional
        ``(low, high)`` percentiles used to set the grayscale window for the
        projection panels. The projections are raw 16-bit detector signal (not
        line integrals), where the bright open-beam region sits near the top of
        the range and the patient is the darker, attenuated part. The window is
        computed per panel (real and simulated are windowed independently, since
        their detector-signal scales differ) over the central detector ROI (see
        ``proj_roi_frac``). Lower ``high`` to saturate the bright open beam and
        expand contrast in the patient. Default ``(1.0, 75.0)``.
    proj_roi_frac : float, optional
        Fraction (0–1) of the detector width/height, centred on the detector, used
        to compute the projection window. The patient projects near the detector
        centre at every gantry angle while the bright open beam is peripheral, so
        restricting the percentiles to this central ROI keeps the window locked on
        the patient regardless of how much open beam is in frame (which otherwise
        crushes thin AP-style views to black). Set to ``1.0`` to use the full
        frame. Default ``0.6``.
    """
    reference = cbct_rtk

    def _res(im, interp=sitk.sitkLinear, default=-1024.0):
        if im is None:
            return None
        return sitk.Resample(im, reference, sitk.Transform(), interp, default)

    # ── Resample everything onto the RTK grid ────────────────────────────
    rtk_im = cbct_rtk
    sim_im = _res(cbct_simulated)
    ctd_im = _res(ct_def_masked)
    fov_im = _res(fov_cbct, sitk.sitkNearestNeighbor, 0)

    for im in (rtk_im, sim_im, ctd_im):
        if im is not None:
            im[im < -1024] = -1024

    def _arr(im):
        return sitk.GetArrayFromImage(im).astype(float) if im is not None else None

    rtk_a = _arr(rtk_im)
    sim_a = _arr(sim_im)
    ctd_a = _arr(ctd_im)
    fov_a = (
        sitk.GetArrayFromImage(fov_im).astype(bool)
        if fov_im is not None
        else np.ones_like(rtk_a, dtype=bool)
    )

    def _win(a, mask=None):
        if a is None:
            return 0.0, 1.0
        vals = a[mask] if mask is not None and mask.any() else a
        return float(np.percentile(vals, 0.1)), float(np.percentile(vals, 99.9))

    _win_basis = rtk_a if rtk_a is not None else (sim_a if sim_a is not None else ctd_a)
    bg, hi = _win(_win_basis, fov_a if rtk_a is not None else None)

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

    def _blank(ax):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=_fs(12), color="gray")

    # ── Panel draw functions (label, fn) ─────────────────────────────────
    def _gray(a, v):
        return _gs(a, v)

    def _draw_rtk(ax, v):
        ax.imshow(_gray(rtk_a, v), cmap="gray", vmin=bg, vmax=hi, aspect="auto")
        fov_sl = _gs(fov_a.astype(float), v)
        if fov_sl is not None and fov_sl.any():
            ax.contour(fov_sl, levels=[0.5], colors="red", linewidths=1.0)

    def _draw_sim(ax, v):
        sl = _gray(sim_a, v)
        if sl is None:
            _blank(ax); return
        ax.imshow(sl, cmap="gray", vmin=bg, vmax=hi, aspect="auto")

    def _draw_ctd(ax, v):
        sl = _gray(ctd_a, v)
        if sl is None:
            _blank(ax); return
        ax.imshow(sl, cmap="gray", vmin=bg, vmax=hi, aspect="auto")

    def _draw_check(ax, v):
        if ct_matched is None:
            _blank(ax); return
        cb = _make_checkerboard(_gs(rtk_a, v), _gs(ct_matched, v), tile=checkerboard_tile)
        ax.imshow(cb, cmap="gray", vmin=bg, vmax=hi, aspect="auto")

    panels = [
        ("CBCT RTK + FOV outline", _draw_rtk),
        ("CBCT simulated", _draw_sim),
        ("CT deformed", _draw_ctd),
        ("CBCT RTK vs. CT deformed", _draw_check),
    ]

    # ── Projection panel selection ───────────────────────────────────────
    # Indices are chosen (and labelled) from the actually-acquired arc unless
    # explicit target_angles are passed, so partial-arc scans don't collapse
    # multiple panels onto the same projection.
    selected = _select_projection_indices(gantry_angles, n=n_proj, target_angles=target_angles)

    n_cols = len(panels)
    has_metadata = metadata not in (None, {})
    n_top_cols = n_cols
    n_bot_cols = len(selected) + (1 if has_metadata else 0)
    row_labels = ["Axial", "Sagittal", "Coronal"]
    # Physical height/width of each orientation panel (column width is uniform):
    #   axial    : Lx wide × Ly tall
    #   sagittal : Ly wide × Lz tall
    #   coronal  : Lx wide × Lz tall
    row_aspect = [Ly / Lx, Lz / Ly, Lz / Lx]
    height_ratios = row_aspect

    col_w = 4.0  # inches per image column
    meta_w = 1.2  # metadata column width relative to an image column
    top_width_ratios = [1.0] * n_cols
    bot_width_ratios = [1.0] * len(selected) + ([meta_w] if has_metadata else [])
    top_width_units = float(np.sum(top_width_ratios))
    bot_width_units = float(np.sum(bot_width_ratios))

    fig_w = max(top_width_units, bot_width_units) * col_w
    img_col_w = fig_w / top_width_units                  # actual reconstruction-column width
    top_h = img_col_w * float(np.sum(row_aspect))        # 3 stacked orientation rows

    # Pixel aspect (height of one pixel / width of one pixel) from detector spacing
    proj_pix_asp = (proj_spacing[1] / proj_spacing[0]) if proj_spacing else 1.0
    # Physical H / W of a projection = (rows/cols) scaled by the pixel aspect
    proj_aspect = proj_real.shape[1] / proj_real.shape[2] * proj_pix_asp
    proj_col_w = fig_w / bot_width_units                 # actual projection-column width
    bot_h = 2 * proj_col_w * proj_aspect                 # 2 rows: real / simulated
    fig_h = top_h + bot_h

    # Font sizes are in absolute points, but the canvas size varies with the
    # image/projection aspect ratios, so a fixed point size occupies a different
    # fraction of the figure each time. Scale all fonts by the canvas's geometric
    # mean dimension (relative to a reference) to keep text a constant relative
    # size. Reference 18.0 ≈ a typical near-cubic-volume / square-projection run.
    fig_scale = float(np.sqrt(fig_w * fig_h) / 18.0)

    def _fs(base):
        return base * fig_scale

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False, facecolor="black")
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[top_h, bot_h],
        hspace=0.12,
    )
    top_grid = outer[0].subgridspec(
        3, n_top_cols,
        height_ratios=height_ratios,
        width_ratios=top_width_ratios,
        wspace=0.02,
        hspace=0.04,
    )
    bot_grid = outer[1].subgridspec(
        2, n_bot_cols,
        width_ratios=bot_width_ratios,
        wspace=0.02,
        hspace=0.04,
    )

    axes_top = np.empty((3, n_top_cols), dtype=object)
    for row in range(3):
        for col in range(n_top_cols):
            axes_top[row, col] = fig.add_subplot(top_grid[row, col])

    for col, (label, draw_fn) in enumerate(panels):
        for v in range(3):
            a = axes_top[v, col]
            a.set_xticks([]); a.set_yticks([])
            draw_fn(a, v)
            if v == 0:
                a.set_title(label, fontsize=_fs(14), fontweight="bold", color="white")
            if col == 0:
                a.set_ylabel(row_labels[v], fontsize=_fs(14), fontweight="bold", color="white")

    # Bottom: rows = real / simulated, cols = angles
    axes_bot = np.empty((2, len(selected)), dtype=object)
    for row in range(2):
        for col in range(len(selected)):
            axes_bot[row, col] = fig.add_subplot(bot_grid[row, col])

    lo_pct, hi_pct = proj_window_percentiles

    def _proj_window(sl):
        if 0 < proj_roi_frac < 1:
            h, w = sl.shape
            rh, rw = int(round(h * proj_roi_frac)), int(round(w * proj_roi_frac))
            r0, c0 = (h - rh) // 2, (w - rw) // 2
            region = sl[r0:r0 + rh, c0:c0 + rw]
        else:
            region = sl
        return np.percentile(region, lo_pct), np.percentile(region, hi_pct)

    for col, (angle, idx) in enumerate(selected):
        for row, (proj, label) in enumerate([(proj_real, "Real"), (proj_sim, "Simulated")]):
            a = axes_bot[row, col]
            sl = proj[idx]
            vmin, vmax = _proj_window(sl)
            a.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(f"{angle}°", fontsize=_fs(14), fontweight="bold", color="white")
            if col == 0:
                a.set_ylabel(label, fontsize=_fs(14), fontweight="bold", color="white")

    # ── Metadata text box in the projection block's right-hand column ─────
    if has_metadata:
        meta_ax = fig.add_subplot(bot_grid[:, len(selected)])
        meta_ax.axis("off")
        meta_ax.text(
            0.06, 0.5, format_metadata(metadata, patient_ID),
            transform=meta_ax.transAxes, fontsize=_fs(11), color="white",
            va="center", ha="left", family="monospace",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="white",
                      boxstyle="round,pad=0.6"),
        )

    fig.suptitle(f'Patient ID: {patient_ID}', color="white", fontsize=_fs(21), fontweight="bold", y=0.998)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.015)

    # ── Vertical section headings down the left margin ────────────────────
    # Rotated and placed left of the per-orientation / per-row labels, vertically
    # centred on each block's bbox so they span its rows (3 for the
    # reconstructions, 2 for the projections) and track the layout at any aspect
    # ratio. The inter-block gap (outer hspace) keeps the projections visually
    # separated from the reconstructions.
    for cell, text in ((outer[0], "Reconstructed Images"), (outer[1], "Projections")):
        pos = cell.get_position(fig)
        fig.text(0.012, 0.5 * (pos.y0 + pos.y1), text, color="white",
                 fontsize=_fs(16), fontweight="bold", ha="center", va="center",
                 rotation=90)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="black")
    plt.close(fig)
