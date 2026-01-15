import os
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend
from matplotlib import pyplot as plt
import utils.img as img

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
    ax[0,0].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:], cmap='gray', vmin=background_ct, vmax=high_ct)
    ax[0,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[slice_ax,:,:], cmap='gray', vmin=background_clinical, vmax=high_clinical)
    ax[0,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[slice_ax,:,:], cmap='gray', vmin=background_rtk, vmax=high_rtk)

    # Row 1 (Sagittal)
    ax[1,0].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag], cmap='gray', vmin=background_ct, vmax=high_ct)
    ax[1,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[::-1,:,slice_sag], cmap='gray', vmin=background_clinical, vmax=high_clinical)
    ax[1,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[::-1,:,slice_sag], cmap='gray', vmin=background_rtk, vmax=high_rtk)

    # Row 2 (Coronal)
    ax[2,0].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:], cmap='gray', vmin=background_ct, vmax=high_ct)
    ax[2,1].imshow(sitk.GetArrayFromImage(cbct_clinical)[::-1,slice_cor,:], cmap='gray', vmin=background_clinical, vmax=high_clinical)
    ax[2,2].imshow(sitk.GetArrayFromImage(cbct_rtk)[::-1,slice_cor,:], cmap='gray', vmin=background_rtk, vmax=high_rtk)

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

