import os
import logging
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def generate_overview_dir(
    ct: sitk.Image | None,
    input: sitk.Image | None,
    ct_deformed: sitk.Image | None,
    output_path: str,
    patient_ID: str,
) -> None:
    """
    Generate an overview PNG image showing slices from different orientations of the input CT, input image, and mask.

    Parameters:
    ct (sitk.Image): The CT image.
    input (sitk.Image): The input image.
    mask (sitk.Image): The mask image.
    output_dir (str): The directory to save the overview PNG image.

    Returns:
    None
    """
    if ct is None or input is None or ct_deformed is None:
        logger.warning("One or more input images are None. Overview image will not be generated.")
        return

    shape = np.shape(sitk.GetArrayFromImage(ct))
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)
    background_input = np.percentile(sitk.GetArrayFromImage(input), 0.1)
    high_input = np.percentile(sitk.GetArrayFromImage(input), 99.9)

    slice_sag = shape[2]//2
    slice_cor = shape[1]//2
    slice_ax = shape[0]//2

    # calculate final size of figure so minimal white space in figure
    sag_len = slice_sag*2
    cor_len = slice_cor*2
    ax_len = slice_ax*2
    height_ratios = [cor_len/cor_len,((ax_len*3)/cor_len)*sag_len/cor_len,ax_len*3/cor_len]
    x_len = (sag_len/cor_len)*3
    y_len = np.array(height_ratios).sum()/x_len
    gridspec_kw={'width_ratios':[1,1,1],'height_ratios':height_ratios}

    size=20
    fig,ax = plt.subplots(3,3,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)

    # ax[0,1].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,0].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='gray',vmin=background_input,vmax=high_input)
    # ax[0,3].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[0,1].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,2].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,2].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)

    # ax[1,1].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,0].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    # ax[1,3].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    # ax[1,3].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[1,1].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,2].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)

    # ax[2,1].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,0].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    # ax[2,3].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    # ax[2,3].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[2,1].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,2].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)

    def add_text(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

    def add_patient(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top',horizontalalignment='right',bbox=props)

    for r,ax_row in enumerate(ax):
        for c,a in enumerate(ax_row):
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                add_text(a,'Input')
                add_patient(a,patient_ID)
                a.set_ylabel('Axial' if r == 0 else 'Sagittal' if r == 1 else 'Coronal',fontsize=12,fontweight='bold')
            if c == 1:
                add_text(a,'CT deformed')
                add_patient(a,patient_ID)
            if c == 2:
                add_text(a,'Overlay Input/CT def')
                add_patient(a,patient_ID)
            if c == 3:
                add_text(a,'Overlay')
                add_patient(a,patient_ID)
            if c == 4:
                add_text(a,'Overlay def')
                add_patient(a,patient_ID)

    fig.subplots_adjust(wspace=0.02,hspace=0.02)
    plt.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path,f'{patient_ID}_DIR.png'),dpi=300,bbox_inches='tight')
