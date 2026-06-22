import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

st.set_page_config(
    page_title="Super Resolution System (EDSR)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# MODEL
# =====================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):

    def __init__(
        self,
        num_blocks=32,
        channels=64,
        scale=4
    ):
        super().__init__()

        self.head = nn.Conv2d(
            3,
            channels,
            3,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * (scale ** 2),
                3,
                padding=1
            ),
            nn.PixelShuffle(scale),
            nn.Conv2d(
                channels,
                3,
                3,
                padding=1
            )
        )

    def forward(self, x):

        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)

        return x


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = EDSR(
        num_blocks=32,
        channels=64,
        scale=4
    )

    model.load_state_dict(
        torch.load(
            "edsr32_epoch100.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model32 = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Settings")

mode = st.sidebar.radio(
    "Input Mode",
    [
        "Upload LR Image",
        "Upload HR Image (Evaluate)"
    ]
)

# =====================================================
# TITLE
# =====================================================

st.title("Super-Resolution Image Enhancement System (EDSR)")

uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    img = Image.open(uploaded).convert("RGB")

    if mode == "Upload HR Image (Evaluate)":

        hr_img = img

        w, h = hr_img.size

        lr_img = hr_img.resize(
            (w // 4, h // 4),
            Image.BICUBIC
        )

        bicubic = lr_img.resize(
            (w, h),
            Image.BICUBIC
        )

        lr_tensor = transforms.ToTensor()(lr_img).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = model32(lr_tensor)

        edsr_img = to_pil_image(
            sr_tensor.squeeze(0).clamp(0, 1)
        )

        hr_np = np.array(
            hr_img.resize(
                edsr_img.size,
                Image.BICUBIC
            )
        )

        sr_np = np.array(edsr_img)
        
        psnr = peak_signal_noise_ratio(
            hr_np,
            sr_np,
            data_range=255
        )

        ssim = structural_similarity(
            hr_np,
            sr_np,
            channel_axis=2,
            data_range=255
        )
    else:

        lr_img = img

        w, h = lr_img.size

        bicubic = lr_img.resize(
            (w * 4, h * 4),
            Image.BICUBIC
        )

        lr_tensor = transforms.ToTensor()(lr_img).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = model32(lr_tensor)

        edsr_img = to_pil_image(
            sr_tensor.squeeze(0).clamp(0, 1)
        )

    st.markdown("---")

    if mode == "Upload HR Image (Evaluate)":

        m1, m2 = st.columns(2)

        m1.metric(
            "PSNR (dB)",
            f"{psnr:.2f}"
        )

        m2.metric(
            "SSIM",
            f"{ssim:.4f}"
        )

        st.markdown("---")

    lr_display = lr_img.resize(
        edsr_img.size,
        Image.NEAREST
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            "<h3 style='text-align:center;'>Input Image</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center;'>Size: {lr_img.size[0]} × {lr_img.size[1]}</p>",
            unsafe_allow_html=True
        )

        st.image(
            lr_display,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "<h3 style='text-align:center;'>EDSR Enhanced Image</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center;'>Size: {edsr_img.size[0]} × {edsr_img.size[1]}</p>",
            unsafe_allow_html=True
        )

        st.image(
            edsr_img,
            use_container_width=True
        )
    with col3:

        st.markdown(
            "<h3 style='text-align:center;'>Input Zoom</h3>",
            unsafe_allow_html=True
        )

        st.image(
            crop_lr,
            use_container_width=True
        )

    with col4:

        st.markdown(
            "<h3 style='text-align:center;'>EDSR Zoom</h3>",
            unsafe_allow_html=True
        )

        st.image(
            crop_edsr,
            use_container_width=True
        )

    img_np = np.array(edsr_img)

    H, W = img_np.shape[:2]

    crop_size = min(H, W) // 4

    x = st.sidebar.slider(
        "Zoom X",
        0,
        max(0, W - crop_size),
        W // 3
    )

    y = st.sidebar.slider(
        "Zoom Y",
        0,
        max(0, H - crop_size),
        H // 3
    )

    lr_np = np.array(lr_display)
    edsr_np = np.array(edsr_img)

    crop_lr = lr_np[
        y:y + crop_size,
        x:x + crop_size
    ]

    crop_edsr = edsr_np[
        y:y + crop_size,
        x:x + crop_size
    ]

    zoom_size = 250

    crop_lr = Image.fromarray(crop_lr).resize(
        (zoom_size, zoom_size)
    )

    crop_edsr = Image.fromarray(crop_edsr).resize(
        (zoom_size, zoom_size)
    )


