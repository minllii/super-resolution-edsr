import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

st.set_page_config(
    page_title="Super Resolution System",
    layout="wide"
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
        num_blocks=16,
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
        num_blocks=16,
        channels=64,
        scale=4
    )

    model.load_state_dict(
        torch.load(
            "edsr16_epoch100.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model16 = load_model()

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

st.title("Super-Resolution Image Enhancement System")

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
            sr_tensor = model16(lr_tensor)

        edsr_img = to_pil_image(
            sr_tensor.squeeze(0).clamp(0, 1)
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
            sr_tensor = model16(lr_tensor)

        edsr_img = to_pil_image(
            sr_tensor.squeeze(0).clamp(0, 1)
        )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Input Size",
        f"{lr_img.size[0]} × {lr_img.size[1]}"
    )

    c2.metric(
        "Output Size",
        f"{edsr_img.size[0]} × {edsr_img.size[1]}"
    )

    c3.metric(
        "Scale Factor",
        "4×"
    )

    st.markdown("---")

    lr_display = lr_img.resize(
        edsr_img.size,
        Image.NEAREST
    )

    st.subheader("Comparison")

    col1, col2 = st.columns(2)

    col1.image(
        lr_display,
        caption="Input Low Resolution Image",
        use_container_width=True
    )

    col2.image(
        edsr_img,
        caption="EDSR16 Enhanced Image",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Zoom Comparison")

    img_np = np.array(edsr_img)

    H, W = img_np.shape[:2]

    crop_size = min(H, W) // 3

    x = st.slider(
        "Zoom X",
        0,
        max(0, W - crop_size),
        W // 3
    )

    y = st.slider(
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

    zoom_size = 600

    crop_lr = Image.fromarray(crop_lr).resize(
        (zoom_size, zoom_size)
    )

    crop_edsr = Image.fromarray(crop_edsr).resize(
        (zoom_size, zoom_size)
    )

    col1, col2, col3 = st.columns(3)

    col1.image(
        crop_lr,
        caption="Input LR Zoom",
        use_container_width=True
    )

    col2.image(
        crop_edsr,
        caption="EDSR16 Zoom",
        use_container_width=True
    )