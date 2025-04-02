
import torch
from PIL import Image
import requests
from io import BytesIO
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
leres = LeresDetector.from_pretrained("lllyasviel/Annotators")


# specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
# det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
# det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
# pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
# pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()

from torchvision import transforms

to_tensor = transforms.ToTensor()

def get_input_tensor(img):
    with torch.no_grad():
        processed_image_midas = midas(img)
        processed_image_open_pose = open_pose(img, hand_and_face=True)
        processed_image_normal_bae = normal_bae(img)
        processed_image_sam = sam(img)
        processed_image_leres = leres(img)
        processed_image_canny = canny(img)
        processed_image_content = content(img)

    images = [
        img,                         # PIL Image
        processed_image_open_pose,   # Might be np.ndarray or PIL Image
        processed_image_normal_bae,
        processed_image_sam,
        processed_image_leres,
        processed_image_canny
    ]

    tensor_images = []
    for im in images:
        # If the annotator returns a PIL image, just call `to_tensor(...)`.
        # If it returns a NumPy array, we can do `torch.from_numpy(...).permute(...)`.
        if isinstance(im, Image.Image):
            t_im = to_tensor(im).to(device)
        else:
            # If `im` is already a NumPy array: shape could be (H, W) or (H, W, C)
            # Convert it to a PyTorch tensor and permute as needed.
            # Example if shape is (H, W, C):
            import numpy as np
            if isinstance(im, np.ndarray):
                if im.ndim == 2:
                    # e.g. grayscale -> expand dims to (H, W, 1) before permute
                    im = np.expand_dims(im, axis=2)
                t_im = torch.from_numpy(im).float()  # put on float
                t_im = t_im.permute(2, 0, 1)         # channel-first
                t_im = t_im / 255.0                 # if in 0..255 range
                t_im = t_im.to(device)
            else:
                # If some other format, handle accordingly
                raise ValueError("Unknown image type")
        tensor_images.append(t_im)
        signal_input = torch.cat(tensor_images, dim=0)

    return signal_input


