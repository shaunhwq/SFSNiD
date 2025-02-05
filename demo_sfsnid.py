import os
from typing import Tuple

import cv2
import torch
import numpy as np

from methods.MyNightDehazing.SFSNiD import build_net


def pre_process(image: np.array, device: str, image_size: Tuple[int, int]) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image


def post_process(model_output: torch.Tensor, original_shape_hw: Tuple[int, int]):
    image_rgb = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.resize(image_bgr, original_shape_hw[::-1])
    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/31_hazy_video.mp4"
    device = "cpu"
    kappa_value = 100
    assert kappa_value in [100, 130, 150, 180, 200, 230, 250, 280, 300], "Invalid Kappa Value"
    weights_path = f"weights/different_kappa/RWNHC_MM23_PseudoLabel_BriRatio100_Kappa{kappa_value}/models/last_SFSNiD_RWNHC_MM23_PseudoLabel.pth"
    input_hw = (256, 256)
    input_hw = list(map(int, input_hw))

    model = build_net(num_res=3)
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        frame_shape_hw = frame.shape[:2]

        in_tensor = pre_process(frame, device, input_hw)
        with torch.no_grad():
            model_outputs = model(in_tensor)
        out_image = post_process(model_outputs, frame_shape_hw)

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
