import sys
import os
import cv2
import torch
import json
import numpy as np
import pyrealsense2 as rs

# --------------------------------------------------
# Path setting
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

from core.flowseek import FlowSeek
from core.utils.utils import InputPadder
from core.utils import flow_viz

DEVICE = 'cuda'


# --------------------------------------------------
# Utils
# --------------------------------------------------
def frame_to_tensor(frame):
    """
    OpenCV BGR frame -> torch tensor (1,3,H,W) RGB
    """
    frame = frame[:, :, ::-1].copy()  # BGR -> RGB
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    return tensor[None].to(DEVICE)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):

    # ------------------------
    # Load FlowSeek model
    # ------------------------
    print("Loading FlowSeek model...")
    model = FlowSeek(args)

    checkpoint = torch.load(args.model, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    # ------------------------
    # Initialize RealSense
    # ------------------------
    print("Initializing RealSense camera...")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected")

    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"Using RealSense device: {serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    prev_tensor = None
    padder = None

    print("Press 'q' to quit.")

    # ------------------------
    # Main loop
    # ------------------------
    with torch.no_grad():
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Show raw camera
            cv2.imshow("Camera View", frame)

            curr_tensor = frame_to_tensor(frame)

            if prev_tensor is not None:
                if padder is None:
                    padder = InputPadder(curr_tensor.shape)

                img1, img2 = padder.pad(prev_tensor, curr_tensor)

                results = model(img1, img2, iters=args.iters, test_mode=True)

                flow = results['flow'][-1][0]     # (2,H,W)
                flow = padder.unpad(flow).cpu()

                flow_np = flow.permute(1, 2, 0).numpy()
                flow_img = flow_viz.flow_to_image(flow_np)

                cv2.imshow("Result Flow", flow_img[:, :, ::-1])  # RGB -> BGR

            prev_tensor = curr_tensor

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    # ------------------------
    # Cleanup
    # ------------------------
    pipeline.stop()
    cv2.destroyAllWindows()


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="weights/flowseek_T_CT.pth")
    parser.add_argument('--cfg', default="config/eval/flowseek-T.json")
    parser.add_argument('--iters', type=int, default=4)

    args = parser.parse_args()

    # Load config JSON
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    main(args)

