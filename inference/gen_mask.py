import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

import glob
import argparse
import numpy as np
import torch
import cv2
import json
from core.flowseek import FlowSeek
from core.utils import flow_viz
from core.utils.utils import InputPadder

# device setting
DEVICE = 'cuda'
torch.backends.cudnn.benchmark = True

def load_image(imfile):
    if not os.path.exists(imfile):
        raise ValueError(f"figure does not exist: {imfile}")
    img = cv2.imread(imfile)
    if img is None:
        raise ValueError(f"can't read image: {imfile}")
    img = img[:, :, ::-1].copy()  # BGR to RGB 
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    # initialization
    print(f"Loading model: {args.model}")
    model = FlowSeek(args)
    checkpoint = torch.load(args.model, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # read sequence
    images = sorted(glob.glob(os.path.join(args.image_folder, "*.png")) + 
                    glob.glob(os.path.join(args.image_folder, "*.jpg")))
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images in {args.image_folder}, found {len(images)}")
    
    num_pairs = min(len(images)-1, args.num_pairs) if args.num_pairs > 0 else len(images)-1
    print(f"Found {len(images)} images, processing {num_pairs} pairs...")

    # output folder
    os.makedirs(args.output, exist_ok=True)

    times_ms = []
    with torch.no_grad():
        # warm-up 
        print("Warm up...")
        im1, im2 = load_image(images[0]), load_image(images[1])
        padder = InputPadder(im1.shape)
        im1, im2 = padder.pad(im1, im2)
        _ = model(im1, im2, iters=args.iters, test_mode=True)
        torch.cuda.synchronize()

        print(f"{'Pair':<6} | {'Time(ms)':<9} | {'FPS':<8} | {'Dynamic%':<8}")
        print("-"*45)

        for i in range(num_pairs):
            image1_path, image2_path = images[i], images[i+1]
            print(f"Processing: {os.path.basename(image1_path)} → {os.path.basename(image2_path)}")
            
            image1 = load_image(image1_path)
            image2 = load_image(image2_path)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # count inference time
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            results_dict = model(image1, image2, iters=args.iters, test_mode=True)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times_ms.append(curr_time)

            # Flow processing
            flow_pr = results_dict['flow'][-1]
            flow = padder.unpad(flow_pr[0]).cpu()
            flow_np = flow.permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(flow_np)
            flow_bgr = flow_img[:, :, ::-1]
            
            # save flow
            flow_name = f"flow_{i+1:05d}.png"
            cv2.imwrite(os.path.join(args.output, flow_name), flow_bgr)

            magnitude = np.linalg.norm(flow_np, axis=2)  # float, real px displacement

            # adaptive + multi-scale
            thresh1 = np.mean(magnitude) + 2 * np.std(magnitude)  # obvious motion
            thresh2 = np.percentile(magnitude, 85)  # top 15%
            mask = ((magnitude > thresh1) | (magnitude > thresh2)).astype(np.uint8) * 255

            # HSV
            #hsv = cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2HSV)
            #sat = hsv[:,:,1]
            #mask |= (sat > 120).astype(np.uint8) * 255  

            # morphology
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

            
            # save mask
            mask_name = f"mask_{i+1:05d}.png"
            cv2.imwrite(os.path.join(args.output, mask_name), mask)
            
            # visualization output
            dynamic_pct = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100
            print(f"{i+1:03d}    | {curr_time:7.1f}  | {1000/curr_time:6.1f} | {dynamic_pct:6.1f}%")

    # summary
    times_ms = np.array(times_ms)
    print("\n" + "="*45)
    print("SUMMARY")
    print("="*45)
    print(f"Total pairs: {num_pairs}")
    print(f"Avg time: {np.mean(times_ms):.1f}ms | Avg FPS: {1000/np.mean(times_ms):.1f}")
    print(f"Output: {args.output}/ (flow_*.png, mask_*.png)")
    print("="*45)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help="path (e.g. TUM rgb/)")
    parser.add_argument('--output', default="sequence_results", help="output dir")
    parser.add_argument('--num_pairs', type=int, default=0, help="Top N (0=all)")
    parser.add_argument('--model', default="weights/flowseek_T_CT.pth")
    parser.add_argument('--cfg', default="config/eval/flowseek-T.json")
    parser.add_argument('--iters', type=int, default=4)
    args = parser.parse_args()
    
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    
    demo(args)

