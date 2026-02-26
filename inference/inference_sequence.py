import sys
import os
import time
import glob
import argparse
import numpy as np
import torch
import cv2
import json

# path setting
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

from core.flowseek import FlowSeek
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    if not os.path.exists(imfile):
        raise ValueError(f"Image not found: {imfile}")
    img = cv2.imread(imfile)
    if img is None:
        raise ValueError(f"Failed to load image: {imfile}")
        
    img = img[:, :, ::-1].copy()  # BGR -> RGB
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    # 1. initialize model
    print(f"Loading model: {args.model}")
    model = FlowSeek(args)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(DEVICE)
    model.eval()

    # 2. acquire image sequence
    images = sorted(glob.glob(os.path.join(args.path, "*.png")) + 
                    glob.glob(os.path.join(args.path, "*.jpg")))
    
    if len(images) < 2:
        print("Error: Need at least 2 images in the directory.")
        return

    # deal with number of image pairs to be processed
    num_pairs = min(len(images) - 1, args.num_pairs) if args.num_pairs > 0 else len(images) - 1
    print(f"Found {len(images)} images. Processing {num_pairs} pairs...")

    # establish output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    # 3. inference and time
    times_ms = [] 

    with torch.no_grad():
        # --- Warm up ---
        print("Warming up GPU...")
        im1_warm = load_image(images[0])
        im2_warm = load_image(images[1])
        padder = InputPadder(im1_warm.shape)
        im1_warm, im2_warm = padder.pad(im1_warm, im2_warm)
        _ = model(im1_warm, im2_warm, iters=args.iters, test_mode=True)
        torch.cuda.synchronize()
        print("Warm up done. Starting batch inference...\n")

        # --- main function ---
        print(f"{'Pair':<10} | {'Time (ms)':<10} | {'FPS':<10}")
        print("-" * 35)

        for i in range(num_pairs):
            # load Pair (i, i+1)
            image1 = load_image(images[i])
            image2 = load_image(images[i+1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # set timer
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            # inference
            results_dict = model(image1, image2, iters=args.iters, test_mode=True)
            ender.record()
            
            # synchornize and counting time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # ms
            times_ms.append(curr_time)

            # show progress
            print(f"Pair {i+1:02d}/{num_pairs:02d} | {curr_time:8.2f}   | {1000/curr_time:8.2f}")

            # get result and store
            flow_pr = results_dict['flow'][-1]
            flow = padder.unpad(flow_pr[0]).cpu()
            flow_np = flow.permute(1, 2, 0).numpy()
            
            # 1. compute Magnitude (dx, dy)
            dx = flow_np[:, :, 0]
            dy = flow_np[:, :, 1]
            mag = np.sqrt(dx**2 + dy**2)
            
            # 2. self-defined color transformation (Masking Approach)

            H, W = mag.shape
            custom_viz = np.zeros((H, W, 3), dtype=np.uint8) 
            
            # color demonstrate the level of pixel movement
            # zone 1 -> infinitesimal motion -> black
            # zone 1 -> moderate motion -> blue
            # zone 1 -> large motion -> orange
            # zone 1 -> extreme motion -> red
            
            # zone 1
            
            # zone 2
            mask_low = (mag >= 5) & (mag < 10)
            custom_viz[mask_low] = [255, 0, 0]
            
            # zone 3
            mask_mid = (mag >= 10) & (mag < 20)
            custom_viz[mask_mid] = [0, 165, 255] 
            
            # zone 4
            mask_high = (mag >= 25)
            custom_viz[mask_high] = [0, 0, 255] 
            
            # 3. save result
            out_mag_name = os.path.join(args.output, f"mag_{i:05d}.png")
            cv2.imwrite(out_mag_name, custom_viz)
            
    # 4. evaluation report
    times_ms = np.array(times_ms)
    avg_time = np.mean(times_ms)
    avg_fps = 1000 / avg_time
    
    print("\n" + "="*35)
    print("Inference Summary")
    print("="*35)
    print(f"Total Pairs  : {num_pairs}")
    print(f"Average Time : {avg_time:.2f} ms")
    print(f"Average FPS  : {avg_fps:.2f}")
    print(f"Output Dir   : {args.output}")
    print("="*35)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset path containing images", required=True)
    parser.add_argument('--output', help="output directory", default="tum_results")
    parser.add_argument('--num_pairs', type=int, default=10, help="number of pairs to process (default: 10)")
    parser.add_argument('--model', default="weights/flowseek_T_CT.pth", help="restore checkpoint")
    parser.add_argument('--cfg', default="config/eval/flowseek-T.json", help="config json path")
    parser.add_argument('--iters', type=int, default=3)

    args = parser.parse_args()

    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)

    demo(args)

