import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

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

    # 1. initialize model
    print(f"Loading model: {args.model}")
    model = FlowSeek(args)
    
    # 2. read checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    
    # check whether packaged in 'state_dict'
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 3. deal with 'module.' 
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 4. load processed weight
    model.load_state_dict(new_state_dict, strict=False) # strict=False for tiny error tolerance
    
    # 5. transfer model to GPU
    model.to(DEVICE)
    model.eval()

    # 2. read input images
    print(f"Image 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    
    image1 = load_image(args.img1)
    image2 = load_image(args.img2)

    # 3. Padding 
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    # 4. inference
    with torch.no_grad():
        # Warm-up) 
        print("Warm up...")
        _ = model(image1, image2, iters=args.iters, test_mode=True)
        
        # set CUDA timer
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        print("Running inference...")
        starter.record() 
        
        # execution
        results_dict = model(image1, image2, iters=args.iters, test_mode=True)
        
        ender.record()  
        
        torch.cuda.synchronize()
        
        curr_time = starter.elapsed_time(ender)
        print(f"⏱️ Inference Time: {curr_time:.2f} ms")
        print(f"🚀 FPS: {1000/curr_time:.2f}")

        flow_pr = results_dict['flow'][-1]
        flow = padder.unpad(flow_pr[0]).cpu()

    # 5. store and visiualization
    flow_np = flow.permute(1, 2, 0).numpy()
    
    # transfer to color flow figure
    flow_img = flow_viz.flow_to_image(flow_np)
    
    # svae file
    output_path = "result_flow.png"
    cv2.imwrite(output_path, flow_img[:, :, ::-1]) # RGB -> BGR
    print(f"optical flow result saved !!! : {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # necessary parmeters
    parser.add_argument('--img1', type=str, required=True, help="path to image 1")
    parser.add_argument('--img2', type=str, required=True, help="path to image 2")
    
    # optinal parameters
    parser.add_argument('--model', default="weights/flowseek_T_CT.pth", help="path to model weights")
    parser.add_argument('--cfg', default="config/eval/flowseek-T.json", help="path to Config JSON")
    parser.add_argument('--iters', type=int, default=4)

    args = parser.parse_args()

    # --- load JSON Config ---
    # for initialzation (dropout, hidden_dim ...)
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            config = json.load(f)
        # inject config parameters to args
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    else:
        print(f"Warning: Config file {args.cfg} not found! Model might fail to initialize.")

    demo(args)

