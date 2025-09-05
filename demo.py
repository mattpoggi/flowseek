# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from flowseek import *
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True) 
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    return flow_down #, info_down

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt, val=None):
    path = f"demo_qualitatives/{name}/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    flow = calc_flow(args, model, image1, image2)

    diff = flow_gt - flow
    diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    if val is not None:
        print("EPE: %.3f"%(epe[0][val>0].mean().cpu().item()))
    else:
        print("EPE: %.3f"%(epe[0].mean().cpu().item()))

    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)

@torch.no_grad()
def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = datasets.MpiSintel(split='training', dstype=dstype, root=args.paths['sintel'])
    image1, image2, flow_gt, _ = dataset[args.id]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_kitti(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = datasets.KITTI(split='training', root=args.paths['kitti'])
    image1, image2, flow_gt, val = dataset[args.id]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('kitti', args, model, image1, image2, flow_gt, val)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = datasets.SpringFlowDemoDataset(split=split, root=args.paths['spring'])
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[args.id]
    else:
        image1, image2,  _ = dataset[args.id]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_layeredflow(model, args, device=torch.device('cuda')):
    dataset = datasets.LayeredFlow(root=args.paths['layeredflow'])
    image1, image2, coords, flow_gt, materials, layers = dataset[args.id]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = image1[:,0:2].to(device) * 0.
    demo_data('layeredflow', args, model, image1, image2, flow_gt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str, required=True)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--dataset', help='dataset type', type=str, required=True)
    parser.add_argument('--id', help='image id', type=int, default=0)

    args = parse_args(parser)
    model = FlowSeek(args)

    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    if args.dataset == 'sintel':
        demo_sintel(model, args)
    elif args.dataset == 'kitti':
        demo_kitti(model, args)  
    elif args.dataset == 'spring':
        demo_spring(model, args, split='train')
    elif args.dataset == 'layeredflow':
        demo_layeredflow(model, args)    

if __name__ == '__main__':
    main()