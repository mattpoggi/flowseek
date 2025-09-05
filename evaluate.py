# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('core')
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from flowseek import *
from tqdm import tqdm
from utils.utils import resize_data, load_ckpt

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
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, root=args.paths['sintel'])
        val_loader = data.DataLoader(val_dataset, batch_size=4, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        for i_batch, data_blob in enumerate(tqdm(val_loader)):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            flow, info = calc_flow(args, model, image1, image2)
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
            px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
            px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
            epe = epe.mean(dim=[1, 2]).cpu().numpy()
            epe_list = np.append(epe_list, epe)
            px1_list = np.append(px1_list, px1)
            px3_list = np.append(px3_list, px3)
            px5_list = np.append(px5_list, px5)

        epe = np.mean(epe_list)
        px1 = np.mean(px1_list)
        px3 = np.mean(px3_list)
        px5 = np.mean(px5_list)
        # print("Validation %s EPE: %.2f, 1px: %.2f"%(dstype,epe,100 * (1 - px1)))
        print("Validation %s EPE: %.2f"%(dstype,epe,))

@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = datasets.KITTI(split='training', root=args.paths['kitti'])
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        val = valid_gt >= 0.5
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        for b in range(out.shape[0]):
            epe_list = np.append(epe_list, epe[b][val[b]].mean().cpu().numpy())
            out_valid_pixels += out[b][val[b]].sum().cpu().numpy()
            num_valid_pixels += val[b].sum().cpu().numpy()
    
    epe = np.mean(epe_list)
    f1 = 100 * out_valid_pixels / num_valid_pixels
    print("Validation KITTI: %.2f, %.1f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = datasets.SpringFlowDataset(split='train', root=args.paths['spring'])
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=8, drop_last=False)
    
    epe_list = np.array([], dtype=np.float32)
    px1_list = np.array([], dtype=np.float32)
    px3_list = np.array([], dtype=np.float32)
    px5_list = np.array([], dtype=np.float32)
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
        px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
        px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
        epe = epe.mean(dim=[1, 2]).cpu().numpy()
        epe_list = np.append(epe_list, epe)
        px1_list = np.append(px1_list, px1)
        px3_list = np.append(px3_list, px3)
        px5_list = np.append(px5_list, px5)

    epe = np.mean(epe_list)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print("Validation Spring EPE: %.3f, 1px: %.3f"%(epe,100 * (1 - px1)))

def validate_layeredflow_first(args, model):
    """ Peform validation using the LayeredFlow (val) split """
    def datapoint_in_subset(mat, layer, subset):
        def in_list(x, l):
            return l is None or x in l
        assert type(subset) == tuple and len(subset) == 2
        return in_list(mat, subset[0]) and in_list(layer, subset[1])
        
    model.eval()
    val_dataset = datasets.LayeredFlow(downsample=8, split='val', root=args.paths['layeredflow'])

    subsets = [
        (None, (0,)), # first layer
        ((1,), (0,)), # first layer, material transparent
        ((2,), (0,)), # first layer, material reflective
        ((0,), (0,)), # first layer, material diffuse
    ]

    bad_n = [1, 3, 5]
    results = {}
    
    for subset in subsets:
        results[subset] = {}
        results[subset]['epe'] = []
        for n in bad_n:
            results[subset][str(n) + 'px'] = []

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, coords, flow_gts, materials, layers = val_dataset[val_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        output = model(image1, image2, test_mode=True, demo=True)
        flow_final = output['flow'][-1]
        flow = padder.unpad(flow_final).cpu()[0]
        
        error_list = {}
        for subset in subsets:
            error_list[subset] = []
        
        for i in range(len(coords)):
            (x, y), mat, lay = coords[i], materials[i], layers[i]

            flow_pd = flow[:, x, y]
            flow_gt = torch.tensor(flow_gts[i])
            error = torch.sum((flow_pd - flow_gt)**2, dim=0).sqrt().item()

            for subset in subsets:
                if datapoint_in_subset(mat, lay, subset):
                    error_list[subset].append(error)

        for subset in subsets:
            if len(error_list[subset]) == 0:
                continue
            error_list[subset] = np.array(error_list[subset])
            results[subset]['epe'].append(np.mean(error_list[subset]))
            for n in bad_n:
                results[subset][str(n) + 'px'].extend(error_list[subset] < n)

    for subset in subsets:
        print(f"Validation LayeredFlow {subset}:")
        for key in results[subset]:
            results[subset][key] = np.mean(results[subset][key])
            if key != 'epe':
                results[subset][key] = 100 - 100 * results[subset][key]
            print(f"{key}: %.2f"%results[subset][key])

    return results

def eval(args):
    args.gpus = [0]
    model = FlowSeek(args)
    if args.model is not None:
        load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, model)
        elif args.dataset == 'sintel':
            validate_sintel(args, model)
        elif args.dataset == 'kitti':
            validate_kitti(args, model)
        elif args.dataset == 'layeredflow':
            validate_layeredflow_first(args, model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--dataset', help='dataset type', type=str, required=True)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()

