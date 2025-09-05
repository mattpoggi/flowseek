# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('core')
import argparse
import torch
from config.parser import parse_args
from flowseek import FlowSeek

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parse_args(parser)
    model = FlowSeek(args)
    model.eval()
    h, w = [540, 960]
    input = torch.zeros(1, 3, h, w)
    model = model.cuda()
    input = input.cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True) as prof:
            output = model(input, input, iters=args.iters, test_mode=True)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: %.1f"% (forward_MACs / 2 / 1e9), "G")

if __name__ == '__main__':
    main()