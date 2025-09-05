# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import sys
sys.path.append('core')

import argparse
import numpy as np
import random

from config.parser import parse_args
from flowseek import FlowSeek

import torch
import torch.optim as optim

from datasets import fetch_dataloader
from utils.utils import load_ckpt
from loss import sequence_loss
import tqdm
import os

os.system("export KMP_INIT_AT_FORK=FALSE")

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop """
    device_id = rank
    model = FlowSeek(args).to(device_id)

    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")
    model = torch.nn.DataParallel(model)
    model.cuda()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    with open('%s/command.txt'%args.savedir, 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(str(args))
        f.write('\n\n')

    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=False)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0
    should_keep_training = True

    while should_keep_training:
        epoch += 1
        for i_batch, data_blob in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(f'cuda:{model.device_ids[0]}') for x in data_blob] 
            output = model(image1, image2, flow_gt=flow, iters=args.iters)
            loss = sequence_loss(output, flow, valid, args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = '%s/%d_%s.pth' % (args.savedir, total_steps+1, args.name)
                torch.save(model.module.state_dict(), PATH)
            
            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            total_steps += 1

    PATH = '%s/%s.pth' % (args.savedir, args.name)
    if rank == 0:
        torch.save(model.module.state_dict(), PATH)

    return PATH

def main(rank, world_size, args, use_ddp):

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--restore_ckpt', help='restore previews weights', default=None)

    parser.add_argument('--savedir', help='enable Depth Anything v2', type=str)
    parser.add_argument('--seed', help='set random seed', type=float, default=0)
    args = parse_args(parser)

    # setting random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(0, 1, args, False)
    print("Done!")