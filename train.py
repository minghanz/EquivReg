import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import logging

import misc
import config

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of iterations'
                            'with exit code 2.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    cfg = misc.load_config(args.config, 'configs/default.yaml')

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    ### configure logger
    out_dir = config.cfg_f_out(cfg)

    ### configure dataset
    train_dataset, val_dataset, train_loader, val_loader, vis_loader, duo_loader = config.cfg_dataloader(cfg)
    
    ### configure model
    model = config.cfg_model(cfg, device)

    ### configure optimizer, lr scheduler, and loss functions
    trainer, optimizer, lr_scheduler = config.cfg_trainer(cfg, device, model)
    
    ### configure checkpoints
    checkpoint_io, epoch_it, it = config.cfg_checkpoint(cfg, out_dir, model, optimizer, lr_scheduler)
    writer = SummaryWriter(os.path.join(out_dir, 'logs'))
    
    ### configure callbacks
    callback_list, callback_dict = config.cfg_callbacks(cfg, trainer, vis_loader, val_loader, checkpoint_io, writer)
    
    # Shorthands
        
    # Iteration on epochs
    while True:
        epoch_it += 1
        train_iter = iter(train_loader)

        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            it += 1
            loss, d_loss = trainer.train_step(batch)
            writer.add_scalar('train/loss', loss, it)
            for key in d_loss:
                writer.add_scalar('train/{}'.format(key), d_loss[key], it)

            for callback in callback_list:
                callback_dict[callback](it=it, epoch_it=epoch_it, loss=loss, d_loss=d_loss)


        if args.exit_after > 0 and it > args.exit_after:
            logging.info(f'Exiting at {it} iterations.')
            break