import torch
import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import misc
import config
from checkpoints import CheckpointIO
from metricrecord import Metric, Record

from transforms import apply_rot
from register_utils import *

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    cfg = misc.load_config(args.config, 'configs/default.yaml')

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir, gen_dir = config.cfg_f_out_test(cfg)

    dataset = config.cfg_dataset(cfg, 'test')
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=3, shuffle=False)

    model = config.cfg_model(cfg, device)

    checkpointio = CheckpointIO(model, checkpoint_dir=out_dir)
    checkpointio.load(cfg['testing']['model_file'])

    generator = config.cfg_generator(cfg, device, model)
    
    metric_dict = dict()
    metric_keys = ['angle', 'angle_180', 'angle_90-', 'angle_90+', 'rmse', 'rmse_tp']
    for key in metric_keys:
        metric_dict[key] = Metric(key)

    worst_dict = dict()
    worst_dict['angle'] = Record('angle', 10, True)
    worst_dict['angle_180'] = Record('angle_180', 10, True)
    best_dict = dict()
    best_dict['angle'] = Record('angle', 10, False)
    
    # evaluate
    for it, data in enumerate(tqdm(test_loader)):
         
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)

        out_1, out_2_rot = generator.generate_latent_conditioned(data)
        batch_size = out_1.shape[0]
        out_1 = out_1.reshape(batch_size, -1, 3)
        out_2_rot = out_2_rot.reshape(batch_size, -1, 3)

        out_1 = out_1.to(torch.float64)
        out_2_rot = out_2_rot.to(torch.float64)
        
        R_est = solve_R(out_1, out_2_rot)

        ##### angular error
        R_gt = data['T21'].to(torch.float64)
        rotdeg = data['T21.deg']

        angle_diff = angle_diff_func(R_est, R_gt)
        angle_diff = torch.abs(angle_diff).max().item()
        rotdeg = torch.abs(rotdeg).max().item()
        logging.debug("angle_diff (res, gt) %.4f %.4f"%(angle_diff, rotdeg ) )

        ##### equivariant embedding error
        out_1_rot = apply_rot(R_gt, out_1)
        out_1_rot_est = apply_rot(R_est, out_1)

        diff_ori = out_1 - out_2_rot
        diff_gt  = out_1_rot - out_2_rot
        diff_est = out_1_rot_est - out_2_rot

        diff_gt_inf = torch.abs(diff_gt).max().item()
        diff_ori_inf = torch.abs(diff_ori).max().item()
        diff_est_inf = torch.abs(diff_est).max().item()
        
        diff_gt_l2 = torch.norm(diff_gt, dim=2).mean().item()     # == torch.sqrt((diff_gt**2).sum())
        diff_ori_l2 = torch.norm(diff_ori, dim=2).mean().item()
        diff_est_l2 = torch.norm(diff_est, dim=2).mean().item()

        logging.debug("feat_diff_Linf (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_inf, diff_est_inf, diff_gt_inf) )
        logging.debug("feat_diff_L2 (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_l2, diff_est_l2, diff_gt_l2) )

        ### input alignment error
        input_1 = data['inputs']
        input_2_rot = data['inputs_2']
        input_1_rot = apply_rot(R_gt, input_1)

        # ### known issue: gpu 64 == cpu 32 == cpu 64 == numpy 32, but is slightly different from gpu 32
        # input_1_rot_64 = apply_rot(R_gt.to(torch.float64), input_1.to(torch.float64)).to(torch.float32)
        # input_1_rot_cpu = apply_rot(R_gt.cpu(), input_1.cpu())
        # input_1_rot_cpu_64 = apply_rot(R_gt.cpu().to(torch.float64), input_1.cpu().to(torch.float64)).to(torch.float32)
        # input_1_rot_np = np.matmul(R_gt.detach().cpu().numpy()[0], input_1.detach().cpu().numpy()[0].swapaxes(-1, -2))[None, ...].swapaxes(-1, -2)

        input_1_rot_est = apply_rot(R_est, input_1)
        input_2 = apply_rot(torch.inverse(R_gt), input_2_rot)

        if input_1.shape == input_2_rot.shape:
            diff_pts_rmse_ori = torch.norm(input_1 - input_2_rot, dim=2).mean()
            diff_pts_rmse_gt = torch.norm(input_1_rot - input_2_rot, dim=2).mean()
            diff_pts_rmse_est = torch.norm(input_1_rot_est - input_2_rot, dim=2).mean()
            logging.debug("diff_pts_rmse (ori, est, gt) %.4f %.4f %.4f"%(diff_pts_rmse_ori, diff_pts_rmse_gt, diff_pts_rmse_est) )

        # update the metrics
        metric_dict['angle'].update(angle_diff)
        if angle_diff > 90:
            metric_dict['angle_180'].update(180-angle_diff)
            metric_dict['angle_90+'].update(180-angle_diff)
        else:
            metric_dict['angle_180'].update(angle_diff)
            metric_dict['angle_90-'].update(angle_diff)

        if input_1.shape == input_2_rot.shape:
            metric_dict['rmse'].update(diff_pts_rmse_est)
            if diff_pts_rmse_est < 0.2:
                metric_dict['rmse_tp'].update(diff_pts_rmse_est)

        ds_cur = dict()
        ds_cur['pcl_1'] = input_1.squeeze(0).cpu().numpy()   # N*3
        ds_cur['pcl_1_rot_est'] = input_1_rot_est.squeeze(0).cpu().numpy()   # N*3
        ds_cur['pcl_1_rot_gt'] = input_1_rot.squeeze(0).cpu().numpy()   # N*3
        ds_cur['pcl_2'] = input_2_rot.squeeze(0).cpu().numpy()   # N*3
        # ds_cur['category_name'] = str(category_id)

        worst_dict['angle'].update(angle_diff, ds_cur)
        if angle_diff > 90:
            worst_dict['angle_180'].update(180-angle_diff, ds_cur)
        else:
            worst_dict['angle_180'].update(angle_diff, ds_cur)
        best_dict['angle'].update(angle_diff, ds_cur)

        # break

    for key in metric_dict:
        logging.info(metric_dict[key])