import os
import yaml
import logging
import torchvision.transforms
import torch
import torch.distributions as dist
import torch.optim as optim
import numpy as np

import misc
import field_tfs
import fields
import dataset
import models
import encoders
import training
import checkpoints
import callbacks
import generation
import transforms

def cfg_f_out_test(cfg):
    out_dir = cfg['training']['out_dir']
    gen_dir = os.path.join(out_dir, cfg['testing']['out_dir'])

    # Output directory
    os.makedirs(gen_dir, exist_ok=True)

    # Set up the logging format and output path
    misc.setup_logging(gen_dir)

    # write cfg to file for record
    with open(os.path.join(gen_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
        logging.info("cfg is saved at {}".format(os.path.join(gen_dir, 'cfg.yaml')))
        logging.info(cfg)

    return out_dir, gen_dir

def cfg_f_out(cfg):
    out_dir = cfg['training']['out_dir']

    # Output directory
    os.makedirs(out_dir, exist_ok=True)

    # Set up the logging format and output path
    misc.setup_logging(out_dir)

    # write cfg to file for record
    with open(os.path.join(out_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
        logging.info("cfg is saved at {}".format(os.path.join(out_dir, 'cfg.yaml')))
        logging.info(cfg)

    return out_dir

def cfg_occ_field(cfg_data, mode):
    ''' Returns the data fields: 'points', (may exist for val/test) 'points_iou', 'voxels'

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    cfg_occ = cfg_data['occ']

    points_transform = field_tfs.SubsamplePoints(cfg_occ['points_subsample'])

    data_fields = {}
    data_fields['points'] = fields.PointsField(
        cfg_occ['points_file'], points_transform,
        unpackbits=cfg_occ['points_unpackbits'],
    )

    if mode in ('val', 'test', 'vis'):
        points_iou_file = cfg_occ['points_iou_file']
        if points_iou_file is not None:
            data_fields['points_iou'] = fields.PointsField(
                points_iou_file,
                unpackbits=cfg_occ['points_unpackbits'],
            )

        voxels_file = cfg_occ['voxels_file']
        if voxels_file is not None:
            data_fields['voxels'] = fields.VoxelsField(voxels_file)

    return data_fields

def cfg_inputs_field(cfg_data, mode, inputs_field):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    cfg_data_mode = cfg_data[mode]
    cfg_data_train = cfg_data['train']

    input_type = 'pointcloud' #cfg_data['input_type']
    duo_mode = False
    # duo_mode = cfg_data['duo_mode']
    reg_benchmark_mode = False
    reg_mode = cfg_data_mode.get('reg', cfg_data_train['reg'])
    if mode == 'test':
        reg_benchmark_mode = cfg_data_mode['reg_benchmark']
    
    if input_type == 'pointcloud':
        if reg_benchmark_mode:
            inputs_field['inputs'] = fields.PointCloudField(
                cfg_data['input_bench']['pointcloud_file_1'], transform=None,
            )
            inputs_field['inputs_2'] = fields.PointCloudField(
                cfg_data['input_bench']['pointcloud_file_2'], transform=None,
            )
            inputs_field['T21'] = fields.RotationField(
                cfg_data['input_bench']['T21_file']
            )
        else:
            # assert reg_mode
            if reg_mode:
                pointcloud_n = cfg_data_mode.get('presamp_n', cfg_data_train['presamp_n'])
                transform = field_tfs.SubsamplePointcloud(pointcloud_n)
            else:
                pointcloud_n = cfg_data_mode.get('presamp_n', cfg_data_train['presamp_n'])
                noise = cfg_data_mode.get('noise', cfg_data_train['noise'])
                transform = torchvision.transforms.Compose([
                    field_tfs.SubsamplePointcloud(pointcloud_n),
                    field_tfs.PointcloudNoise(noise),
                ])
            inputs_field['inputs'] = fields.PointCloudField(
                cfg_data['input']['pointcloud_file'], transform,
            )
            if duo_mode:
                inputs_field['T'] = fields.TransformationField(cfg_data['input']['T_file'])
    return

def cfg_dataset(cfg, mode):
    cfg_data = cfg['data']
    
    cfg_data_mode = cfg_data[mode]
    cfg_data_train = cfg_data['train']

    reg_benchmark_mode = False
    reg_mode = cfg_data_mode.get('reg', cfg_data_train['reg'])
    if mode == 'test':
        reg_benchmark_mode = cfg_data_mode['reg_benchmark']

    dataset_folder = cfg_data['input_bench']['path'] if reg_benchmark_mode else cfg_data['input']['path']
    duo_mode = False

    categories = None
    if mode == 'vis':
        split = cfg_data_mode['split']
    else:
        split = mode

    data_field = dict() if reg_benchmark_mode else cfg_occ_field(cfg_data, mode)
    cfg_inputs_field(cfg_data, mode, data_field)

    if mode == 'test':
        data_field['idx'] = fields.IndexField()
    # data_field['category'] = data.CategoryField()

    if reg_mode:
        output_dataset = dataset.Shapes3dDataset(
            dataset_folder, data_field,
            split=split,
            # categories=categories,
            )
        rot_magmax = cfg_data_mode.get('rotate', cfg_data_train['rotate'])
        pcl_noise = cfg_data_mode.get('noise', cfg_data_train['noise'])
        resamp_mode = cfg_data_mode.get('resamp', cfg_data_train['resamp'])
        output_dataset = dataset.PairedDataset(output_dataset, rot_magmax, duo_mode, reg_benchmark_mode, resamp_mode, pcl_noise)
    else:
        rot_magmax = cfg_data_mode.get('rotate', cfg_data_train['rotate'])
        output_dataset = dataset.Shapes3dDataset(
            dataset_folder, data_field,
            split=split,
            # categories=categories,
            rot_magmax=rot_magmax,
            )
        
    return output_dataset

def cfg_dataloader(cfg):
    config_dataloader = cfg['dataloader']

    # Dataset
    train_dataset = cfg_dataset(cfg, 'train')
    val_dataset = cfg_dataset(cfg, 'val')
    vis_dataset = cfg_dataset(cfg, 'vis')

    batch_size = config_dataloader['train']['batch_size']
    num_workers = config_dataloader['train']['num_workers']
    
    if isinstance(train_dataset, list):
        # Mix two datasets (compared with ConcatDataset, we want one batch to only include data from one dataset. )
        duo_loader = True
        pass
    else:
        duo_loader = False
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
            collate_fn=dataset.collate_remove_none,
            worker_init_fn=dataset.worker_init_fn)
        
    batch_size_val = config_dataloader['val']['batch_size']
    num_workers_val = config_dataloader['val']['num_workers']

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val, num_workers=num_workers_val, shuffle=False,
        collate_fn=dataset.collate_remove_none,
        worker_init_fn=dataset.worker_init_fn)

    batch_size_vis = config_dataloader['vis']['batch_size']

    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        vis_dataset, batch_size=batch_size_vis, shuffle=False,
        collate_fn=dataset.collate_remove_none,
        worker_init_fn=dataset.worker_init_fn)
    # vis_iter = iter(vis_loader)

    return train_dataset, val_dataset, train_loader, val_loader, vis_loader, duo_loader

def cfg_prior_z(config_model, device):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = config_model['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z

def cfg_model(cfg, device):
    config_model = cfg['model']

    decoder = config_model['decoder']
    encoder = config_model['encoder']
    encoder_latent = config_model['encoder_latent']
    dim = 3 #cfg['data']['dim']
    z_dim = config_model['z_dim']
    c_dim = config_model['c_dim']
    decoder_kwargs = config_model['decoder_kwargs']
    encoder_kwargs = config_model['encoder_kwargs']
    encoder_latent_kwargs = config_model['encoder_latent_kwargs']

    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )
    
    ### in our case, z_dim == 0
    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs)
    else:
        encoder_latent = None

    encoder = encoders.encoder_dict[encoder](
        c_dim=c_dim,
        **encoder_kwargs)
    
    p0_z = cfg_prior_z(config_model, device)

    model = models.OccupancyNetwork(
        decoder, encoder, encoder_latent, p0_z, device=device)
    
    if torch.cuda.device_count() > 1:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    nparameters = sum(p.numel() for p in model.parameters())
    # logging.info(model)
    logging.info('Total number of parameters: %d' % nparameters)

    return model

def cfg_batchprep(config_data, mode, device):
    cfg_data_mode = config_data[mode]
    cfg_data_train = config_data['train']

    reg = cfg_data_mode.get('reg', cfg_data_train['reg'])
    reg_benchmark_mode = False
    if mode == 'test':
        reg_benchmark_mode = cfg_data_mode['reg_benchmark']

    # config_batchprep = cfg['batch_prep'][mode]
    # config_batchprep_train = cfg['batch_prep']['train']

    subsamp = cfg_data_mode.get('subsamp', cfg_data_train['subsamp'])
    n2_min = cfg_data_mode.get('n2_min', cfg_data_train['n2_min'])
    n2_max = cfg_data_mode.get('n2_max', cfg_data_train['n2_max'])
    centralize = cfg_data_mode.get('centralize', cfg_data_train['centralize'])
    op_list = []
    if reg_benchmark_mode:
        pass
    elif reg:
        n1 = cfg_data_mode.get('n1', cfg_data_train['n1'])
        if subsamp:
            sub_op = transforms.SubSamplePairBatchIP(n1, n2_min, n2_max, device)
            op_list.append(sub_op)
        if centralize:
            ctr_op = transforms.CentralizePairBatchIP()
            op_list.append(ctr_op)
    else:
        if subsamp:
            sub_op = transforms.SubSampleBatchIP(n2_min, n2_max, device)
            op_list.append(sub_op)
        if centralize:
            ctr_op = transforms.CentralizeBatchIP()
            op_list.append(ctr_op)
    transform = torchvision.transforms.Compose(op_list)
    return transform

def cfg_trainer(cfg, device, model):
    config_training = cfg['training']

    lr = config_training['lr']
    logging.info("learning rate: {}".format(lr))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_schedule = config_training['lr_schedule']
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, lr_schedule) if lr_schedule is not None else None

    out_dir = config_training['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    config_data = cfg['data']
    transform_train = cfg_batchprep(config_data, 'train', device)
    transform_val = cfg_batchprep(config_data, 'val', device)
    transform_vis = cfg_batchprep(config_data, 'vis', device)
    reg_train = config_data['train']['reg']

    config_trainer = cfg['trainer']
    if reg_train:
        trainer = training.DualTrainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device, input_type='pointcloud',
            vis_dir=vis_dir, 
            transform_train=transform_train,
            transform_val=transform_val,
            transform_vis=transform_vis,
            **config_trainer
        )
    else:
        trainer = training.Trainer(
            model, optimizer,
            lr_scheduler=lr_scheduler,
            device=device, input_type='pointcloud',
            vis_dir=vis_dir, 
            transform_train=transform_train,
            transform_val=transform_val,
            transform_vis=transform_vis,
            **config_trainer
        )

    return trainer, optimizer, lr_scheduler

def cfg_generator(cfg, device, model):
    config_data = cfg['data']

    transform_test = cfg_batchprep(config_data, 'test', device)

    config_tester = cfg['tester']
    
    generator = generation.Generator3D(
        model,
        device=device,
        transform_test=transform_test,
        **config_tester
    )
    return generator


def cfg_checkpoint(cfg, out_dir, model, optimizer, lr_scheduler):
    config_checkpoint = cfg['checkpoint']
    
    checkpoint_io = checkpoints.CheckpointIO(model, optimizer, lr_scheduler, out_dir)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    model_selection_metric = config_checkpoint['model_selection_metric']
    if config_checkpoint['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif config_checkpoint['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                        'either maximize or minimize.')
    
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)
    
    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    checkpoint_io.set_selection_criteria(model_selection_metric, model_selection_sign, metric_val_best)

    logging.info('Current best validation metric (%s): %.8f'
        % (model_selection_metric, metric_val_best))
    
    return checkpoint_io, epoch_it, it

def cfg_callbacks(cfg, trainer, vis_loader, val_loader, checkpoint_io, writer):
    config_callback = cfg['callback']

    print_every = config_callback['print_every']
    visualize_every = config_callback['visualize_every']
    validate_every = config_callback['validate_every']
    checkpoint_every = config_callback['checkpoint_every']
    autosave_every = config_callback['autosave_every']

    callback_list = []
    callback_dict = dict()
    if print_every > 0:
        callback_list.append('print')
        callback_dict['print'] = callbacks.PrintCallback(print_every)
    if visualize_every > 0:
        callback_list.append('visualize')
        callback_dict['visualize'] = callbacks.VisualizeCallback(visualize_every, trainer, vis_loader)
    if validate_every > 0:
        callback_list.append('validation')
        callback_dict['validation'] = callbacks.ValidationCallback(
            validate_every, checkpoint_io, trainer, val_loader, writer)
    if checkpoint_every > 0:
        callback_list.append('checkpoint')
        callback_dict['checkpoint'] = callbacks.CheckpointsaveCallback(checkpoint_every, checkpoint_io)
    if autosave_every > 0:
        callback_list.append('autosave')
        callback_dict['autosave'] = callbacks.AutosaveCallback(autosave_every, checkpoint_io)

    return callback_list, callback_dict
