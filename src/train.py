# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset, build_dataset_target_domain
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.plot_utils import get_vis_win_names
from trackformer.vis import build_visualizers
from trackformer.util.net_utils import FocalLoss
from torch.utils.tensorboard import SummaryWriter

ex = sacred.Experiment('train')
ex.add_config('cfgs/train.yaml')
ex.add_named_config('deformable', 'cfgs/train_deformable.yaml')
ex.add_named_config('tracking', 'cfgs/train_tracking.yaml')
ex.add_named_config('crowdhuman', 'cfgs/train_crowdhuman.yaml')
ex.add_named_config('mot17', 'cfgs/train_mot17.yaml')
ex.add_named_config('mot17_cross_val', 'cfgs/train_mot17_cross_val.yaml')
ex.add_named_config('mots20', 'cfgs/train_mots20.yaml')
ex.add_named_config('coco_person_masks', 'cfgs/train_coco_person_masks.yaml')
ex.add_named_config('full_res', 'cfgs/train_full_res.yaml')
ex.add_named_config('focal_loss', 'cfgs/train_focal_loss.yaml')

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train(args: Namespace) -> None:
    print(args)
    writer = SummaryWriter(f'runs/{args.name}')

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.debug:
        # args.tracking_eval = False
        args.num_workers = 0

    if not args.deformable:
        assert args.num_feature_levels == 1
    if args.tracking:
        assert args.batch_size == 1

        if args.tracking_eval:
            assert 'mot' in args.dataset

    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ["NCCL_TREE_THRESHOLD"] = "0"

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    visualizers = build_visualizers(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names + args.lr_linear_proj_names + ['layers_track_attention']) and p.requires_grad],
         "lr": args.lr,},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
         "lr":  args.lr * args.lr_linear_proj_mult}]
    if args.track_attention:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, ['layers_track_attention']) and p.requires_grad],
            "lr": args.lr_track})

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop])

    dataset_train_source = build_dataset(split='train', args=args)
    dataset_val_source = build_dataset(split='val', args=args)

    dataset_train_target = build_dataset_target_domain(split='train', args=args)
    dataset_val_target = build_dataset_target_domain(split='val', args=args)

    if args.distributed:
        sampler_train_source = utils.DistributedWeightedSampler(dataset_train_source)
        # sampler_train = DistributedSampler(dataset_train)
        sampler_val_source = DistributedSampler(dataset_val_source, shuffle=False)

        sampler_train_target = utils.DistributedWeightedSampler(dataset_train_target)
        sampler_val_target = DistributedSampler(dataset_val_target, shuffle=False)
    else:
        sampler_train_source = torch.utils.data.RandomSampler(dataset_train_source)
        sampler_val_source = torch.utils.data.SequentialSampler(dataset_val_source)

        sampler_train_target = torch.utils.data.RandomSampler(dataset_train_target)
        sampler_val_target = torch.utils.data.SequentialSampler(dataset_val_target)

    batch_sampler_train_source = torch.utils.data.BatchSampler(
        sampler_train_source, args.batch_size, drop_last=True)

    batch_sampler_train_target = torch.utils.data.BatchSampler(
        sampler_train_target, args.batch_size, drop_last=True)

    data_loader_train_source = DataLoader(
        dataset_train_source,
        batch_sampler=batch_sampler_train_source,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    data_loader_val_source = DataLoader(
        dataset_val_source, args.batch_size,
        sampler=sampler_val_source,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)

    data_loader_train_target = DataLoader(
        dataset_train_target,
        batch_sampler=batch_sampler_train_target,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    data_loader_val_target = DataLoader(
        dataset_val_target, args.batch_size,
        sampler=sampler_val_target,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    data_iter_train_target = iter(cycle(data_loader_train_target))

    best_val_stats = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        checkpoint_state_dict = {
            k.replace('detr.', ''): v for k, v in checkpoint['model'].items()}

        resume_state_dict = {}
        for k, v in model_state_dict.items():
            if k not in checkpoint_state_dict:
                resume_value = v
                print(f'Load {k} {tuple(v.shape)} from scratch.')
            elif v.shape != checkpoint_state_dict[k].shape:
                checkpoint_value = checkpoint_state_dict[k]
                num_dims = len(checkpoint_value.shape)

                if 'norm' in k:
                    resume_value = checkpoint_value.repeat(2)
                elif 'multihead_attn' in k or 'self_attn' in k:
                    resume_value = checkpoint_value.repeat(num_dims * (2, ))
                elif 'linear1' in k or 'query_embed' in k:
                    if checkpoint_value.shape[1] * 2 == v.shape[1]:
                        # from hidden size 256 to 512
                        resume_value = checkpoint_value.repeat(1, 2)
                    elif checkpoint_value.shape[0] * 5 == v.shape[0]:
                        # from 100 to 500 object queries
                        resume_value = checkpoint_value.repeat(5, 1)
                    elif checkpoint_value.shape[0] > v.shape[0]:
                        resume_value = checkpoint_value[:v.shape[0]]
                    elif checkpoint_value.shape[0] < v.shape[0]:
                        resume_value = v
                    else:
                        raise NotImplementedError
                elif 'linear2' in k or 'input_proj' in k:
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
                elif 'class_embed' in k:
                    # person and no-object class
                    # resume_value = checkpoint_value[[1, -1]]
                    # resume_value = checkpoint_value[[0, -1]]
                    # resume_value = checkpoint_value[[1,]]
                    resume_value = v
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

                print(f"Load {k} {tuple(v.shape)} from resume model "
                      f"{tuple(checkpoint_value.shape)}.")
            elif args.resume_shift_neuron and 'class_embed' in k:
                checkpoint_value = checkpoint_state_dict[k]
                # no-object class
                resume_value = checkpoint_value.clone()
                # no-object class
                # resume_value[:-2] = checkpoint_value[1:-1].clone()
                resume_value[:-1] = checkpoint_value[1:].clone()
                resume_value[-2] = checkpoint_value[0].clone()
                print(f"Load {k} {tuple(v.shape)} from resume model and "
                      "shift class embed neurons to start with label=0 at neuron=0.")
            else:
                resume_value = checkpoint_state_dict[k]

            resume_state_dict[k] = resume_value

        if args.masks and args.load_mask_head_from_model is not None:
            checkpoint_mask_head = torch.load(
                args.load_mask_head_from_model, map_location='cpu')

            for k, v in resume_state_dict.items():

                if (('bbox_attention' in k or 'mask_head' in k)
                    and v.shape == checkpoint_mask_head['model'][k].shape):
                    print(f'Load {k} {tuple(v.shape)} from mask head model.')
                    resume_state_dict[k] = checkpoint_mask_head['model'][k]

        model_without_ddp.load_state_dict(resume_state_dict)

        # RESUME OPTIM
        if not args.eval_only and args.resume_optim:
            if 'optimizer' in checkpoint:
                for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                    c_p['lr'] = p['lr']

                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1

            best_val_stats = checkpoint['best_val_stats']

        # RESUME VIS
        if not args.eval_only and args.resume_vis and 'vis_win_names' in checkpoint:
            for k, v in visualizers.items():
                for k_inner in v.keys():
                    visualizers[k][k_inner].win = checkpoint['vis_win_names'][k][k_inner]

    if args.eval_only:
        _, coco_evaluator, _ = evaluate(
            model, criterion, postprocessors, data_loader_val_target, device,
            output_dir, visualizers['val'], args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        return

    FL1 = FocalLoss(class_num=2)
    FL2 = FocalLoss(class_num=2)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        # TRAIN
        if args.distributed:
            sampler_train_source.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, postprocessors, data_loader_train_source, data_iter_train_target, optimizer, device, epoch,
            FL1, FL2, visualizers['train'], args)

        if args.eval_train:
            random_transforms = data_loader_train_source.dataset._transforms
            data_loader_train_source.dataset._transforms = data_loader_val_source.dataset._transforms
            evaluate(
                model, criterion, postprocessors, data_loader_train_source, device,
                output_dir, visualizers['train'], args, epoch)
            data_loader_train_source.dataset._transforms = random_transforms

        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']

        # VAL
        val_stats = {}
        if epoch == 1 or not epoch % args.val_interval:
            val_stats, _, val_stats_all = evaluate(
                model, criterion, postprocessors, data_loader_val_target, device,
                output_dir, visualizers['val'], args, epoch)

            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
            #     checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            # checkpoint for best validation stats
            stat_names = ['BBOX_AP_IoU_0_50-0_95', 'BBOX_AP_IoU_0_50', 'BBOX_AP_IoU_0_75']
            if args.masks:
                stat_names.extend(['MASK_AP_IoU_0_50-0_95', 'MASK_AP_IoU_0_50', 'MASK_AP_IoU_0_75'])
            if args.tracking and args.tracking_eval:
                stat_names.extend(['MOTA', 'IDF1'])

            if best_val_stats is None:
                best_val_stats = val_stats
            best_val_stats = [best_stat if best_stat > stat else stat
                              for best_stat, stat in zip(best_val_stats,
                                                         val_stats)]
            for b_s, s, n in zip(best_val_stats, val_stats, stat_names):
                if b_s == s:
                    checkpoint_paths.append(output_dir / f"checkpoint_best_{n}.pth")
        if utils.is_main_process():
            for k, v in train_stats.items():
                writer.add_scalar(f'{k}/train', v, epoch)
            for k, v in val_stats_all.items():
                if 'coco' in k:
                    writer.add_scalar(f'mAP/val_target', v[0], epoch)
                    writer.add_scalar(f'AP@0.50/val_target', v[1], epoch)
                    writer.add_scalar(f'AP@0.75/val_target', v[2], epoch)
                elif 'track_bbox' in k:
                    writer.add_scalar(f'mota/val_target', v[0], epoch)
                    writer.add_scalar(f'idf1/val_target', v[1], epoch)
                else:
                    writer.add_scalar(f'{k}/val_target', v, epoch)

        # MODEL SAVING
        if args.output_dir:
            if args.save_model_interval and not epoch % args.save_model_interval:
                checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'vis_win_names': get_vis_win_names(visualizers),
                    'best_val_stats': best_val_stats
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    # TODO: hierachical Namespacing for nested dict
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    # args.train = Namespace(**config['train'])
    train(args)
