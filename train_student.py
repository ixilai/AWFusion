import os
import sys
import argparse
import pathlib
import logging.config
from model.ckt import CKTModule
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional
import torch.nn as nn
import cv2, datetime, time
from model.AWFusion_module import AWF
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from dataloader.fuse_data_vsm import GetDataset_type4
from utils22.loss_vif import fusion_loss_vif
torch.autograd.set_detect_anomaly(True)

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')
    # dataset
    parser.add_argument('--haze', default='./Train/Haze/vi', type=str)
    parser.add_argument('--haze_GT', default='./Train/Haze/gt_vi', type=str)
    parser.add_argument('--haze_ir', default='./Train/Haze/ir', type=str)
    parser.add_argument('--haze_ir_GT', default='./Train/Haze/gt_ir', type=str)

    parser.add_argument('--rain', default='./Train/Rain/vi', type=str)
    parser.add_argument('--rain_GT', default='./Train/Rain/gt_vi', type=str)
    parser.add_argument('--rain_ir', default='./Train/Rain/ir', type=str)
    parser.add_argument('--rain_ir_GT', default='./Train/Rain/gt_ir', type=str)

    parser.add_argument('--snow', default='./Train/Snow/vi', type=str)
    parser.add_argument('--snow_GT', default='./Train/Snow/gt_vi', type=str)
    parser.add_argument('--snow_ir', default='./Train/Snow/ir', type=str)
    parser.add_argument('--snow_ir_GT', default='./Train/Snow/gt_ir', type=str)
    # train loss weights
    parser.add_argument('--t_dehaze_ckpt',
                        default='./ckpt_haze/Fuse_haze.pth',
                        )
    parser.add_argument('--t_derain_ckpt',
                        default='./ckpt_rain/Fuse_rain.pth',
                        )
    parser.add_argument('--t_desnow_ckpt',
                        default='./ckpt_snow/Fuse_snow.pth',
                        )
    # implement details
    parser.add_argument('--size', default = 96, type=int, help='图片裁剪大小')
    parser.add_argument('--batchsize', default=1, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument("--step", type=int, default=200,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--interval', default=2, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_fuse", default='./ckpt_student/Allweather_Fuse.pth', help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt', default='./ckpt_student', help='checkpoint cache folder')

    args = parser.parse_args()
    return args


def main(args):

    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    interval = args.interval
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    data_train = GetDataset_type4(split='train', size=args.size,
                                  haze=args.haze, haze_GT=args.haze_GT, haze_ir=args.haze_ir, haze_ir_GT=args.haze_ir_GT,
                                  rain=args.rain, rain_GT=args.rain_GT, rain_ir=args.rain_ir, rain_ir_GT=args.rain_ir_GT,
                                  snow=args.snow, snow_GT=args.snow_GT, snow_ir=args.snow_ir, snow_ir_GT=args.snow_ir_GT)
    train_sampler = DistributedSampler(data_train)
    training_data_loader = torch.utils.data.DataLoader(data_train, args.batchsize, sampler=train_sampler,
                                                       pin_memory=True)

    print("===> Building models")
    Deweathernet = AWF().to(device)
    Deweathernet = torch.nn.parallel.DistributedDataParallel(Deweathernet, device_ids=[local_rank],
                                                             find_unused_parameters=True)

    print("===> Prepare the teacher net")
    t_net_dehaze = AWF().to(device)
    t_net_derain = AWF().to(device)
    t_net_desnow = AWF().to(device)
    t_net_dehaze = torch.nn.parallel.DistributedDataParallel(t_net_dehaze, device_ids=[local_rank],
                                                             find_unused_parameters=True)
    t_net_derain = torch.nn.parallel.DistributedDataParallel(t_net_derain, device_ids=[local_rank],
                                                             find_unused_parameters=True)
    t_net_desnow = torch.nn.parallel.DistributedDataParallel(t_net_desnow, device_ids=[local_rank],
                                                             find_unused_parameters=True)

    state_dehaze = torch.load(str(args.t_dehaze_ckpt), map_location='cpu')
    state_derain = torch.load(str(args.t_derain_ckpt), map_location='cpu')
    state_desnow = torch.load(str(args.t_desnow_ckpt), map_location='cpu')

    t_net_dehaze.load_state_dict(state_dehaze['AWF'])
    t_net_derain.load_state_dict(state_derain['AWF'])
    t_net_desnow.load_state_dict(state_desnow['AWF'])
    print('Loading pre-trained teacher_net from %s' % args.t_dehaze_ckpt)
    print('Loading pre-trained teacher_net from %s' % args.t_derain_ckpt)
    print('Loading pre-trained teacher_net from %s' % args.t_desnow_ckpt)

    print("===> Prepare the CKT modules")
    ckt_modules = nn.ModuleList([])
    for c in [48, 96, 192, 256]:
        ckt_modules.append(CKTModule(channel_t=c, channel_s=c, channel_h=c // 2, n_teachers=3))
    ckt_modules = ckt_modules.to(device)

    print("===> Setting Optimizers")
    optimizer_dehaze = torch.optim.Adam(params=Deweathernet.parameters(), lr=args.lr)

    # TODO: optionally copy weights from a checkpoint
    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse), map_location='cpu')
        Deweathernet.load_state_dict(state, False)
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        prev_time = time.time()
        train_step1(args, training_data_loader, optimizer_dehaze, Deweathernet,
                    t_net_dehaze, t_net_derain, t_net_desnow, ckt_modules,
                    epoch, device, prev_time)

        # TODO: save checkpoint
        if local_rank == 0:
            save_checkpoint(Deweathernet, epoch, cache) if epoch % interval == 0 else None


def train_step1(args, tqdm_loader, optimizer_dehaze, Deweathernet, t_net_dehaze, t_net_derain, t_net_desnow,
                ckt_modules, epoch, device, prev_time):
    Deweathernet.train()
    MSELoss = nn.MSELoss().to(device)
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer_dehaze, epoch - 1)
    print("Epoch={}, lr_F={} ".format(epoch, lr_F))

    distillate_loss_total = []
    for i, (data_haze, data_rain, data_snow) in (enumerate(tqdm_loader)):

        t_net_dehaze.eval()
        t_net_derain.eval()
        t_net_desnow.eval()
        # print(i)

        image_haze, image_haze_GT, image_haze_ir, image_haze_ir_GT = (
        data_haze[0].cuda(non_blocking=True), data_haze[1].cuda(non_blocking=True),
        data_haze[2].cuda(non_blocking=True), data_haze[3].cuda(non_blocking=True))
        image_rain, image_rain_GT, image_rain_ir, image_rain_ir_GT = (
        data_rain[0].cuda(non_blocking=True), data_rain[1].cuda(non_blocking=True),
        data_rain[2].cuda(non_blocking=True), data_rain[3].cuda(non_blocking=True))
        image_snow, image_snow_GT, image_snow_ir, image_snow_ir_GT = (
        data_snow[0].cuda(non_blocking=True), data_snow[1].cuda(non_blocking=True),
        data_snow[2].cuda(non_blocking=True), data_snow[3].cuda(non_blocking=True))

        with torch.no_grad():
            _, preds_dehaze, _, _, t_feature_dehaze = t_net_dehaze(image_haze, image_haze_ir)
            _, preds_derain, _, _, t_feature_derain = t_net_derain(image_rain, image_rain_ir)
            _, preds_desnow, _, _, t_feature_desnow = t_net_desnow(image_snow, image_snow_ir)
        _, Final_dehaze, I_Rtx_dehaze, I_Rtx2_dehaze, s_feature_dehaze = Deweathernet(image_haze, image_haze_ir)
        _, Final_derain, I_Rtx_derain, I_Rtx2_derain, s_feature_derain = Deweathernet(image_rain, image_rain_ir)
        _, Final_desnow, I_Rtx_desnow, I_Rtx2_desnow, s_feature_desnow = Deweathernet(image_snow, image_snow_ir)

        l1_loss = torch.nn.L1Loss()
        PFE_loss, PFV_loss = 0., 0.
        T_loss = l1_loss(preds_dehaze, Final_dehaze) + l1_loss(preds_derain, Final_derain) + l1_loss(preds_desnow, Final_desnow)
        # criterion_scr = SCRLoss()
        # SCR_loss = 0.1 * (criterion_scr(image_out_dehaze, image_haze_GT, image_haze) +
        #                   criterion_scr(image_out_derain, image_rain_GT, image_rain) +
        #                   criterion_scr(image_out_desnow, image_snow_GT, image_snow))
        for ii in range(3):
            # print(t_feature_dehaze[i].shape)
            # print('-----------------------------------------------')
            t_proj_features1, t_recons_features1, s_proj_features1 = ckt_modules[ii](t_feature_dehaze[ii],
                                                                                    s_feature_dehaze[ii])
            t_proj_features2, t_recons_features2, s_proj_features2 = ckt_modules[ii](t_feature_derain[ii],
                                                                                    s_feature_derain[ii])
            t_proj_features3, t_recons_features3, s_proj_features3 = ckt_modules[ii](t_feature_desnow[ii],
                                                                                    s_feature_desnow[ii])
            PFE_loss += (l1_loss(s_proj_features1, t_proj_features1) +
                         l1_loss(s_proj_features2, t_proj_features2) +
                         l1_loss(s_proj_features3, t_proj_features3))
            PFV_loss += 0.05 * (l1_loss(t_recons_features1, t_feature_dehaze[ii]) +
                                l1_loss(t_recons_features2, t_feature_derain[ii]) +
                                l1_loss(t_recons_features3, t_feature_desnow[ii]))

        distillate_loss = T_loss + PFE_loss + PFV_loss

        data_IR_expanded = image_haze_ir_GT.expand_as(Final_dehaze)
        Loss_mse = (MSELoss(data_IR_expanded, Final_dehaze) + MSELoss(image_haze_GT, Final_dehaze)) / 2
        Loss_con =  l1_loss(image_haze_GT, I_Rtx_dehaze) +  l1_loss(image_haze_GT, I_Rtx2_dehaze)
        loss_ = fusion_loss_vif(device)
        loss__, loss_gradient, loss_l1, loss_SSIM, Loss_color = loss_(image_haze_GT, image_haze_ir_GT, Final_dehaze)
        fusion_loss_dehaze =  ( loss_SSIM +  loss_gradient  + loss_l1 +
                 Loss_color)

        data_IR_expanded = image_rain_ir_GT.expand_as(Final_derain)
        Loss_mse = ( MSELoss(data_IR_expanded, Final_derain) + MSELoss(image_rain_GT, Final_derain)) / 2
        Loss_con =  l1_loss(image_rain_GT, I_Rtx_derain) +  l1_loss(image_rain_GT, I_Rtx2_derain)
        loss_ = fusion_loss_vif(device)
        loss__, loss_gradient, loss_l1, loss_SSIM, Loss_color = loss_(image_rain_GT, image_rain_ir_GT, Final_derain)
        fusion_loss_derain =  ( loss_SSIM +  loss_gradient  + loss_l1 +
                 Loss_color)

        data_IR_expanded = image_snow_ir_GT.expand_as(Final_desnow)
        Loss_mse = (MSELoss(data_IR_expanded, Final_desnow) + MSELoss(image_snow_GT, Final_desnow)) / 2
        Loss_con = l1_loss(image_snow_GT, I_Rtx_desnow) + l1_loss(image_snow_GT, I_Rtx2_desnow)
        loss_ = fusion_loss_vif(device)
        loss__, loss_gradient, loss_l1, loss_SSIM, Loss_color = loss_(image_snow_GT, image_snow_ir_GT, Final_desnow)
        fusion_loss_desnow = (loss_SSIM + loss_gradient + loss_l1 +
                Loss_color)

        total_loss = (fusion_loss_dehaze + fusion_loss_derain + fusion_loss_desnow) / 3 + distillate_loss

        torch.distributed.barrier()
        optimizer_dehaze.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer_dehaze.step()

        batches_done = epoch * len(tqdm_loader) + i
        batches_left = args.nEpochs * len(tqdm_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [distillate_loss: %f] [total_loss: %f] ETA: %.10s"
            % (
                epoch,
                args.nEpochs,
                i,
                len(tqdm_loader),
                distillate_loss.item(),
                total_loss.item(),
                time_left,
            )
        )


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'deweather_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


first_execution = True


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    args = hyper_args()
    # visdom = visdom.Visdom(port=8097, env='Fusion')

    # main(args, visdom)
    main(args)


