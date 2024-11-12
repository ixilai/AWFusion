from model.AWFusion_module import AWF
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils22.loss_vif import fusion_loss_vif
from dataloader.fuse_data_vsm import GetDataset_type2, GetDataset_type3
import kornia
from tqdm import tqdm
import matplotlib.pyplot as plt
first_execution = True

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # dataset
    parser.add_argument('--ir_path', default='../datasets/Fusion_Train_all/ALL_Rain_train/small2/ir', type=str)
    parser.add_argument('--vi_path', default='../datasets/Fusion_Train_all/ALL_Rain_train/small2/vi', type=str)
    parser.add_argument('--gt_path', default='../datasets/Fusion_Train_all/ALL_Rain_train/small2/gt_vi', type=str)
    parser.add_argument('--gt_ir_path',default='../datasets/Fusion_Train_all/ALL_Rain_train/small2/gt_ir',type=str)

    # implement details
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=101, type=int, help='total epoch')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--optim_gamma', default=0.8, help='resume checkpoint')
    parser.add_argument('--optim_step', default=1, help='resume checkpoint')
    parser.add_argument('--weight_decay', default=0, help='Adam weight_decay')
    parser.add_argument('--clip_grad_norm_value', default=0.0001, help='resume checkpoint')
    parser.add_argument('--interval', default=1, help='record interval')
    # checkpoint
    parser.add_argument("--ckpt_path", default=None, help="path to pretrained model (default: none)")
    args = parser.parse_args()
    return args
args = hyper_args()
local_rank = int(os.environ['LOCAL_RANK'])
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

# Model
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
AWF = AWF().to(device)
AWF = torch.nn.parallel.DistributedDataParallel(AWF, device_ids=[local_rank], find_unused_parameters=True)

# 读取权重文件
if args.ckpt_path is not None:
    state = torch.load(str(args.ckpt_path), map_location='cpu')
    AWF.load_state_dict(state['AWF'], False)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    AWF.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.optim_step, gamma=args.optim_gamma)
MSELoss = nn.MSELoss().to(device)
l1_loss = torch.nn.L1Loss().to(device)
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean').to(device)


# data loader
data = GetDataset_type3('train', ir_path=args.ir_path, vi_path=args.vi_path, gt_path=args.gt_path, method_path=args.gt_path, gt_ir_path=args.gt_ir_path)
train_sampler = torch.utils.data.distributed.DistributedSampler(data)
training_data_loader = torch.utils.data.DataLoader(data, sampler=train_sampler,
                                                       batch_size=args.batch_size,
                                                       pin_memory=True,
                                                       num_workers=16)

tqdm_loader = tqdm(training_data_loader, disable=True)
loader = {'train': training_data_loader, }

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'------------------------------------------------------------------------------'
def init_plot():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, args.nEpochs)  # 设置x轴范围
    ax.set_ylim(0, 1)    # 设置y轴范围
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    return fig, ax, line

fig, ax, line = init_plot()

epochs = []
losses = []

def update_plot(epoch, loss):
    epochs.append(epoch)
    losses.append(loss)
    line.set_data(epochs, losses)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

# step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(args.nEpochs):
    ''' train '''
    for i, (data_IR, data_VIS, data_GT, _, data_gt_ir) in (enumerate(tqdm_loader)):
        data_VIS_rgb, data_IR, data_GT, data_gt_ir = (data_VIS.cuda(non_blocking=True), data_IR.cuda(non_blocking=True),
                                                    data_GT.cuda(non_blocking=True), data_gt_ir.cuda(non_blocking=True))

        AWF.train()
        AWF.zero_grad()
        rgb_Fuse, Final, I_Rtx, I_Rtx2, feature = AWF(data_VIS_rgb, data_IR)
        data_IR_expanded = data_gt_ir.expand_as(Final)

        Loss_mse = ( MSELoss(data_IR_expanded, Final) + MSELoss(data_GT, Final)) / 2
        Loss_con =  l1_loss(data_GT, I_Rtx) +  l1_loss(data_GT, I_Rtx2)
        loss_ = fusion_loss_vif(device)
        loss__, loss_gradient, loss_l1, loss_SSIM, Loss_color = loss_(data_GT, data_gt_ir, Final)
        loss =  loss_SSIM + loss_l1 + loss_gradient + Loss_color + Loss_con

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        nn.utils.clip_grad_norm_(
            AWF.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)

        update_plot(epoch, loss.item())

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = args.nEpochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                args.nEpochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    if epoch % 1 == 0:
        checkpoint = {
            'AWF': AWF.state_dict(),
        }
        if not os.path.exists('./ckpt_LRR'):
            os.makedirs('./ckpt_LRR')
        torch.save(checkpoint, os.path.join(f"./ckpt_LRR/{epoch}" + '.pth'))
    # adjust the learning rate
    scheduler1.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6

plt.ioff()
plt.show()



