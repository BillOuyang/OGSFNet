import argparse
from dataset_f3d import SceneflowDataset

import torch, numpy as np, glob, math, torch.utils.data
import datetime

from tqdm import tqdm
from model import OGSFNet as PointConvSceneFlow
from model import multiScaleLoss

from pathlib import Path
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
from main_utils import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]


def epe_non_occ(pred, labels, mask):
    '''
        return the non occluded EPE
    '''

    pred = pred.permute(0, 2, 1).cpu().numpy()
    labels = labels.permute(0, 2, 1).cpu().numpy()
    mask = mask.cpu().numpy()
    err = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)
    mask_sum = np.sum(mask, 1)
    epe = np.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)

    return epe


def eval_one_epoch(model, loader, batchsize):
    metrics = defaultdict(lambda: list())
    occ_sum = 0
    occ_total_loss = 0
    total_seen = 0
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, mask = data

        # move to cuda
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            pred_flows, pred_mask, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
            eval_loss, occ_loss, occ_acc = multiScaleLoss(pred_flows, flow,pred_mask, mask, fps_pc1_idxs)
            epe_full = torch.norm((pred_flows[0].permute(0, 2, 1) - flow), dim=2).mean()
            epe = epe_non_occ(pred_flows[0], flow.permute(0, 2, 1), mask.squeeze())

        occ_sum += occ_acc
        occ_total_loss += occ_loss
        metrics['epe_full'].append(epe_full.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())
        metrics['epe'].append(epe)
        total_seen += batchsize

    final_occ_acc = occ_sum / len(loader)
    occ_total_loss = occ_total_loss/total_seen
    mean_epe_full = np.mean(metrics['epe_full'])
    mean_epe = np.mean(metrics['epe'])
    mean_eval_loss = np.mean(metrics['eval_loss'])

    return mean_epe_full, mean_epe, mean_eval_loss, occ_total_loss, final_occ_acc


def train_one_epoch(model, train_loader,optimizer,occ_factor, batch_size):
    total_flow_loss = 0
    total_occ_loss = 0
    total_seen = 0
    occ_sum = 0
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, mask = data

        # move to cuda
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
        mask = mask.cuda()
        model = model.train()
        pred_flows, pred_mask, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

        flow_loss, occ_loss, occ_acc = multiScaleLoss(pred_flows, flow, pred_mask, mask, fps_pc1_idxs)

        loss = flow_loss + occ_factor * occ_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_flow_loss += flow_loss.cpu().data * batch_size
        total_occ_loss += occ_loss.cpu().data * batch_size

        total_seen += batch_size
        occ_sum += occ_acc

    final_occ_acc = occ_sum / len(train_loader)
    train_flow_loss = total_flow_loss / total_seen
    total_occ_loss = total_occ_loss / total_seen

    return final_occ_acc, train_flow_loss, total_occ_loss


def main(num_points, batch_size, epochs, use_multi_gpu, pretrain):
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    # general args
    # num_points = 4096
    # batch_size = 16
    # epochs = 150
    # use_multi_gpu = True
    # pretrain = None

    optimizer = 'Adam'
    gpu = "0"
    multi_gpu = '0,1' if use_multi_gpu is True else None
    model_name = 'OGSFNet'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu if multi_gpu is None else '0,1'
    learning_rate = 0.001
    weight_decay = 0.0001

    # create check point file path
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-' % model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # file backup
    os.system('cp %s %s' % ('model.py', log_dir))
    os.system('cp %s %s' % ('pointconv_occ_util.py', log_dir))
    os.system('cp %s %s' % ('train.py', log_dir))

    # create model
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()
    print('# parameters: ', parameter_count(model))

    # F3D dataloader
    train_dataset = SceneflowDataset(npoints=num_points, train=True, cache=None)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=20,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)
    print('train:', len(train_dataset), '/', len(train_loader))

    val_dataset = SceneflowDataset(npoints=num_points, train=False, cache=None)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=20,
                                             pin_memory=True,
                                             drop_last=True)
    print('test:', len(val_dataset), '/', len(val_loader))

    # use pretrained model
    if pretrain is not None:
        model.load_state_dict(torch.load(pretrain))
        print('load model %s' % pretrain)
    else:
        print('Training from scratch...')

    # GPU selection and multi-GPU
    if multi_gpu is not None:
        device_ids = [int(x) for x in multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print('use multi-GPU')
    else:
        model.cuda()
        print('use single GPU')

    # pretrain = pretrain
    init_epoch = int(pretrain[-14:-11]) if pretrain is not None else 0

    # Initialize optimizer
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay)


    optimizer.param_groups[0]['initial_lr'] = learning_rate
    MIN_LR = 0.00001
    STEP_SIZE_LR = 10
    GAMMA_LR = 0.85
    scheduler = ClippedStepLR(optimizer, STEP_SIZE_LR, MIN_LR, GAMMA_LR, last_epoch=init_epoch-1)
    LEARNING_RATE_CLIP = 1e-5



    best_epe = 1000.0
    for epoch in range(init_epoch, epochs):

        ## lr update
        if epoch ==75:
            scheduler = ClippedStepLR(optimizer, STEP_SIZE_LR, MIN_LR, 0.8, last_epoch=epoch - 1)
            print('update gama...')

        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()

        ## occlusion weight update
        occ_factor = min(0.4, 0.3+epoch*0.001)
        if epoch >=50:
            occ_factor = 0.6
        if epoch >=75:
            occ_factor = 0.1

        ## training for one epoch
        final_occ_acc, train_flow_loss, total_occ_loss = train_one_epoch(model, train_loader, optimizer, occ_factor, batch_size)
        scheduler.step()


        str_out = 'EPOCH %d %s mean loss: %f  occ loss: %f OCC_acc: %f' % (epoch, blue('train'), train_flow_loss, total_occ_loss, 100*final_occ_acc)
        print('occ_weight: %f'%(occ_factor))
        print(str_out)


        ## evaluation for one epoch
        epe_full, epe, eval_loss, occ_loss, final_occ_acc = eval_one_epoch(model.eval(), val_loader, batch_size)

        str_out = 'EPOCH %d %s epe_full: %f  epe: %f  mean eval loss: %f occ loss: %f occ_acc: %f' % (epoch, blue('eval'), epe_full,epe, eval_loss, occ_loss, 100*final_occ_acc)
        print(str_out)

        ## save model
        if epe_full < best_epe:
            best_epe = epe_full
            if multi_gpu is not None:
                torch.save(model.module.state_dict(),
                           '%s/%s_%.4f_%.3d_%.4f.pth' % (checkpoints_dir, model_name,100*final_occ_acc, epoch, epe_full))
            else:
                torch.save(model.state_dict(), '%s/%s_%.4f_%.3d_%.4f.pth' % (checkpoints_dir, model_name, 100*final_occ_acc, epoch, epe_full))
            print('Save model ...')
        print('Best EPE_full is: %.5f' % (best_epe))



if __name__ == '__main__':

    # Args
    parser = argparse.ArgumentParser(description='train OGSFNet.')
    parser.add_argument('--num_points', type=int, default=4096, help='number of point in the input point cloud')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the training')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=True, help='whether to use mult-gpu for the training')
    parser.add_argument('--pretrain', type=str, default=None, help='train from pretrained model')
    args = parser.parse_args()
    main(args.num_points, args.batch_size, args.epochs, args.use_multi_gpu, args.pretrain)




