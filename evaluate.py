import argparse
import torch, numpy as np, torch.utils.data
from tqdm import tqdm
from model import OGSFNet as PointConvSceneFlow
from model import multiScaleLoss
from collections import defaultdict


from main_utils import *
import time

def error(pred, labels, mask):
    pred = pred.permute(0, 2, 1).cpu().numpy()
    labels = labels.permute(0, 2, 1).cpu().numpy()

    mask = mask.cpu().numpy() #if len(mask.shape)>1 else mask.unsqueeze(0).cpu().numpy()

    err = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)  # B,N
    acc050 = np.sum(np.logical_or((err <= 0.05) * mask, (err / gtflow_len <= 0.05) * mask), axis=1)
    acc010 = np.sum(np.logical_or((err <= 0.1) * mask, (err / gtflow_len <= 0.1) * mask), axis=1)
    outlier = np.sum(np.logical_or((err >0.3) * mask, (err / gtflow_len > 0.1) * mask), axis=1)


    mask_sum = np.sum(mask, 1)
    acc050 = acc050[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc050 = np.mean(acc050)
    acc010 = acc010[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc010 = np.mean(acc010)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    epe = np.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)
    return epe, acc050, acc010, outlier


def outlier_roc(pred, labels, mask):
    pred = pred.permute(0, 2, 1).cpu().numpy()
    labels = labels.permute(0, 2, 1).cpu().numpy()

    mask = mask.cpu().numpy()

    err = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

    roc01 = np.sum((err > 0.1) * mask, axis=1)
    roc02 = np.sum((err > 0.2) * mask, axis=1)
    roc03 = np.sum((err > 0.3) * mask, axis=1)
    roc04 = np.sum((err > 0.4) * mask, axis=1)
    roc05 = np.sum((err > 0.5) * mask, axis=1)
    roc06 = np.sum((err > 0.6) * mask, axis=1)
    roc07 = np.sum((err > 0.7) * mask, axis=1)
    roc08 = np.sum((err > 0.8) * mask, axis=1)
    roc09 = np.sum((err > 0.9) * mask, axis=1)
    roc10 = np.sum((err > 1.0) * mask, axis=1)



    mask_sum = np.sum(mask, 1)

    roc01 = roc01[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc01 = np.mean(roc01)
    roc02 = roc02[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc02 = np.mean(roc02)
    roc03 = roc03[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc03 = np.mean(roc03)
    roc04 = roc04[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc04 = np.mean(roc04)
    roc05 = roc05[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc05 = np.mean(roc05)
    roc06 = roc06[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc06 = np.mean(roc06)
    roc07 = roc07[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc07 = np.mean(roc07)
    roc08 = roc08[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc08 = np.mean(roc08)
    roc09 = roc09[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc09 = np.mean(roc09)
    roc10 = roc10[mask_sum > 0] / mask_sum[mask_sum > 0]
    roc10 = np.mean(roc10)

    return roc01,roc02,roc03,roc04,roc05,roc06,roc07,roc08,roc09,roc10


def F1(pred_mask, gt_mask):
    pred_mask=pred_mask.squeeze().cpu().numpy()
    gt_mask = gt_mask.squeeze().cpu().numpy()

    tp = (pred_mask <= 0.5).astype(np.float32) * (gt_mask==0.0).astype(np.float32)
    fp = (pred_mask<=0.5).astype(np.float32) * (gt_mask==1.0).astype(np.float32)
    fn = (pred_mask>0.5).astype(np.float32) * (gt_mask==0.0).astype(np.float32)
    tn = (pred_mask>0.5).astype(np.float32) * (gt_mask==1.0).astype(np.float32)

    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    tn = tn.sum()


    return tp, fp, fn, tn


def eval_sceneflow(model, loader):
    metrics = defaultdict(lambda: list())
    occ_sum=0
    occ_total_loss = 0
    total_seen = 0
    roc_dict=['roc01','roc02','roc03','roc04','roc05','roc06','roc07','roc08','roc09','roc10']
    tp_tot, tn_tot, fp_tot, fn_tot =0.0,0.0,0.0,0.0
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
            epe_occ, acc050, acc010, outlier = error(pred_flows[0], flow.permute(0, 2, 1), mask.squeeze(-1))
            roc = outlier_roc(pred_flows[0], flow.permute(0, 2, 1), mask.squeeze(-1))
            tp, fp, fn, tn = F1(pred_mask[0], mask)
            epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) - flow), dim=2).mean()

        if len(roc_dict)!= len(roc):
            print('error in roc')

        for i in range(len(roc)):
            metrics[roc_dict[i]].append(roc[i])

        occ_sum += occ_acc
        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())
        metrics['epe3d_occ'].append(epe_occ)
        metrics['acc050'].append(acc050)
        metrics['acc010'].append(acc010)
        metrics['outlier'].append(outlier)

        tp_tot+=tp
        fp_tot += fp
        fn_tot += fn
        tn_tot += tn

        total_seen += 5

    final_occ_acc = occ_sum / len(loader)
    occ_total_loss = occ_total_loss/total_seen
    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_epe_occ = np.mean(metrics['epe3d_occ'])

    acc050 = np.mean(metrics['acc050'])
    acc010 = np.mean(metrics['acc010'])
    outlier = np.mean(metrics['outlier'])
    mean_eval = np.mean(metrics['eval_loss'])

    roc_final = []
    for i in range(len(roc_dict)):
        roc_final.append(np.mean(metrics[roc_dict[i]]))

    return mean_epe3d, mean_epe_occ, acc050, acc010, outlier, final_occ_acc, roc_final,[tp_tot, fp_tot, fn_tot, tn_tot]


def main(num_points, dataset, ckp_path):

    # args
    # num_points = 4096
    # dataset = 'f3d'
    # ckp_path = './pretrained_model/PointConv_93.6745_080_0.1919.pth'

    # choose dataset
    if dataset.lower() == 'f3d':
        from dataset_f3d import SceneflowDataset
        val_dataset = SceneflowDataset(npoints=num_points, train=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=5,
                                                     num_workers=8,
                                                     pin_memory=True,
                                                     drop_last=False)
        print('evaluate on Flyingthings3D...')
        print('test:', len(val_dataset), '/', len(val_loader))

    elif dataset.lower() == 'kitti':
        from dataset_kitti import SceneflowDataset
        val_dataset = SceneflowDataset(npoints=num_points, train=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=5,
                                                 num_workers=8,
                                                 pin_memory=True,
                                                 drop_last=False)
        print('evaluate on KITTI...')
        print('test:', len(val_dataset), '/', len(val_loader))

    else:
        raise ValueError('Unknown dataset: '+ dataset)

    # load the pretrained model
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = PointConvSceneFlow().cuda()

    try:
        model.load_state_dict(torch.load(ckp_path))
    except:
        raise ValueError('Incorrect file path')


    epe3d, epe3d_occ, acc05, acc10, outlier, occ_acc, roc_final, occ_f1 = eval_sceneflow(model.eval(), val_loader)
    str_out = '%s mean EPE_full: %f  EPE: %f  acc05: %f acc10: %f outlier: %f ' % ('eval', epe3d, epe3d_occ, acc05, acc10, outlier)
    print(str_out)

    if dataset == 'f3d':
        tp = occ_f1[0]
        fp = occ_f1[1]
        fn = occ_f1[2]
        tn = occ_f1[3]

        precision = tp/(tp+fp)
        recall = tp/(tp + fn)
        f1 = 2.0*precision*recall/(precision+recall)
        print('Occlusion Accuracy: %.4f' % (100*occ_acc))
        print('F1 score: %.4f' % (f1))


if __name__ == '__main__':

    # Args
    parser = argparse.ArgumentParser(description='evaluate OGSFNet.')
    parser.add_argument('--num_points', type=int, default=8192, help='number of points in the input point cloud')
    parser.add_argument('--dataset', type=str, default='f3d', help='choosed the dataset for the evaluation')
    parser.add_argument('--ckp_path', type=str, default='./pretrained_model/OGSFNet_94.8932_090_0.1636.pth', help='file path of the pretrained model to evaluate')
    args = parser.parse_args()
    main(args.num_points, args.dataset, args.ckp_path)