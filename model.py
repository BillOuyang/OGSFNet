import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_occ_util import PointConv, PointConvD, Warping, UpsampleFlow, Mask_CV
from pointconv_occ_util import SceneFlow_Occ_Estimator
from pointconv_occ_util import index_points_gather as index_points, index_points_group, Conv1d, knn_point
import time

scale = 1.0


class OGSFNet(nn.Module):
    def __init__(self):
        super(OGSFNet, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        # l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        self.cost0 = Mask_CV(flow_nei, 32 + 32 + 32 + 32 + 3, [32, 32])
        self.flow0 = SceneFlow_Occ_Estimator(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        # l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)                        ## 2048
        self.cost1 = Mask_CV(flow_nei, 64 + 32 + 64 + 32 + 3, [64, 64])
        self.flow1 = SceneFlow_Occ_Estimator(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        # l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)                       ## 512
        self.cost2 = Mask_CV(flow_nei, 128 + 64 + 128 + 64 + 3, [128, 128])
        self.flow2 = SceneFlow_Occ_Estimator(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        # l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)                       ## 256
        self.cost3 = Mask_CV(flow_nei, 256 + 64 + 256 + 64 + 3, [256, 256])
        self.flow3 = SceneFlow_Occ_Estimator(256, 256, flow_ch=0, occ_ch=0)
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        # l4: 128
        self.level4 = PointConvD(128, feat_nei, 512 + 3, 256)                        ## 128

        # deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        # warping
        self.warping = Warping()

        # upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3

        # l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)  # B 3 N
        color2 = color2.permute(0, 2, 1)  # B 3 N
        feat1_l0 = self.level0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)  ## [B, 64, 8192]

        feat2_l0 = self.level0(color2)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)  ## [B, 64, 8192]

        # l1  S=2048
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)  ##[B,C=3,S] [B,D=64,S]  [B, S]
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)  ##[B,D=128,S]

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)  ##[B,C=3,S] [B,D=64,S]  [B, S]
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)  ##[B,D=128,S]

        # l2 S=512
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)  ##[B,C=3,S] [B,D=128,S]  [B, S]
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)  ##[B,D=256,S]

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)  ##[B,C=3,S] [B,D=128,S]  [B, S]
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)  ##[B,D=256,S]

        # l3  S=256
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)  ##[B,C=3,S] [B,D=256,S]  [B, S]
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)  ##[B,D=512,S]

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        # l4  S=128
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)  ##[B,C=3,S] [B,D=256,S]  [B, S]
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)  ##[B,D=256, N=256]
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)  ##[B,D=64, N=256]

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)  ##[B,D=256, N=256]
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)  ##[B,D=64, N=256]

        # l3  S=256
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim=1)  ##[B,D=256+64,S]
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim=1)
        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)  ##[B,c=256,S]
        feat3, flow3, occ_mask3 = self.flow3(pc1_l3, feat1_l3, cost3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim=1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim=1)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim=1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim=1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim=1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim=1)

        # l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        up_occ_mask2 = self.upsample(pc1_l2, pc1_l3, occ_mask3)
        ## flow3;[B, 3, N]/ occ: [B, 1, N]

        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2, up_occ_mask2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim=1)
        feat2, flow2, occ_mask2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2, up_occ_mask2)

        # l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        up_occ_mask1 = self.upsample(pc1_l1, pc1_l2, occ_mask2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1, up_occ_mask1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim=1)
        feat1, flow1, occ_mask1 = self.flow1(pc1_l1, new_feat1_l1, cost1, up_flow1, up_occ_mask1)

        # l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        up_occ_mask0 = self.upsample(pc1_l0, pc1_l1, occ_mask1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0, up_occ_mask0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim=1)
        _, flow0, occ_mask0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0, up_occ_mask0)

        flows = [flow0, flow1, flow2, flow3]
        occ_masks = [occ_mask0, occ_mask1, occ_mask2, occ_mask3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, occ_masks, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2


def multiScaleLoss(pred_flows, gt_flow, pred_occ_masks, gt_occ_masks, fps_idxs, alpha=[0.02, 0.04, 0.08, 0.16]):
    # num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    # generate GT list and masks
    gt_flows = [gt_flow]
    gt_masks = [gt_occ_masks]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        sub_gt_mask = index_points(gt_masks[-1], fps_idx)
        gt_flows.append(sub_gt_flow)
        gt_masks.append(sub_gt_mask)

    occ_sum=0
    flow_loss = torch.zeros(1).cuda()
    occ_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = (pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset])
        diff_mask = pred_occ_masks[i].permute(0, 2, 1) - gt_masks[i + offset]

        occ_loss += 1.4*alpha[i] *torch.norm(diff_mask, dim=2).sum(dim=1).mean()
        flow_loss += alpha[i] *(torch.norm(diff_flow, dim=2).sum(dim=1).mean() + torch.norm(diff_flow*gt_masks[i + offset], dim=2).sum(dim=1).mean())


    pred_occ_mask = pred_occ_masks[0].permute(0, 2, 1) > 0.5
    occ_acc = torch.mean((pred_occ_mask.type(torch.float32) - gt_masks[0].type(torch.float32)) ** 2)
    occ_acc = 1.0 - occ_acc.item()
    occ_sum += occ_acc

    return flow_loss, occ_loss, occ_sum


