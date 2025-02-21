import cv2
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import Voxelization
from mmdet3d.models import builder
import matplotlib.pyplot as plt
import numpy as np

import math
class PtsBackbone(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 return_context=True,
                 return_occupancy=True,
                 **kwargs,
                 ):
        super(PtsBackbone, self).__init__()

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.return_context = return_context

        self.return_occupancy = return_occupancy
        mid_channels = pts_backbone['out_channels'][-1]
        self.mlp = nn.Sequential(
                nn.Linear(2,10),
                nn.ReLU(),
                nn.Linear(10,10),
                nn.ReLU(),
                nn.Linear(10,2),
            )
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
            mid_channels = sum(pts_neck['out_channels'])
        else:
            self.pts_neck = None

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 80
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))

    def gaussian_3d(self,size, sigma=1.0):
        """返回一个3D高斯核"""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy, zz = np.meshgrid(ax, ax, ax)
        kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
        kernel /= kernel.sum()  # 归一化
        return kernel
    def generate_gaussian_kernel(self, size=3, sigma=1):
        """Generate a 3D Gaussian kernel."""
        size = torch.tensor(size)
        sigma = torch.tensor(sigma)
        kernel = torch.tensor(
            [[(1 / (2 * math.pi * sigma ** 2)) * torch.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2))
              for x in range(size)] for y in range(size)]
        )
        return kernel / kernel.sum()

    def get_para(self,v_data):
        B, Z, X, Y, F = v_data.shape[:]
        v_data = v_data.view(B * X * Y * Z, F)
        para = self.mlp(v_data[:,3:])
        para = para.view(B*Z*X,Y,2)
        self.size = para[:,:,0].unsqueeze(-1)
        self.sigma = para[:,:,1].unsqueeze(-1)
        # print(size.shape)
        #
        # return size,sigma



    def gaussian_expand(self, voxel_data):
        """
        Apply Gaussian expansion to voxel data.

        Args:
            voxel_data (torch.Tensor): The input voxel data of shape [N, D, H, W].

        Returns:
            torch.Tensor: The expanded voxel data.

        """
        # voxel_data = voxel_data.squeeze(0)
        
        self.get_para(voxel_data)

        norm_values = torch.tensor([1, 3, 5], dtype=torch.float)
        norm_values_expand = norm_values.view(1, 1, -1).cuda()
        diff_size = torch.abs(self.size - norm_values_expand)
        min_diff_idx = torch.argmin(diff_size, dim=-1)
        normalized_size = norm_values[min_diff_idx].unsqueeze(-1)

        # normalized_size2= normalized_size.clone()
        # voxel_data1 = voxel_data.clone()
        # voxel_data1 = voxel_data1.squeeze(0).squeeze(0).cpu()
        # data_vis = normalized_size2.numpy().squeeze()
        # x_coords = voxel_data1[:,:,0].numpy().astype(np.int32)
        # y_coords = voxel_data1[:,:,1].numpy().astype(np.int32)
        # image = np.zeros((60,705,3),dtype=np.int8)
        # data_nor = ((data_vis-np.min(data_vis))/(np.max(data_vis)-np.min(data_vis))*255).astype(np.uint8)
        # color_map = cv2.COLORMAP_VIRIDIS
        # for i in range(data_vis.shape[0]):
        #     x = x_coords[i]
        #     y = y_coords[i]
        #     values = data_nor[i]
        #     for xi,yi,value in zip(x,y,values):
        #         color = cv2.applyColorMap(np.array([[value]],dtype=np.uint8), color_map)[0,0]
        #         center = (xi,yi)
        #
        #         cv2.circle(image,center,5,color.tolist(),-1)
        # a =str(data_vis.shape[0])
        # cv2.imwrite(f'/home/sr4/yjt1/CRN-main/maps/image_{a}.png',image)



        voxel_data = voxel_data.squeeze(0)
        expanded_voxel_grid = torch.zeros_like(voxel_data)


        # 对每个特征维度进行卷积
        for feature in range(voxel_data.shape[2]):
        

            feature_map = voxel_data[0, :, feature, :].unsqueeze(0).unsqueeze(0)
            normalized_size1 = normalized_size[:, feature, :].unsqueeze(0).unsqueeze(0)
            sigma1 = torch.mean(self.sigma[:, feature, :].unsqueeze(-1))
            for i in range(3):

                size = 2 * i + 1
                mask = (normalized_size1[:, :, :, 0] == size)
                mask = [mask.unsqueeze(-1) for i in range(5)]
                mask = torch.cat(mask, dim=-1).cuda()
                # print('------------------------------------------------------')
                # print(feature_map.shape)
                # print(mask.shape)
                # print("--------------------------------------------------------------------------------------")

                featurei = feature_map * mask
                # print(featurei.shape)
                # print("--------------------------------------------------------------------------------------")
                if size == 1:
                    feature1 = featurei
                if size == 3:
                    gaussian_kernel = self.generate_gaussian_kernel(size, sigma1).unsqueeze(0).unsqueeze(0).cuda()


                    feature3 = F.conv2d(featurei, gaussian_kernel, padding=1)

                if size == 5:
                    gaussian_kernel = self.generate_gaussian_kernel(size, sigma1).unsqueeze(0).unsqueeze(0).cuda()
                    feature5 = F.conv2d(featurei, gaussian_kernel, padding=2)


            expanded_voxel_grid[0, :, feature, :] = feature1.squeeze() + feature3.squeeze() + feature5.squeeze()

            # print((voxel_data-expanded_voxel_grid)*100)
            # print("------------------------------------------------------------------------------------------------------------------")
            # print(expanded_voxel_grid)


        return expanded_voxel_grid

    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        # print(points.shape)
        voxels, coors, num_points = [], [], []
        batch_size, _, _ = points.shape
        points_list = [points[i] for i in range(batch_size)]

        for res in points_list:
            # print(res.shape)
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            # print(res_num_points.shape)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def _forward_single_sweep(self, pts):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        # print(pts.shape)

        B, N, P, F = pts.shape

        # mask_size = (B*N, P,F)  # 根据实际数据范围调整
        # mask = np.zeros(mask_size)
        # gaussian_kernel = self.gaussian_3d(size=3, sigma=1.0)
        batch_size = B * N
        pts = pts.contiguous().view(B*N, P, F)

        # for i in range (B*N):
        #     for j in range (P):
        #         x,y,z = pts[i,j,:3]
        #         x,y,z = int(x),int(y),int(z)
        #
        #         mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] += gaussian_kernel
        # mask /= mask.max()
        # # print(mask.shape)
        # mask = torch.from_numpy(mask).cuda()
        # mask = mask.float()
        # print(mask)

        #pts  [12,1536,5]
        voxels, num_points, coors = self.voxelize(pts)

        # mean_feature = torch.mean(voxels[:, :, 0], axis=0)
        #
        # # 创建 X 和 Y 坐标
        # x = np.arange(mean_feature.shape[1])
        # y = np.arange(mean_feature.shape[0])
        # X, Y = np.meshgrid(x, y)
        #
        # # 可视化
        # fig, ax = plt.subplots(figsize=(12, 12))
        #
        # # 使用 imshow 显示体素数据
        # cax = ax.imshow(mean_feature, cmap='viridis', interpolation='nearest')
        #
        # # 添加颜色条
        # plt.colorbar(cax, ax=ax, label='Feature Value')
        #
        # # 设置图形属性
        # plt.title(f'Radar Voxel Visualization (Feature {0})')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.gca().invert_yaxis()  # Optional: to match the origin at the top-left corner
        # plt.gca().set_aspect('equal', adjustable='box')
        #
        # plt.show()
        # voxels_m,num,coor = self.voxelize(mask)

        # print(num_points.shape)

        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts_voxelize'].append(t1.elapsed_time(t2))
        ori_voxel = voxels.clone()
        ori_voxel = ori_voxel.unsqueeze(0).unsqueeze(0)

        ori_voxel = self.gaussian_expand(ori_voxel)

        ori_voxel = ori_voxel.squeeze(0).squeeze(0)




        ori_voxel_features = self.pts_voxel_encoder(ori_voxel,num_points,coors)
        ori_voxel_features = torch.sigmoid(ori_voxel_features)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)#[n,64]
        # print(torch.mean(ori_voxel_features - voxel_features))
        voxel_features = (voxel_features+ori_voxel_features)/2


        # print(voxel_features.shape)
        # print("------------------------------------------")
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)#[12,64,140,88]
        
        
        x = self.pts_backbone(x)#0:[12,64,140,88] 1:[12,128,70,44] 2:[12,256,35,22]

        if self.pts_neck is not None:
            x = self.pts_neck(x)#[12,192,70,44]

        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['pts_backbone'].append(t2.elapsed_time(t3))

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid()

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts_head'].append(t3.elapsed_time(t4))
        # print(x_occupancy.shape)
        return x_context, x_occupancy

    def forward(self, ptss, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, _, _ = ptss.shape

        key_context, key_occupancy = self._forward_single_sweep(ptss[:, 0, ...])
        
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_context, key_occupancy, self.times

        context_list = [key_context]
        occupancy_list = [key_occupancy]
        for sweep_index in range(1, num_sweeps):

            with torch.no_grad():
                context, occupancy = self._forward_single_sweep(ptss[:, sweep_index, ...])
                # print(context.shape)
                context_list.append(context)
                occupancy_list.append(occupancy)

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = torch.cat(context_list, 1)
        if self.return_occupancy:
            ret_occupancy = torch.cat(occupancy_list, 1)
        return ret_context, ret_occupancy, self.times
