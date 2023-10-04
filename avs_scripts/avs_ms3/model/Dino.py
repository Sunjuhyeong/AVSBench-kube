import torch
from torch import nn 
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

class Pred_endecoder(nn.Module):
    def __init__(self, channel=256, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.dino =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def segment(self, x):
        
        img_feat = self.dino(x, is_training=True)['x_norm_patchtokens'].squeeze()

        # Sklearn PCA
        # pca = PCA(n_components=3)
        # pca_features = pca.fit_transform(img_feat.detach().cpu().numpy())
        # pca_features = pca_features.reshape(16, 16, 3).astype(np.uint8)
        # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        # pca_features *= 255

        # Torch PCA
        _, _, V = torch.pca_lowrank(img_feat, q=3)
        pca_features = torch.matmul(img_feat, V)
        pca_features = pca_features.reshape(img_feat.shape[0], 16, 16, 3)
        # print(pca_features[:, :, :, 0].unsqueeze(1).shape)
        heatmap_obj0 = F.interpolate(pca_features[:, :, :, 0].unsqueeze(1).float(), size=(224, 224), mode='bilinear',
                                     align_corners=True).data
        heatmap_obj1 = F.interpolate(pca_features[:, :, :, 1].unsqueeze(1).float(), size=(224, 224), mode='bilinear',
                                     align_corners=True).data
        heatmap_obj2 = F.interpolate(pca_features[:, :, :, 2].unsqueeze(1).float(), size=(224, 224), mode='bilinear',
                                     align_corners=True).data
        
        do_normalize = True
        if do_normalize:
            heatmap_obj0 = F.normalize(heatmap_obj0, p=2, dim=0, eps=1e-12, out=heatmap_obj0)
            heatmap_obj1 = F.normalize(heatmap_obj1, p=2, dim=0, eps=1e-12, out=heatmap_obj1)
            heatmap_obj2 = F.normalize(heatmap_obj2, p=2, dim=0, eps=1e-12, out=heatmap_obj2)

        return heatmap_obj0, heatmap_obj1, heatmap_obj2

    def forward(self, x, y=None):
        x = self.segment(x)
        return x