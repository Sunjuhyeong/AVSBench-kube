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

        pca_methods = "torch"
        pca_dim = 3
        pca_features_batch = torch.zeros((img_feat.shape[0], 16, 16, pca_dim)).cpu()
        if pca_methods == "sklearn":
            for i in range(img_feat.shape[0]):
                pca = PCA(n_components=pca_dim)
                pca_features = pca.fit_transform(img_feat[i].detach().cpu().numpy())
                pca_features = pca_features.reshape(16, 16, pca_dim).astype(np.uint8)
                pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
                pca_features_batch[i] = torch.from_numpy(pca_features)
            pca_features = pca_features_batch.cuda()

        elif pca_methods == "torch":
            _, _, V = torch.pca_lowrank(img_feat, q=pca_dim)
            pca_features = torch.matmul(img_feat, V)
            pca_features = pca_features.reshape(img_feat.shape[0], 16, 16, pca_dim)

        else:
            raise NotImplementedError

        results = []
        do_normalize = False
        for i in range(pca_dim):
            heatmap_obj = F.interpolate(pca_features[:, :, :, i].unsqueeze(1).float(), size=(224, 224), mode='bilinear', align_corners=True).data
            if do_normalize:
                heatmap_obj = F.normalize(heatmap_obj, p=2, dim=1, eps=1e-12, out=heatmap_obj)
            results.append(heatmap_obj)

        return results

    def forward(self, x, y=None):
        x = self.segment(x)
        return x