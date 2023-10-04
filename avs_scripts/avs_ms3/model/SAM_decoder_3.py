from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer

import math
import torch
import numpy as np
from torch import Tensor, nn 
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Type, Optional

from einops import rearrange, repeat

class Decoder(nn.Module):

    mask_threshold: float = 0.0
    image_format: str = "RGB"
    prompt_embed_dim: int = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_embedding_size = (image_embedding_size, image_embedding_size)

    def __init__(self, depth=4, use_global_embedding=False, av_fusion=True, load_pretrained_sam=True):
        super(Decoder, self).__init__()
        # self.sam_checkpoint = "../sam_sandbox/sam_vit_h_4b8939.pth"
        # self.model_type = "vit_h"
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)
        
        self.audio_embed_layer = nn.Linear(128, self.prompt_embed_dim)

        # AV fusion Conv 
        self.av_fusion = av_fusion
        if self.av_fusion:
            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(
                    AVFusionBlock()
            )

        self.use_global_embedding = use_global_embedding
        if self.use_global_embedding:
            self.global_audio_embedding = nn.parameter.Parameter(torch.randn(1, self.prompt_embed_dim), requires_grad=True)

        self.load_pretrained_sam = load_pretrained_sam
        if self.load_pretrained_sam:
            self.load_sam_checkpoint()

    def load_sam_checkpoint(self):
        prompt_path = "../sam_sandbox/prompt_encoder.pth"
        mask_decoder_path = "../sam_sandbox/mask_decoder.pth"
        self.prompt_encoder.load_state_dict(torch.load(prompt_path))
        self.mask_decoder.load_state_dict(torch.load(mask_decoder_path))
        print("load pretrained sam checkpoint")
        return
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size[0], input_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    
    def find_bounding_box(self, mask):
        assert len(mask.shape) == 2
        device = mask.device
        if mask.min() < 0:
            # output = torch.sigmoid(output)
            mask = (mask > 0).int()

        mask = mask.detach().cpu().numpy()
        indices = np.argwhere(mask)

        if len(indices) == 0:
            return None

        x_min = np.min(indices[:, 1])
        x_max = np.max(indices[:, 1])
        y_min = np.min(indices[:, 0])
        y_max = np.max(indices[:, 0])

        return torch.tensor([[[x_min, y_min, x_max, y_max]]], device=device)

        
    def reg_loss(self, visual_feature, audio_feature, mask):
        # Audio : BT, 1, C
        # Visual: BT, C, H, W
        B, C, H, W = visual_feature.shape

        vis_aud = audio_feature.reshape(B, 256, 1, 1).expand(-1, -1, 64, 64)
        vis_img = visual_feature

        cos_sim = torch.nn.CosineSimilarity(dim=1)(vis_img, vis_aud)
        cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())
        
        downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(mask)[:,0,:,:] # [bs*5, 1, H, W]
        
        sim_obj = torch.mul(cos_sim, downsample_pred_masks)  # [bs*5, 1, H, W]
        sim_bg = torch.mul(cos_sim, 1 - downsample_pred_masks)  # [bs*5, 1, H, W]

        mean_sim_obj = torch.mean(sim_obj, dim=(1, 2))  # [bs*5, 1]
        mean_sim_bg = torch.mean(sim_bg, dim=(1, 2))  # [bs*5, 1]

        logit = torch.cat((mean_sim_obj.unsqueeze(1), mean_sim_bg.unsqueeze(1)), dim=1).float()  # [bs*5, 2]
        target = torch.zeros_like(mean_sim_obj).to(logit.device).type(torch.int64)
        loss = nn.CrossEntropyLoss()(logit, target)
        # loss = loss(mean_sim_obj, target) + loss(mean_sim_bg, target)
        # loss = torch.nn.MSELoss()(sim_bg, sim_obj)
        # loss = torch.log(1 + 1e-3 - torch.mean(loss))  
        # loss = - torch.nn.KLDivLoss()(mean_sim_obj, mean_sim_bg) - torch.nn.KLDivLoss()(mean_sim_bg, mean_sim_obj) 

        # obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*5, 1]
        # masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6) # [bs*5, C]
        # masked_v_fea = F.normalize(masked_v_fea, dim=-1)
        
        return loss


    def fuse_audio_visual_features(self, audio_feature, visual_feature):
        # audio_feature: B 1 256
        # visual_feature: B 256 H W
        # output: B 256 H W
        b, c, h, w = visual_feature.shape

        image_pe = self.get_dense_pe()
        image_pe = rearrange(image_pe, 'b c h w -> b (h w) c')
        visual_feature = rearrange(visual_feature, 'b c h w -> b (h w) c')
        # audio_feature = repeat(audio_feature, 'b n c -> b (repeat n) c', repeat=int(h*w))

        fused_visual_feature = visual_feature
        fused_audio_feature = audio_feature
        
        for layer in self.layers:
            fused_visual_feature, fused_audio_feature = layer(fused_audio_feature, fused_visual_feature, audio_feature, image_pe)

        fused_audio_feature = fused_audio_feature + audio_feature
        # fused_audio_feature = torch.nn.AdaptiveAvgPool1d(1)(fused_audio_feature.transpose(1, 2)).transpose(1, 2)

        fused_visual_feature = rearrange(fused_visual_feature, 'b (h w) c -> b c h w', b=b, h=h, w=w, c=c)

        return fused_visual_feature, fused_audio_feature 
    
    
    def forward_audio(
        self,
        image_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        masks
    ):
        bs = image_embeddings.shape[0]
        
        if audio_embeddings.shape[-1] == 128:  # Vggish
            audio_embeddings = self.audio_embed_layer(audio_embeddings) # B 2 256
            audio_embeddings = nn.functional.normalize(audio_embeddings, dim=-1)
        
        if self.av_fusion:
            dense_embeddings, audio_embeddings = self.fuse_audio_visual_features(audio_embeddings, image_embeddings)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    5, -1, self.image_embedding_size[0], self.image_embedding_size[1]
                ).to(image_embeddings.device)
            
        if masks is not None:
            loss = self.reg_loss(dense_embeddings + image_embeddings, audio_embeddings, mask=masks)
        else:
            loss = torch.tensor(0).to(image_embeddings.device)

        if self.use_global_embedding:
            audio_embeddings = torch.cat([audio_embeddings, self.global_audio_embedding.expand(audio_embeddings.shape[0], 1, -1)], dim=1)
        else:
            audio_embeddings = audio_embeddings.repeat(1, 2, 1)
        

        # SAM doesn't support batched inputs, so we have to loop over the batch
        outputs = []
        for i in range(bs):
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=audio_embeddings[i].unsqueeze(0), 
                dense_prompt_embeddings=dense_embeddings[i],
                multimask_output=True,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(self.image_size, self.image_size),
                original_size=(224, 224),
            ).squeeze(0)

            masks = torch.sum(masks, dim=0)[None, :, :]

            outputs.append(masks)
        outputs = torch.stack(outputs, dim=0)
        return outputs, loss
    
    def forward_box(
        self,
        image_embeddings: torch.Tensor,
        input_masks: torch.Tensor
    ):
        bs = image_embeddings.shape[0]
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            ).to(image_embeddings.device)

        # SAM doesn't support batched inputs, so we have to loop over the batch
        outputs = []
        for i in range(bs):
            box = self.find_bounding_box(input_masks[i].squeeze(0))
            
            sparse_embeddings, _ = self.prompt_encoder(
                  points=None,
                  boxes=box,
                  masks=None
                )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, 
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(self.image_size, self.image_size),
                original_size=(224, 224),
            ).squeeze(0)
            
            masks = torch.sum(masks, dim=0)[None, :, :]

            outputs.append(masks)
        outputs = torch.stack(outputs, dim=0)
        return outputs

    def forward(
        self,
        image_embeddings: torch.Tensor,
        masks=None,
        audio_feature=None
    ):
        if audio_feature is not None:
          output, loss = self.forward_audio(image_embeddings, audio_feature, masks=masks)
          return output, loss

        raise ValueError("Either audio_feature or box must be provided")
    

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

   
class AVFusionBlock(nn.Module):
    def __init__(self, prompt_embed_dim=256, num_heads=8):
        super(AVFusionBlock, self).__init__()

        self.prompt_embed_dim = prompt_embed_dim
        self.num_heads = num_heads
        # self.self_attn = nn.MultiheadAttention(embed_dim=self.prompt_embed_dim, num_heads=self.num_heads, dropout=0.1),

        self.embed_vis = MLPBlock(self.prompt_embed_dim, self.prompt_embed_dim)
        self.embed_audio = MLPBlock(self.prompt_embed_dim, self.prompt_embed_dim)
        self.embed_audio2 = MLPBlock(self.prompt_embed_dim, self.prompt_embed_dim)
        
        self.embed_av = MLPBlock(self.prompt_embed_dim, self.prompt_embed_dim)

        self.avt_attention = nn.MultiheadAttention(embed_dim=self.prompt_embed_dim, num_heads=self.num_heads, dropout=0.1)
        self.avs_attention = nn.MultiheadAttention(embed_dim=self.prompt_embed_dim, num_heads=self.num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm2 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm3_1 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm3_2 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm4 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm5_1 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm5_2 = nn.LayerNorm(self.prompt_embed_dim) 
        self.norm6 = nn.LayerNorm(self.prompt_embed_dim) 
        
    
    def forward(self, audio_feature, visual_feature, audio_pe, visual_pe):
        # audio_feature: B 256 1 1
        # visual_feature: B 256 H W
        # output: B 256 H W

        # Embed audio and visual features
        b, n_hw, c = visual_feature.shape

        audio_feature = audio_feature + self.embed_audio(audio_feature)
        audio_feature = self.norm1(audio_feature)

        visual_feature = visual_feature + self.embed_vis(visual_feature)
        visual_feature = self.norm2(visual_feature)

        # Attention with PE features

        # Temporal attn
        audio_feature_pe = audio_feature + audio_pe
        visual_feature_pe = visual_feature + visual_pe
        avt_audio_attn = self.avt_attention(visual_feature_pe, audio_feature_pe.repeat(1, n_hw, 1), audio_feature.repeat(1, n_hw, 1))[0] # B HW C
        avt_audio_attn = torch.nn.AdaptiveAvgPool1d(1)(avt_audio_attn.transpose(1, 2)).transpose(1, 2)
        fused_audio_feature = audio_feature + avt_audio_attn # B, 1, C
        fused_audio_feature = self.norm3_1(fused_audio_feature)

        # Spatial attn
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        visual_feature_pe = visual_feature + visual_pe
        avs_audio_attn = self.avs_attention(visual_feature_pe, fused_audio_feature_pe, fused_audio_feature)[0] # B HW C
        avs_audio_attn = torch.nn.AdaptiveAvgPool1d(1)(avs_audio_attn.transpose(1, 2)).transpose(1, 2)
        fused_audio_feature = fused_audio_feature + avs_audio_attn # B 1 C
        fused_audio_feature = self.norm3_2(fused_audio_feature)
        
        # MLP block
        fused_audio_feature = fused_audio_feature + self.embed_audio2(fused_audio_feature)
        fused_audio_feature = self.norm4(fused_audio_feature)
        
        # Attention with PE features

        # Temporal attn
        visual_feature_pe = visual_feature + visual_pe
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        avt_visual_attn = self.avt_attention(fused_audio_feature_pe.repeat(1, n_hw, 1), visual_feature_pe, visual_feature)[0] # B HW C
        fused_visual_feature = visual_feature + avt_visual_attn
        fused_visual_feature = self.norm5_1(fused_visual_feature)

        # Spatial
        fused_visual_feature_pe = fused_visual_feature + visual_pe
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        avs_visual_attn = self.avs_attention(fused_audio_feature_pe, fused_visual_feature_pe, fused_visual_feature)[0] # B 1 C
        fused_visual_feature = fused_visual_feature + avs_visual_attn.repeat(1, n_hw, 1)
        fused_visual_feature = self.norm5_2(fused_visual_feature)

        # MLP block
        fused_visual_feature = fused_visual_feature + self.embed_av(fused_visual_feature)
        fused_visual_feature = self.norm6(fused_visual_feature)

        return fused_visual_feature, fused_audio_feature
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Refiner(nn.Module):

    mask_threshold: float = 0.0
    image_format: str = "RGB"
    prompt_embed_dim: int = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_embedding_size = (image_embedding_size, image_embedding_size)

    def __init__(self, depth=4, refiner_decoder_path=None):
        super(Refiner, self).__init__()

        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.decoder = Decoder(depth=depth)

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

        self.load_sam_checkpoint(refiner_decoder_path)
        for params in self.prompt_encoder.parameters():
            params.requires_grad = False

    def load_sam_checkpoint(self, refiner_decoder_path):
        prompt_path = "../sam_sandbox/prompt_encoder.pth"
        mask_decoder_path = "../sam_sandbox/mask_decoder.pth"
        decoder_path = refiner_decoder_path

        assert decoder_path is not None

        self.prompt_encoder.load_state_dict(torch.load(prompt_path))
        self.mask_decoder.load_state_dict(torch.load(mask_decoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

        print("load pretrained sam checkpoint")
        return
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size[0], input_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    
    def find_bounding_box(self, mask):
        assert len(mask.shape) == 2
        device = mask.device
        if mask.min() < 0:
            # output = torch.sigmoid(output)
            mask = (mask > 0).int()

        mask = mask.detach().cpu().numpy()
        indices = np.argwhere(mask)

        if len(indices) == 0:
            return None

        x_min = np.min(indices[:, 1])
        x_max = np.max(indices[:, 1])
        y_min = np.min(indices[:, 0])
        y_max = np.max(indices[:, 0])

        return torch.tensor([[[x_min, y_min, x_max, y_max]]], device=device)
    

    def forward_box(
        self,
        image_embeddings: torch.Tensor,
        input_masks: torch.Tensor
    ):
        bs = image_embeddings.shape[0]
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            ).to(image_embeddings.device)

        # SAM doesn't support batched inputs, so we have to loop over the batch
        outputs = []
        p12d = (16, 16, 16, 16)
        padded_mask = F.pad(input_masks, p12d, mode='constant', value=-10)
        for i in range(bs):
            box = self.find_bounding_box(input_masks[i].squeeze(0))
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                  points=None,
                  boxes=box,
                  masks=padded_mask[i]
                )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, 
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(self.image_size, self.image_size),
                original_size=(224, 224),
            ).squeeze(0)
            
            masks = torch.sum(masks, dim=0)[None, :, :]

            outputs.append(masks)
        outputs = torch.stack(outputs, dim=0)
        return outputs

    def forward(
        self,
        image_embeddings: torch.Tensor,
        input_masks=None,
        audio_feature=None
    ):
        if audio_feature is not None:
            mask = self.decoder(image_embeddings, audio_feature=audio_feature, masks=input_masks)
            output = self.forward_box(image_embeddings, input_masks=mask)
        #   loss = torch.tensor(0).to(output.device)
            return output

        if input_masks is not None:
          output = self.forward_box(image_embeddings, input_masks)
          return output

        raise ValueError("Either audio_feature or box must be provided")
    