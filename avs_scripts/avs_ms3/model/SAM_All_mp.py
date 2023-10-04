from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
from model.SAM_encoder import Pred_endecoder, ImageEncoderViT
import math
import torch
import numpy as np
from torch import Tensor, nn 
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Type, Optional
from functools import partial

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.image_encoder = Pred_endecoder()
        self.decoder = Decoder()
    
    def forward(self, 
        image,
        image_embeddings=None,
        masks=None,
        audio_feature=None):
        
        if image_embeddings is None:
            image_embeddings = self.image_encoder(image)
        
        output = self.decoder(image_embeddings, masks=masks, audio_feature=audio_feature)
        return output

class Decoder(nn.Module):

    mask_threshold: float = 0.0
    image_format: str = "RGB"
    prompt_embed_dim: int = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_embedding_size = (image_embedding_size, image_embedding_size)

    encoder_embed_dim=1280
    encoder_depth=32
    encoder_num_heads=16
    encoder_global_attn_indexes=[7, 15, 23, 31]


    def __init__(self, use_global_embedding=False, av_fusion=True, load_pretrained_sam=True):
        super(Decoder, self).__init__()
        # self.sam_checkpoint = "../sam_sandbox/sam_vit_h_4b8939.pth"
        # self.model_type = "vit_h"
        self.image_encoder = Pred_endecoder()
        # self.image_encoder = ImageEncoderViT(
        #     depth=self.encoder_depth,
        #     embed_dim=self.encoder_embed_dim,
        #     img_size=self.image_size,
        #     mlp_ratio=4,
        #     norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        #     num_heads=self.encoder_num_heads,
        #     patch_size=self.vit_patch_size,
        #     qkv_bias=True,
        #     use_rel_pos=True,
        #     global_attn_indexes=self.encoder_global_attn_indexes,
        #     window_size=14,
        #     out_chans=self.prompt_embed_dim,
        # )

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
            depth = 1
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

    def fuse_audio_visual_features(self, audio_feature, visual_feature):
        # audio_feature: B 256 1 1
        # visual_feature: B 256 H W
        # output: B 256 H W
        audio_feature = audio_feature.permute(0, 2, 1).unsqueeze(-1)
        audio_feature = audio_feature.expand(-1, -1, visual_feature.shape[-2], visual_feature.shape[-1])
        fused_visual_feature = visual_feature
        fused_audio_feature = audio_feature

        for layer in self.layers:
            fused_visual_feature, fused_audio_feature = layer(fused_audio_feature, fused_visual_feature)

        fused_audio_feature = torch.nn.AdaptiveAvgPool2d((1, 1))(fused_audio_feature).reshape(fused_audio_feature.shape[0], 1, fused_audio_feature.shape[1])

        return fused_visual_feature, fused_audio_feature 
    
    
    def forward_audio(
        self,
        image_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor
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
        return outputs
    
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
          output = self.forward_audio(image_embeddings, audio_feature)
          return output

        if masks is not None:
          output = self.forward_box(image_embeddings, masks)
          return output

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

        self.embed_vis = nn.Sequential(
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.embed_audio = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.embed_audio2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),

        )
        
        self.embed_av = nn.Sequential(
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(in_channels=self.prompt_embed_dim, out_channels=self.prompt_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.av_attention = nn.MultiheadAttention(embed_dim=self.prompt_embed_dim, num_heads=self.num_heads, dropout=0.1)
        self.norm1 = LayerNorm2d(self.prompt_embed_dim) 
        self.norm2 = LayerNorm2d(self.prompt_embed_dim)
        self.norm3 = LayerNorm2d(self.prompt_embed_dim)
        self.norm4 = LayerNorm2d(self.prompt_embed_dim)
        self.norm5 = LayerNorm2d(self.prompt_embed_dim)
        self.norm6 = LayerNorm2d(self.prompt_embed_dim)

    
    def forward(self, audio_feature, visual_feature):
        # audio_feature: B 256 1 1
        # visual_feature: B 256 H W
        # output: B 256 H W
        b, c, h, w = visual_feature.shape
        audio_feature = audio_feature + self.embed_audio(audio_feature)
        audio_feature = self.norm1(audio_feature)

        visual_feature = visual_feature + self.embed_vis(visual_feature)
        visual_feature = self.norm2(visual_feature)

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        audio_feature = audio_feature.flatten(2).permute(0, 2, 1)
        visual_feature = visual_feature.flatten(2).permute(0, 2, 1)

        fused_audio_feature = audio_feature + self.av_attention(visual_feature, audio_feature, audio_feature)[0]
        fused_audio_feature = fused_audio_feature.transpose(1, 2).view(b, c, h, w)
        fused_audio_feature = self.norm3(fused_audio_feature)
        
        fused_audio_feature = fused_audio_feature + self.embed_audio2(fused_audio_feature)
        fused_audio_feature = self.norm4(fused_audio_feature)
        fused_audio_feature = fused_audio_feature.flatten(2).permute(0, 2, 1)

        fused_visual_feature = visual_feature + self.av_attention(fused_audio_feature, visual_feature, visual_feature)[0]
        fused_visual_feature = fused_visual_feature.transpose(1, 2).view(b, c, h, w)
        fused_audio_feature = fused_audio_feature.transpose(1, 2).view(b, c, h, w)
        
        fused_visual_feature = self.norm5(fused_visual_feature)
        fused_visual_feature = fused_visual_feature + self.embed_av(fused_visual_feature)
        fused_visual_feature = self.norm6(fused_visual_feature)

        return fused_visual_feature, fused_audio_feature
    