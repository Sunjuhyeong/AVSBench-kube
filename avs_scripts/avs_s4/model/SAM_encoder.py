from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator 
from segment_anything.utils.transforms import ResizeLongestSide
from torch import nn 
import torch.nn.functional as F 
import torch

class Pred_endecoder(nn.Module):
    def __init__(self):
        super(Pred_endecoder, self).__init__()
        self.sam_checkpoint = "../sam_sandbox/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        # save_path = "../sam_sandbox/"
        # torch.save(self.sam.prompt_encoder.state_dict(), save_path + "prompt_encoder.pth")
        # torch.save(self.sam.mask_decoder.state_dict(), save_path + "mask_decoder.pth")
        # print("save")
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(-1, 1, 1)
        
        # Normalize colors
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device) 
        return image.permute(2, 0, 1).contiguous()

    def forward(self, x):
        """
        Args:
            x (tensor): a batch of images of shape (B, 3, H, W)
        
        Returns:
            output (tensor): a batch of images of shape (B, D)
        """
        bs = x.shape[0]
        x_transformed = []
        device = x.device
        x = x.cpu().detach().numpy()
        for i in range(bs):
            one_frame = self.prepare_image(image=x[i], transform=self.transform, device=device)
            one_frame = self.preprocess(one_frame)
            x_transformed.append(one_frame)
        x_transformed = torch.stack(x_transformed, dim=0)
        image_embeddings = self.sam.image_encoder(x_transformed)
        return image_embeddings