from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator 
from segment_anything.utils.transforms import ResizeLongestSide

import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import sys

class Pred_endecoder(nn.Module):
    def __init__(self, channel=256, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.visualize = False
            # Prepare the SAM
        self.sam_checkpoint = "../sam_sandbox/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda"
        self.strong_mask = False
        self.multi_mask = False
        self.normalize_mask = False
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        return 

    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device) 
        return image.permute(2, 0, 1).contiguous()

    def find_bounding_box(self, mask):
        assert len(mask.shape) == 2
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

        return [x_min, y_min, x_max, y_max]

    def center_points_from_box(self, box):
        if box is None:
            return [[112, 112]]
        x_min, y_min, x_max, y_max = box
        x_mean = (x_min + x_max) / 2
        y_mean = (y_min + y_max) / 2

        return [[x_mean, y_mean]]

    def extract_pos_points(mask, num_points=2):
        # True 인 점 추출
        true_indices = np.where(mask)
        random_indices = np.random.choice(len(true_indices[0]), size=num_points, replace=False)
        true_points = np.column_stack((true_indices[0][random_indices], true_indices[1][random_indices]))

        return true_points

    def extract_neg_points(mask, num_points=2):
        false_indices = np.where(~mask)
        random_indices = np.random.choice(len(false_indices[0]), size=num_points, replace=False)
        false_points = np.column_stack((false_indices[0][random_indices], false_indices[1][random_indices]))

        return false_points

    def pos_neg_points_from_mask(self, mask, num_points=1):
        assert len(mask.shape) == 2
        max_indices = torch.topk(mask.view(-1), k=num_points, largest=True).indices
        min_indices = torch.topk(mask.view(-1), k=num_points, largest=False).indices

        max_coords = torch.stack([max_indices // mask.shape[1], max_indices % mask.shape[1]], dim=1)
        min_coords = torch.stack([min_indices // mask.shape[1], min_indices % mask.shape[1]], dim=1)
        coords = torch.cat([max_coords, min_coords], dim=0)

        return coords.cpu().numpy()

    def segment_by_prompts(self, x, prompts_box=None, prompts_points=None, prompts_label=None, prompts_mask=None):

        results = []
        for i in range(x.shape[0]):
            one_frame_x = x[i].permute(1, 2, 0)
            one_frame_x = one_frame_x.cpu().numpy()
            one_frame_x = cv2.normalize(one_frame_x, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            one_frame_x = one_frame_x.astype(np.uint8)

            image = cv2.cvtColor(one_frame_x, cv2.COLOR_BGR2RGB)

            predictor = SamPredictor(self.sam)

            # Segment the image with prompts
            predictor.set_image(image)

            if prompts_box is not None:
                if prompts_box[i] is not None:
                    input_box = np.array(prompts_box[i])[None, :]
                else:
                    input_box = None
            else:
                input_box = None

            if prompts_points is not None:
                if prompts_points[i] is not None:
                    input_point = np.array(prompts_points[i])
                    input_label = np.array(prompts_label)
                else:
                    input_point = None
                    input_label = None
            else:
                input_point = None
                input_label = None

            if prompts_mask is not None:
                if prompts_mask[i] is not None:
                    # mask_input = mask_input[i][None, :, :]
                    mask_input = prompts_mask[i].unsqueeze(0)
                    if self.normalize_mask:
                        mask_input = 30 * (mask_input - mask_input.min()) / (mask_input.max() - mask_input.min()) - 15
                    if self.strong_mask:
                        mask_input = torch.where(mask_input > 0, 10, -10)
                    p12d = (400, 400, 400, 400) 
                    mask_input = F.pad(mask_input, p12d, mode='constant', value=-10)
                else:
                    mask_input = None
            else:
                mask_input = None

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                mask_input=mask_input,
                multimask_output=self.multi_mask,
            )

            if self.multi_mask:
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    mask_input=logits[np.argmax(scores), :, :][None, :, :],
                    multimask_output=False,
                )

            results.append(masks)
            
            if self.visualize:
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    self.show_mask(mask, plt.gca())
                    self.show_points(input_point, input_label, plt.gca())
                    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                    plt.axis('off')
                    # plt.show()  
                    plt.savefig(f"sam_samples/sample_mask{i}.png")
                    plt.close()

        results = np.concatenate(results, axis=0)
        results = torch.from_numpy(results).to(device=x.device).unsqueeze(1)
            
        return results, None, None 

    def segment_all(self, x):
        # TODO: how to select 1 box per frame?
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        init_flag = True 
        results = []
        for i in range(x.shape[0]):
            one_frame_x = x[i].permute(1, 2, 0)
            one_frame_x = one_frame_x.cpu().numpy()
            one_frame_x = cv2.normalize(one_frame_x, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            one_frame_x = one_frame_x.astype(np.uint8)

            image = cv2.cvtColor(one_frame_x, cv2.COLOR_BGR2RGB)
        
            masks = mask_generator.generate(image)
            results.append(masks)

        results = np.concatenate(results, axis=0)
        results = torch.from_numpy(results).to(device=x.device).unsqueeze(1)
        return results, None, None 

    def forward(self, x, mask=None):
        if mask is not None:
            prompts_box = [self.find_bounding_box(mask[m]) for m in range(mask.shape[0])]

            use_points = "pos_neg"
            if use_points == "center":  
                prompts_points = [self.center_points_from_box(prompts_box[i])[0] for i in range(mask.shape[0])]
                prompts_label = np.array([1 ,1])
                
            elif use_points == "pos_neg":
                num_points = 100
                prompts_points = [self.pos_neg_points_from_mask(mask[m], num_points=num_points) for m in range(mask.shape[0])]
                pos_label = torch.ones(num_points, dtype=torch.uint8)
                neg_label = torch.zeros(num_points, dtype=torch.uint8)
                prompts_label = torch.cat([pos_label, neg_label], dim=0).cpu().numpy()

            elif use_points == "all":
                num_points = 1
                num_neg_points = 2
                center_points = np.array([self.center_points_from_box(prompts_box[i] for i in range(mask.shape[0]))])
                random_neg_points = np.array([self.extract_neg_points(mask[i], num_points=num_neg_points) for i in range(mask.shape[0])])
                topk_points = np.array([self.pos_neg_points_from_mask(mask[i], num_points=num_points) for i in range(mask.shape[0])])
                # points = np.array([np.concatenate([random_neg_points, topk_points], axis=0)])
                prompts_points = np.array([np.concatenate([center_points, random_neg_points, topk_points], axis=0)])

                topk_label = torch.ones(num_points, dtype=torch.uint8)
                neg_label = torch.zeros(num_neg_points, dtype=torch.uint8)
                prompts_label = torch.cat([torch.tensor([1]), neg_label,
                                        topk_label, topk_label * 0], dim=0).cpu().numpy()

            prompts_points = np.array(prompts_points).astype(np.uint8)

        use_points = False
        use_box = True
        use_mask_input = False

        x = self.segment_by_prompts(x, 
                                    prompts_points=prompts_points if use_points else None,
                                    prompts_label=prompts_label if use_points else None, 
                                    prompts_box=prompts_box if use_box else None, 
                                    prompts_mask=mask if use_mask_input else None)
        
        return x