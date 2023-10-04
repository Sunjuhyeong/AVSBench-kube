import os
import random
import torch
import numpy as np
import argparse
from tqdm import tqdm

from dataloader import S4Dataset_SAM
from model import SAM_encoder_ann
from config import cfg
from train_sam_decoder_rep import audio_extractor
import yaml
from torch import Tensor, nn 
from functools import partial

def save_embedding(image_embedding, img_path, video_name):
    for i in range(image_embedding.shape[0]):
        torch.save(
            image_embedding[i], os.path.join(img_path, f"{video_name[0]}_{i+1}.pt")
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("-s", "--split", nargs="+", default=["train", "val", "test"])
    parser.add_argument('--config', default="sam_avs_adapter.yaml")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    encoder_mode = config['model']['args']['encoder_mode']
        
    model = SAM_encoder_ann.Extractor(encoder_mode=encoder_mode)

    model = torch.nn.DataParallel(model).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.module.load_state_dict(torch.load("../sam_sandbox/sama_s4_best.pth"), strict=False)
    
    audio_backbone = audio_extractor(cfg, device, model_type="vggish")
    audio_backbone.cuda()
    
    data_root = "/mnt/ssd1/seon/AVSBench-main/avsbench_data/Single-source/s4_data/visual_embeddings_sama"
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    print(f"{args.split}")

    model.eval()
    audio_backbone.eval()
    with torch.no_grad():
        for split in args.split:
            dataset = S4Dataset_SAM(split, get_img_embedding=False)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            dir_path = os.path.join(data_root, split)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for n_iter, batch_data in enumerate(tqdm(dataloader)):
                (
                    imgs,
                    audio,
                    _,
                    category,
                    video_name,
                ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

                category_path = os.path.join(dir_path, category[0])
                if not os.path.exists(category_path):
                    os.makedirs(category_path)

                img_path = os.path.join(category_path, video_name[0])
                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                # imgs = imgs.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                imgs = imgs.cuda()
                audio = audio.cuda()
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                image_embeddings_from_encoder_list = []
                with torch.no_grad():
                    for i in range(5):  
                        image_embeddings_from_encoder = model(imgs[i].unsqueeze(0), audio_feature[i])
                        image_embeddings_from_encoder_list.append(image_embeddings_from_encoder[0])

                img_embeddings = torch.stack(image_embeddings_from_encoder_list, dim=0)
                save_embedding(img_embeddings, img_path, video_name)

            print(f"{split} data done")
