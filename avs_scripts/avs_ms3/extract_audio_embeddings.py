import os
import random
import torch
import numpy as np
import argparse
from tqdm import tqdm

from dataloader import MS3Dataset_SAM
from model import SAM_encoder_ann
from config import cfg
from train_test_SAM_All_2 import audio_extractor
import yaml
from torch import Tensor, nn 
from functools import partial

def save_embedding(image_embedding, img_path, video_name):
    
    for i in range(image_embedding.shape[0]):
        torch.save(image_embedding[i], os.path.join(img_path, f'{video_name[0]}_{i+1}.pt'))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('-s', '--split', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--config', default="sam_avs_adapter.yaml")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    encoder_mode = config['model']['args']['encoder_mode']
        
    # Model
    model = SAM_encoder_ann.AudioPromptGenerator(
        embed_dim=encoder_mode['embed_dim'], 
        depth=encoder_mode['depth'], 
        aud_in_dim=128, 
        aud_hidden_dim=512
    )
    model = torch.nn.DataParallel(model).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.module.load_state_dict(torch.load("../sam_sandbox/audio_prompt_generator_sama_ms3.pth"))
    
    audio_backbone = audio_extractor(cfg, device, model_type="vggish")
    audio_backbone.cuda()
    data_root = "/mnt/ssd1/seon/AVSBench-main/avsbench_data/Multi-sources/ms3_data/audio_prompts_sama"
    if not os.path.exists(data_root):   
        os.makedirs(data_root)
    
    print(f"{args.split}")

    model.eval()
    with torch.no_grad():
        for split in args.split:
            dataset = MS3Dataset_SAM(split, get_img_embedding=False)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
            
            for n_iter, batch_data in enumerate(tqdm(dataloader)):
                imgs, audio, _, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

                audio_path = os.path.join(data_root, video_name[0])
                if not os.path.exists(audio_path):   
                    os.makedirs(audio_path)

                audio = audio.cuda()
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                audio_feature = audio_backbone(audio)
                audio_prompts = model(audio_feature[:,0,:])
                audio_prompts = torch.stack(audio_prompts, dim=0)
                audio_prompts = torch.nn.AdaptiveAvgPool1d(1)(audio_prompts.permute(1,2,0))[:, :, 0]
                save_embedding(audio_prompts, audio_path, video_name)

            print(f'{split} data done')