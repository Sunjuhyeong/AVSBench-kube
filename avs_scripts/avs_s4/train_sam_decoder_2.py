import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import yaml
from config import cfg
from dataloader import S4Dataset_SAM
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask

import torch.nn as nn
from torchvision.models import resnet18 
import sys
from torch.utils.tensorboard import SummaryWriter


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device, model_type='vggish', prompt_dim=256):
        super(audio_extractor, self).__init__()
        self.model_type = model_type
        self.prompt_dim = prompt_dim

        if self.model_type == 'vggish':
            self.audio_backbone = vggish.VGGish(cfg, device)
            # for name, param in self.audio_backbone.named_parameters():
                # print(name, param.requires_grad)

        elif self.model_type == 'resnet18':
            self.audio_backbone = resnet18()
            self.audio_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.audio_backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.audio_backbone.fc = nn.Linear(512, prompt_dim)

            self.init_random_params(self.audio_backbone)

        elif self.model_type == 'beats':
            sys.path.append('../BEATs')
            from BEATs import BEATs, BEATsConfig

            checkpoint = torch.load('../BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.audio_backbone = BEATs(cfg)
            self.audio_backbone.load_state_dict(checkpoint['model'])

    def init_random_params(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(
                    m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward_resnet18(self, audio):
        audio_fea = self.audio_backbone(audio)
        audio_fea = nn.functional.normalize(audio_fea, dim=1)
        return audio_fea.unsqueeze(1)

    def forward_vggish(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea.unsqueeze(1)

    def forward_beats(self, audio):
        # audio: 2, 80320
        audio = audio[0, 0]
        desired_shape = 80320
        if audio.shape[0] < desired_shape:
            padding_length = desired_shape - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding_length), mode='constant', value=0)

        audio = audio.reshape(5, -1) # 5, 16064
        
        # audio = audio.reshape(1, 2, 5, -1).squeeze(0).permute(1, 0, 2) # 5, 2, 16064
        # audio = audio.reshape(10, audio.shape[-1])
        padding_mask = torch.zeros_like(audio).bool()
        audio_fea = self.audio_backbone.extract_features(audio, padding_mask=padding_mask)[0]
        # audio_fea = audio_fea.reshape(5, 2, audio_fea.shape[-1])
        return audio_fea.unsqueeze(1)

    def forward(self, audio):
        if self.model_type == 'vggish':
            audio_fea = self.forward_vggish(audio)
        elif self.model_type == 'resnet18':
            audio_fea = self.forward_resnet18(audio) # B 1 256
        elif self.model_type == 'beats':
            audio_fea = self.forward_beats(audio)

        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--session_name", default="S4_SAM_decoder", type=str, help="the S4 setting")

    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=250, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument('--sa_loss_flag', action='store_true', default=False, help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int, help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--prompts", type=str, default='audio', help='')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument("--test_weights", type=str, default='', help='path of trained model')
    parser.add_argument("--train_weights", type=str, default='train_logs/S4_SAM_decoder_20230910-145056/checkpoints/S4_SAM_decoder_best.pth', help='path of trained model')
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument(
        "--audio_model_type", default="vggish", help="audio model type"
    )
    parser.add_argument(
        "--attn_depth", default=8, type=int, help="audio model type"
    )
    # parser.add_argument(
        # "--device", default=1, type=str, help="audio model type"
    # )
    parser.add_argument('--config', default="sam_avs_adapter.yaml")
    
    args = parser.parse_args()

    from model import SAM_decoder_2 as SAM_decoder
    print('==> Use SAM decoder as the visual backbone...')

    torch.multiprocessing.set_start_method('spawn')
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)
    
    if args.mode == 'test':
        args.log_dir = './test_logs'

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train_sam_decoder_2.py', 'config.py', 'dataloader.py', './model/SAM_decoder_2.py', './model/SAM_encoder_ann.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))
    writer = SummaryWriter(args.log_dir)
    # Model
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.config = config
    model = SAM_decoder.Decoder(depth=args.attn_depth, config=args.config)
    # model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    
    # model.load_state_dict(torch.load("../sam_sandbox/sama_s4_best.pth"), strict=False)
    prompt_path = "../sam_sandbox/prompt_encoder.pth"
    mask_decoder_path = "../sam_sandbox/mask_decoder.pth"
    pe_layer_path = "../sam_sandbox/pe_layer.pth"
    model.prompt_encoder.load_state_dict(torch.load(prompt_path))
    model.mask_decoder.load_state_dict(torch.load(mask_decoder_path))
    model.pe_layer.load_state_dict(torch.load(pe_layer_path))
    for i in range(32):
        # if i < 9: 
            # model.image_encoder.blocks[i].to("cuda:3")
        # elif 9 <= i < 18:
            # model.image_encoder.blocks[i].to("cuda:2")
        if i < 26:
            model.image_encoder.blocks[i].to("cuda:1")
        else:
            model.image_encoder.blocks[i].to("cuda:0")

    if os.path.exists(args.train_weights):
        model.load_state_dict(torch.load(args.train_weights), strict=False)
        print("load pre_trained weights")
    # for k, v in model.named_parameters():
    #         print(k, v.requires_grad)

    for name, para in model.named_parameters():        
        if "prompt_generator" in name:
            para.requires_grad_(True)
        elif "mask_decoder" in name:
            para.requires_grad_(True)
        else:
            para.requires_grad_(False)

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device, model_type=args.audio_model_type)
    audio_backbone.cuda()
    # audio_backbone.eval()

    # Data
    train_dataset = S4Dataset_SAM('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset_SAM('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)
    test_dataset = S4Dataset_SAM('test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
        audio_backbone_params = audio_backbone.parameters()
        optimizer_audio = torch.optim.Adam(audio_backbone_params, args.lr * 10)
    else:
        audio_backbone.eval()
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    
    if args.mode == 'train':
        for epoch in range(args.max_epoches):
            # Train:
            model.train()
            if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
                audio_backbone.train()
            for n_iter, batch_data in enumerate(train_dataloader):
                imgs, img_embeddings, audio, mask, category, video_name = batch_data # imgs[bs, 5, 3, 224, 224], audio[bs, 5, 1, 96, 64], mask[bs, 1, 1, 224, 224]

                imgs = imgs.cuda()
                img_embeddings = img_embeddings.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                img_embeddings = img_embeddings.view(B*frame, img_embeddings.shape[-3], img_embeddings.shape[-2], img_embeddings.shape[-1])
                mask = mask.view(B, mask.shape[-2], mask.shape[-1])
                if args.audio_model_type != 'beats':
                    audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]
                
                with torch.no_grad():
                    audio_feature = audio_backbone(audio) # [B*T, 128]
                
                if args.prompts == 'audio':
                    output = model(imgs, img_embeddings, audio_feature=audio_feature, mode=1) # [bs*5, 1, 224, 224]
                elif args.prompts == 'box':
                    output = model(img_embeddings, masks=mask) # [bs*5, 1, 224, 224]

                loss, loss_dict = IouSemanticAwareLoss(output.squeeze(1), mask.unsqueeze(1), \
                                                    a_fea_list=None, v_map_list=None, \
                                                    lambda_1=args.lambda_1, \
                                                    count_stages=args.sa_loss_stages, \
                                                    sa_loss_flag=args.sa_loss_flag, \
                                                    mask_pooling_type=args.mask_pooling_type)

                avg_meter_total_loss.add({'total_loss': loss.item()})
                avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
                avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
                    optimizer_audio.zero_grad()
                    optimizer_audio.step()
                writer.add_scalar("train loss",
                                loss.item(), global_step)
                global_step += 1

                if (global_step-1) % 50 == 0:
                    train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.4f'%(
                                global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), args.lambda_1, optimizer.param_groups[0]['lr'])
                    # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                    #         'Total_Loss:%.4f' % (avg_meter_loss.pop('total_loss')),
                    #         'iou_loss:%.4f' % (avg_meter_iou_loss.pop('iou_loss')),
                    #         'sa_loss:%.4f' % (avg_meter_sa_loss.pop('sa_loss')),
                    #         'lambda_1:%.4f' % (args.lambda_1),
                    #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                    # print(train_log, flush=True)
                    logger.info(train_log)

            # Validation:
            model.eval()
            audio_backbone.eval()
            with torch.no_grad():
                # for n_iter, batch_data in enumerate(val_dataloader):
                for n_iter, batch_data in enumerate(test_dataloader):
                    imgs, img_embeddings, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                    imgs = imgs.cuda()
                    img_embeddings = img_embeddings.cuda()
                    audio = audio.cuda()
                    mask = mask.cuda()
                    B, frame, C, H, W = imgs.shape
                    imgs = imgs.view(B*frame, C, H, W)
                    img_embeddings = img_embeddings.view(B*frame, img_embeddings.shape[-3], img_embeddings.shape[-2], img_embeddings.shape[-1])
                    mask = mask.view(B*frame, mask.shape[-2], mask.shape[-1])
                    if args.audio_model_type != 'beats':
                        audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                    with torch.no_grad():
                        audio_feature = audio_backbone(audio)

                    if args.prompts == 'audio':
                        output = model(imgs, img_embeddings, audio_feature=audio_feature, mode=5) # [bs*5, 1, 224, 224]
                    elif args.prompts == 'box':
                        output = model(img_embeddings, masks=mask) # [bs*5, 1, 224, 224]

                    miou = mask_iou(output.squeeze(1).squeeze(1), mask)
                    avg_meter_miou.add({'miou': miou})

                miou = (avg_meter_miou.pop('miou'))
                if miou > max_miou:
                    model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                    torch.save(model.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save best model to %s'%model_save_path)

                miou_list.append(miou)
                max_miou = max(miou_list)

                val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
                # print(val_log)
                logger.info(val_log)
                writer.add_scalar("val miou",
                    miou.item(), epoch)
        logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
        writer.close()
    
    elif args.mode == 'test':
        if os.path.exists(args.test_weights):
            model.load_state_dict(torch.load(args.test_weights))
       
        audio_backbone.eval()
        model.eval()
        
        with torch.no_grad():
            # Testing:
            for n_iter, batch_data in enumerate(test_dataloader):
                imgs, img_embeddings, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                img_embeddings = img_embeddings.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                
                B, frame, H, W, C = imgs.shape
                
                imgs = imgs.view(B*frame, C, H, W)
                img_embeddings = img_embeddings.view(B*frame, img_embeddings.shape[2], img_embeddings.shape[3], img_embeddings.shape[4])
                mask = mask.view(B*frame, H, W)
                if args.audio_model_type != 'beats':
                    audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                if args.prompts == 'audio':
                    output = model(img_embeddings, audio_feature=audio_feature) # [bs*5, 1, 224, 224]
                elif args.prompts == 'box':
                    output = model(img_embeddings, masks=mask) # [bs*5, 1, 224, 224]


                if args.save_pred_mask:
                    mask_save_path = os.path.join(log_dir, 'pred_masks')
                    save_mask(output.squeeze(1), mask_save_path, category, video_name)
                miou = mask_iou(output.squeeze(1).squeeze(1), mask)

                avg_meter_miou.add({'miou': miou})
                F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
                avg_meter_F.add({'F_score': F_score})
                print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
            
            miou = (avg_meter_miou.pop('miou'))
            F_score = (avg_meter_F.pop('F_score'))
            print('test miou:', miou.item())
            print('test F_score:', F_score)
            logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))
