import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import MS3Dataset_SAM1 as MS3Dataset_SAM
from torchvggish import vggish
from loss import IouSemanticAwareLoss, IouSemanticAwareLoss_AV

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask

import torch.nn as nn
from torchvision.models import resnet18 
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

    def forward(self, audio):
        if self.model_type == 'vggish':
            audio_fea = self.forward_vggish(audio)
        elif self.model_type == 'resnet18':
            audio_fea = self.forward_resnet18(audio) # B 1 256

        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_name", default="MS3_SAM_decoder", type=str, help="the MS3 setting"
    )

    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=200, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument(
        "--masked_av_flag",
        action="store_true",
        default=False,
        help="additional sa/masked_va loss for five frames",
    )
    parser.add_argument(
        "--lambda_1", default=0, type=float, help="weight for balancing l4 loss"
    )
    parser.add_argument(
        "--masked_av_stages",
        default=[],
        nargs="+",
        type=int,
        help="compute sa/masked_va loss in which stages: [0, 1, 2, 3]",
    )
    parser.add_argument(
        "--threshold_flag",
        action="store_true",
        default=False,
        help="whether thresholding the generated masks",
    )
    parser.add_argument(
        "--mask_pooling_type",
        default="avg",
        type=str,
        help="the manner to downsample predicted masks",
    )
    parser.add_argument(
        "--norm_fea_flag",
        action="store_true",
        default=False,
        help="normalize audio-visual features",
    )
    parser.add_argument(
        "--closer_flag",
        action="store_true",
        default=False,
        help="use closer loss for masked_va loss",
    )
    parser.add_argument(
        "--euclidean_flag",
        action="store_true",
        default=False,
        help="use euclidean distance for masked_va loss",
    )
    parser.add_argument(
        "--kl_flag",
        action="store_true",
        default=False,
        help="use kl loss for masked_va loss",
    )

    parser.add_argument(
        "--load_s4_params",
        action="store_true",
        default=False,
        help="use S4 parameters for initilization",
    )
    parser.add_argument(
        "--trained_s4_model_path", type=str, default="", help="pretrained S4 model"
    )

    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--prompts", type=str, default="audio", help="")
    parser.add_argument("--log_dir", default="./train_logs", type=str)
    parser.add_argument(
        # "--test_weights", type=str, default="", help="path of trained model"
        "--test_weights", type=str, default="train_logs/MS3_SAM_decoder_20230920-001637/checkpoints/MS3_SAM_decoder_best.pth", help="path of trained model"
    )
    parser.add_argument(
        "--train_weights", type=str, default="", help="path of trained model"
    )
    parser.add_argument(
        "--save_pred_mask",
        action="store_true",
        default=True,
        help="save predited masks or not",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=False,
        help="scheduler",
    )
    parser.add_argument(
        "--device",
        # action="store_true",
        default=2,
        help="scheduler",
    )
    parser.add_argument(
        "--depth",
        # action="store_true",
        default=7,
        type=int,
        help="scheduler",
    )
    parser.add_argument(
        "--refiner_decoder_path",
        # action="store_true",
        default='train_logs/MS3_SAM_decoder_20230911-002457/checkpoints/MS3_SAM_decoder_best.pth'
    )
    parser.add_argument(
        "--av_fusion_loss_flag",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lambda_2", default=1, type=float, help="weight for balancing l4 loss"
    )
    

    args = parser.parse_args()

    from model import SAM_decoder_refiner as SAM_decoder
    print('==> Use SAM decoder as the visual backbone...')

    torch.multiprocessing.set_start_method("spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    if args.mode == "test":
        args.log_dir = "./test_logs"

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(
        args.log_dir, "{}".format(time.strftime(prefix + "_%Y%m%d-%H%M%S"))
    )
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, "scripts")
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [
        "train_test_SAM_decoder_refiner.py",
        "config.py",
        "dataloader.py",
        "./model/SAM_decoder_refiner.py",
        "loss.py",
    ]
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, "log.txt"))
    logger = logging.getLogger(__name__)
    logger.info("==> Config: {}".format(cfg))
    logger.info("==> Arguments: {}".format(args))
    logger.info("==> Experiment: {}".format(args.session_name))
    writer = SummaryWriter(args.log_dir)

    # Model
    # model = SAM_decoder.Decoder(depth=args.depth)
    model = SAM_decoder.Refiner(depth=args.depth, refiner_decoder_path=args.refiner_decoder_path)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(
        "==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6)
    )
    if os.path.exists(args.train_weights):
        model.module.load_state_dict(torch.load(args.train_weights))
        print("load latest weights")

    # load pretrained S4 model
    if args.load_s4_params:  # fine-tune single sound source segmentation model
        model_dict = model.state_dict()
        s4_state_dicts = torch.load(args.trained_s4_model_path)
        state_dict = {
            "module." + k: v
            for k, v in s4_state_dicts.items()
            if "module." + k in model_dict.keys()
        }
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        logger.info(
            "==> Reload pretrained S4 model from %s" % (args.trained_s4_model_path)
        )

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device, model_type='vggish')
    audio_backbone.cuda()
    print("==> Total params of audio backbone: %.2fM" % (sum(p.numel() for p in audio_backbone.parameters()) / 1e6))

    # Data
    train_dataset = MS3Dataset_SAM("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = MS3Dataset_SAM("val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_dataset = MS3Dataset_SAM("test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.3, min_lr=1e-6)

    if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
        audio_backbone_params = audio_backbone.parameters()
        optimizer_audio = torch.optim.Adam(audio_backbone_params, args.lr * 10)
    else:
        audio_backbone.eval()
    avg_meter_total_loss = pyutils.AverageMeter("total_loss")
    avg_meter_sa_loss = pyutils.AverageMeter("sa_loss")
    avg_meter_iou_loss = pyutils.AverageMeter("iou_loss")
    avg_meter_av_loss = pyutils.AverageMeter("av_loss")

    avg_meter_miou = pyutils.AverageMeter("miou")
    avg_meter_F = pyutils.AverageMeter("F_score")

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0

    if args.mode == "train":
        for epoch in range(args.max_epoches):
            model.train()
            if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
                audio_backbone.train()

            for n_iter, batch_data in enumerate(train_dataloader):
                (
                    imgs,
                    img_embeddings,
                    audio,
                    mask,
                    video_name,
                ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]

                imgs = imgs.cuda()
                img_embeddings = img_embeddings.cuda()
                audio = audio.cuda()
                mask = mask.cuda()

                # B, frame, C, H, W = imgs.shape
                B, frame, H, W, C = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                img_embeddings = img_embeddings.view(
                    B * frame,
                    img_embeddings.shape[2],
                    img_embeddings.shape[3],
                    img_embeddings.shape[4],
                )
                mask_num = 5
                mask = mask.view(B * mask_num, 1, H, W)
                audio = audio.view(
                    -1, audio.shape[2], audio.shape[3], audio.shape[4]
                )  # [B*T, 1, 96, 64]
                # with torch.no_grad():
                audio_feature = audio_backbone(audio)  # [B*T, 128]

                if args.prompts == "audio":
                    output = model(
                        img_embeddings, audio_feature=audio_feature, masks=mask
                    )  # [bs*5, 1, 224, 224]
                elif args.prompts == "box":
                    output = model(img_embeddings, masks=mask)  # [bs*5, 1, 224, 224]

                # output, v_map_list, a_fea_list = model(imgs, audio_feature) # [bs*5, 1, 224, 224]
                loss, loss_dict = IouSemanticAwareLoss_AV(
                    output,
                    mask,
                    a_fea_list=None,
                    v_map_list=None,
                    sa_loss_flag=args.masked_av_flag,
                    lambda_1=args.lambda_1,
                    count_stages=args.masked_av_stages,
                    mask_pooling_type=args.mask_pooling_type,
                    threshold=args.threshold_flag,
                    norm_fea=args.norm_fea_flag,
                    closer_flag=args.closer_flag,
                    euclidean_flag=args.euclidean_flag,
                    kl_flag=args.kl_flag,
                    av_fusion_loss_flag=args.av_fusion_loss_flag,
                    av_fusion_loss=None,
                    lambda_2=args.lambda_2,
                )

                avg_meter_total_loss.add({"total_loss": loss.item()})
                avg_meter_iou_loss.add({"iou_loss": loss_dict["iou_loss"]})
                avg_meter_sa_loss.add({"sa_loss": loss_dict["sa_loss"]})
                avg_meter_av_loss.add({"av_loss": loss_dict["av_loss"]})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
                    optimizer_audio.zero_grad()
                    optimizer_audio.step()
                writer.add_scalar("train loss",
                                   loss.item(), global_step)
                global_step += 1
                if (global_step - 1) % 20 == 0:
                    train_log = (
                        "Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, av_loss:%.4f, lr: %.5f"
                        % (
                            global_step - 1,
                            max_step,
                            avg_meter_total_loss.pop("total_loss"),
                            avg_meter_iou_loss.pop("iou_loss"),
                            avg_meter_sa_loss.pop("sa_loss"),
                            avg_meter_av_loss.pop("av_loss"),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                    # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                    #         'Total_Loss:%.4f' % (avg_meter_total_loss.pop('total_loss')),
                    #         'iou_loss:%.4f' % (avg_meter_L1.pop('iou_loss')),
                    #         'sa_loss:%.4f' % (avg_meter_L4.pop('sa_loss')),
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
                    (
                        imgs,
                        img_embeddings,
                        audio,
                        mask,
                        video_name,
                    ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                    imgs = imgs.cuda()
                    img_embeddings = img_embeddings.cuda()
                    audio = audio.cuda()
                    mask = mask.cuda()
                    B, frame, H, W, C = imgs.shape
                    imgs = imgs.view(B * frame, C, H, W)
                    img_embeddings = img_embeddings.view(
                        B * frame,
                        img_embeddings.shape[2],
                        img_embeddings.shape[3],
                        img_embeddings.shape[4],
                    )
                    mask = mask.view(B * frame, H, W)
                    audio = audio.view(
                        -1, audio.shape[2], audio.shape[3], audio.shape[4]
                    )
                    audio_feature = audio_backbone(audio)

                    if args.prompts == "audio":
                        output = model(
                            img_embeddings, audio_feature=audio_feature
                        )  # [bs*5, 1, 224, 224]
                    elif args.prompts == "box":
                        output = model(
                            img_embeddings, masks=mask
                        )  # [bs*5, 1, 224, 224]

                    miou = mask_iou(output.squeeze(1), mask)
                    avg_meter_miou.add({"miou": miou})

                miou = avg_meter_miou.pop("miou")
                if miou > max_miou:
                    model_save_path = os.path.join(
                        checkpoint_dir, "%s_best.pth" % (args.session_name)
                    )
                    torch.save(model.module.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info("save best model to %s" % model_save_path)

                miou_list.append(miou)
                max_miou = max(miou_list)

                val_log = "Epoch: {}, Miou: {}, maxMiou: {}".format(
                    epoch, miou, max_miou
                )
                # print(val_log)
                logger.info(val_log)
                writer.add_scalar("val miou",
                    miou.item(), epoch)

            if args.use_scheduler:
                pre_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(miou)
                if pre_lr != optimizer.param_groups[0]["lr"]:
                    logger.info("lr decay to {}".format(optimizer.param_groups[0]["lr"]))
            
        logger.info("best val Miou {} at peoch: {}".format(max_miou, best_epoch))

        writer.close()


    elif args.mode == "test":
        if os.path.exists(args.test_weights):
            model.module.load_state_dict(torch.load(args.test_weights))
            print(f"load from {args.test_weights}")
        model.eval()
        audio_backbone.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(test_dataloader):
                (
                    imgs,
                    img_embeddings,
                    audio,
                    mask,
                    video_name,
                ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

                imgs = imgs.cuda()
                img_embeddings = img_embeddings.cuda()
                audio = audio.cuda()
                mask = mask.cuda()

                B, frame, C, H, W = imgs.shape

                imgs = imgs.view(B * frame, C, H, W)
                img_embeddings = img_embeddings.view(
                    B * frame,
                    img_embeddings.shape[2],
                    img_embeddings.shape[3],
                    img_embeddings.shape[4],
                )
                mask = mask.view(B * frame, mask.shape[-2], mask.shape[-1])
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                audio_feature = audio_backbone(audio)

                if args.prompts == "audio":
                    output = model(
                        img_embeddings, audio_feature=audio_feature
                    )  # [bs*5, 1, 224, 224]
                elif args.prompts == "box":
                    output = model(img_embeddings, masks=mask)  # [bs*5, 1, 224, 224]

                if args.save_pred_mask:
                    mask_save_path = os.path.join(log_dir, "pred_masks")
                    save_mask(output.squeeze(1), mask_save_path, video_name)
                miou = mask_iou(output.squeeze(1).squeeze(1), mask)

                avg_meter_miou.add({"miou": miou})
                F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
                avg_meter_F.add({"F_score": F_score})
                print("n_iter: {}, iou: {}, F_score: {}".format(n_iter, miou, F_score))

            miou = avg_meter_miou.pop("miou")
            F_score = avg_meter_F.pop("F_score")
            print("test miou:", miou.item())
            print("test F_score:", F_score)
            logger.info("test miou: {}, F_score: {}".format(miou.item(), F_score))
