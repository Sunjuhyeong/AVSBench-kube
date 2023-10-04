setting='S4'
# visual_backbone="resnet"
# visual_backbone="pvt"
# visual_backbone="dino"
# visual_backbone="resnet_visual_only" 
visual_backbone="sam"

# spring.submit arun --gpu -n1 --gres=gpu:1 --quotatype=auto --job-name="test_${setting}_${visual_backbone}" \
# "
CUDA_VISIBLE_DEVICES=0 python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    --save_pred_mask \
# "
    # --weights "./train_logs/S4_pvt_20230708-144225/checkpoints/S4_pvt_best.pth" \
