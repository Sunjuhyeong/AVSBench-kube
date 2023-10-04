setting='MS3'
# visual_backbone="resnet" # "resnet" or "pvt"
# visual_backbone="pvt" # "resnet" or "pvt"
# visual_backbone="dino" # "resnet" or "pvt"
visual_backbone="sam"


# spring.submit arun --gpu -n1 --gres=gpu:1 --quotatype=auto --job-name="test_${setting}_${visual_backbone}" \
python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    --save_pred_mask \

# --weights "./train_logs/MS3_pvt_20230708-215219/checkpoints/MS3_pvt_best.pth" \
