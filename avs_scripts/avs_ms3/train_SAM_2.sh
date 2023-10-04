
for i in 0.00001 0.0001 
do
        python train_test_SAM_decoder_3.py \
                --depth 7 \
                --device 1 \
                --lambda_2 $i \
                --train_weights train_logs/MS3_SAM_decoder_20230911-002457/checkpoints/MS3_SAM_decoder_best.pth
done
