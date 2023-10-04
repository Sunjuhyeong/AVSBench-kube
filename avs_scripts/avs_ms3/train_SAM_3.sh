
for i in 0.00001 0.0001 
do
        python train_test_SAM_decoder_3.py \
                --depth 7 \
                --device 0 \
                --lambda_2 $i
done
