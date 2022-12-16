now=$(date +"%Y%m%d_%H%M%S")

now=FT30_4E4_D065_EMA98_DPR00
MODEL=CLIP_L14
KEY=CLIP_openai_Large_P14

OUTPUT_DIR=OUTPUT/CLIP_ft/${KEY}/${now}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

DATA_PATH=/tmp/DATA/IN1K
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model ${MODEL} --data_path $DATA_PATH \
    --input_size 224 \
    --finetune True \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 --lr 4e-4 --update_freq 4 \
    --warmup_epochs 5 --epochs 30 \
    --layer_decay 0.65 --backbone_decay 1 \
    --drop_path 0 \
    --dist_eval --eval_all \
    --enable_deepspeed \
    --clip_mean_and_std \
    --layer_scale_init_value 0 \
    --abs_pos_emb --disable_rel_pos_bias \
    --weight_decay 0.05 --mixup 0 --cutmix 0 \
    --nb_classes 1000 --model_prefix visual.\
    --model_ema --model_ema_decay 0.9998 \
    2>&1 | tee -a ${OUTPUT_DIR}/log.txt




