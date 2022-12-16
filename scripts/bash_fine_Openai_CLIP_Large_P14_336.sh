now=$(date +"%Y%m%d_%H%M%S")
GPUS=`nvidia-smi -L | wc -l`

now=FT30_6E4_D065_EMA98
MODEL=CLIP_L14_336
KEY=CLIP_openai_Large_P14_336

OUTPUT_DIR=OUTPUT/CLIP_ft/${KEY}/${now}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR


DATA_PATH=/tmp/DATA/IN1K
[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZ_BATCHAI_TASK_INDEX" ]] && RANK=0 || RANK=$AZ_BATCHAI_TASK_INDEX
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=192.168.0.78 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=45788 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

echo "run command: python -m torch.distributed.launch --nnodes ${NODE_COUNT} --node_rank ${RANK} --master_addr ${MASTER_ADDR}  --master_port ${MASTER_PORT} --nproc_per_node ${GPUS} new_main.py $@"

python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS} \
    run_class_finetuning.py \
    --model ${MODEL} --data_path $DATA_PATH \
    --input_size 336 \
    --finetune True \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 20 --lr 6e-4 --update_freq 4 \
    --warmup_epochs 5 --epochs 30 \
    --layer_decay 0.65 --backbone_decay 1 \
    --drop_path 0 \
    --dist_eval --eval_all --no_save_ckpt \
    --enable_deepspeed \
    --clip_mean_and_std \
    --layer_scale_init_value 0 \
    --abs_pos_emb --disable_rel_pos_bias \
    --weight_decay 0.05 --mixup 0 --cutmix 0 \
    --nb_classes 1000 --model_prefix visual.\
    --model_ema --model_ema_decay 0.9998 \
    2>&1 | tee -a ${OUTPUT_DIR}/log.txt



