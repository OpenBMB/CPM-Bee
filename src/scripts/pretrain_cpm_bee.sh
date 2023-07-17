#! /bin/bash
GPUS_PER_NODE=8

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345

OPTS=""
# model and dataset settings
OPTS+=" --model-config config/cpm-bee-10b.json"
OPTS+=" --dataset ../datasets/datasets.json"
# training settings
OPTS+=" --train-iters 200000"
OPTS+=" --batch-size 2"
OPTS+=" --max-length 2048"
OPTS+=" --lr 0.01"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --loss-scale-factor 2"
OPTS+=" --loss-scale-steps 128"
# log settings
OPTS+=" --inspect-iters 100"
OPTS+=" --log-dir ../logs/train/"
OPTS+=" --tensorboard ../logs/tensorboard/cpm_live_48_4096/"
# saving ckpts
OPTS+=" --save-iters 500"
OPTS+=" --save ../results/"
OPTS+=" --save-name cpm_live_checkpoint"
# loading ckpts
# MODEL_STEPS="0"
# OPTS+=" --start-step ${MODEL_STEPS}"
# OPTS+=" --load ../results/cpm_live_checkpoint-${MODEL_STEPS}.pt"
# OPTS+=" --load-grad"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD

