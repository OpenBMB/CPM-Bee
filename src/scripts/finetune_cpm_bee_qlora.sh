#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config config/cpm-bee-10b.json"
OPTS+=" --dataset path/to/dataset"
OPTS+=" --eval_dataset path/to/eval/dataset"
OPTS+=" --epoch 10"
OPTS+=" --batch-size 2"
OPTS+=" --train-iters 100"
OPTS+=" --save-name cpm_bee_finetune"
OPTS+=" --max-length 2048"
OPTS+=" --save results/"
OPTS+=" --lr 0.0001"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 10"
OPTS+=" --early-stop-patience 10"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load quantized_model.pt"

CMD="python finetune_cpm_bee_qlora.py ${OPTS}"

echo ${CMD}
$CMD