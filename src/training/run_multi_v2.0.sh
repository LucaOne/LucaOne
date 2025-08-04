#!/bin/bash
# for dataset
## the dataset name
## the dataset contain the entire dataset(gene+prot)
DATASET_NAME="lucagplm"
## the dataset version
DATASET_TYPE="v2.0"

# for pretrain tasks
## pretraining task levels
TASK_TYPE="token_level,span_level,seq_level,structure_level"
## pretraining task names
PRETRAIN_TASK_LEVEL_NAME="gene_mask,gene_type,gene_taxonomy,prot_mask,prot_site,prot_homo,prot_domain,prot_taxonomy,prot_keyword,prot_structure"


# for input
## max seq length
MAX_LENGTH=1280
## padding strategy
PADDING_TYPE="right"
## truncation strategy
TRUNCATION_TYPE="right"

# for dataloader
# 流式加载的缓存大小
buffer_size=10240
# 数据加载器worker数
worker_num=8
# seed 用于复现或者中间训练出错保护现场(因为是伪随机)
seed=1111

# for model
## model type
MODEL_TYPE="lucaone_gplm"
## embedding dim
HIDDEN_SIZE=2560
## num layers(transformer blocks)
NUM_ATTENTION_LAYERS=20
## num heads(transformer heads)
NUM_ATTENTION_HEADS=40
## pooling type(not use)
POOLING_TYPE="value_attention"
## best metric type for model finalization
BEST_METRIC_TYPE="loss"

# for training
## max epochs
num_train_epochs=5
## accumulation gradient steps
gradient_accumulation_steps=32
# 间隔多少个step在log文件中写入loss（实际上是gradient_accumulation_steps与loss_logging_steps的最小公倍数, 这里是4000）
loss_logging_steps=1000
# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数, 这里是32000）
logging_steps=32000
# checkpoint的间隔step数
save_steps=200000
# warmup_steps个step到达peak lr，实际上是warmup_steps=warmup_steps/gradient_accumulation_steps
warmup_steps=64000
# 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
# 最后想要变成多大的值比如从lr1->lr2，那么就是(max_epochs*sample_cnt)*lr1/(n_gpu * batch_size * gradient_accumulation_steps*lr2))进行估算
max_steps=16000000
# batch size for one GPU
batch_size=1
# 最大学习速率(peak learning rate)
learning_rate=2e-4

time_str=$(date "+%Y%m%d%H%M%S")

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ..
python -W ignore -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=8 \
       run.py \
       --time_str $time_str \
       --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
       --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
       --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
       --num_attention_heads $NUM_ATTENTION_HEADS \
       --num_hidden_layers $NUM_ATTENTION_LAYERS \
       --hidden_size $HIDDEN_SIZE \
       --max_length $MAX_LENGTH  \
       --vocab_path ../vocab/$DATASET_NAME/$DATASET_TYPE/vocab.txt \
       --tokenizer_dir ../vocab/$DATASET_NAME/$DATASET_TYPE/vocab.txt \
       --add_special_tokens \
       --padding $PADDING_TYPE \
       --truncation $TRUNCATION_TYPE \
       --pooling_type $POOLING_TYPE \
       --model_type $MODEL_TYPE \
       --model_config ../config/$MODEL_TYPE.json \
       --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/train/ \
       --dev_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/dev/ \
       --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/test/ \
       --gene_mask_label_filepath ../vocab/$DATASET_NAME/$DATASET_TYPE/vocab.txt \
       --prot_mask_label_filepath ../vocab/$DATASET_NAME/$DATASET_TYPE/vocab.txt \
       --gene_type_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/gene_type_span_level_label_v2.txt \
       --prot_homo_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_homo_span_level_label_v2.txt \
       --prot_site_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_site_span_level_label_v2.txt \
       --prot_domain_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_domain_span_level_label_v2.txt \
       --gene_taxonomy_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/gene_taxonomy_seq_level_label_v2.txt \
       --prot_taxonomy_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_taxonomy_seq_level_label_v2.txt \
       --prot_keyword_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_keyword_seq_level_label_v2.txt \
       --prot_structure_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/prot_structure_structure_level_label_v2.txt \
       --trans_label_filepath ../label/$DATASET_NAME/$DATASET_TYPE/trans_pair_level_label_v2.txt \
       --gene_mask_output_mode multi_class \
       --prot_mask_output_mode multi_class \
       --gene_type_output_mode multi_class \
       --prot_homo_output_mode multi_class \
       --prot_site_output_mode multi_class \
       --prot_domain_output_mode multi_class \
       --gene_taxonomy_output_mode multi_class\
       --prot_taxonomy_output_mode multi_class\
       --prot_keyword_output_mode multi_label \
       --prot_structure_output_mode regression \
       --trans_output_mode binary_class \
       --gene_mask_loss_type cce \
       --prot_mask_loss_type cce \
       --gene_type_loss_type cce \
       --prot_homo_loss_type cce \
       --prot_site_loss_type cce \
       --prot_domain_loss_type cce \
       --gene_taxonomy_loss_type cce \
       --prot_taxonomy_loss_type cce \
       --prot_keyword_loss_type bce \
       --prot_structure_loss_type l1 \
       --trans_loss_type bce \
       --ignore_index -100 \
       --gene_mask_classifier_size 2048 \
       --prot_mask_classifier_size 2048 \
       --gene_type_classifier_size 128 \
       --prot_homo_classifier_size 4096 \
       --prot_site_classifier_size 1024 \
       --prot_domain_classifier_size 10240 \
       --gene_taxonomy_classifier_size 2048 \
       --prot_taxonomy_classifier_size 2048 \
       --prot_keyword_classifier_size 2048 \
       --prot_structure_classifier_size 128 \
       --trans_classifier_size 128 \
       --gene_mask_weight 1.0 \
       --prot_mask_weight 1.0 \
       --gene_type_weight 0.2 \
       --prot_homo_weight 0.2 \
       --prot_site_weight 0.2 \
       --prot_domain_weight 0.2 \
       --gene_taxonomy_weight 0.2 \
       --prot_taxonomy_weight 0.2 \
       --prot_keyword_weight 1.0 \
       --prot_structure_weight 1.0 \
       --prot_secondary_weight 1.0 \
       --prot_contact_weight 1.0 \
       --trans_weight 1.0 \
       --buffer_size $buffer_size \
       --worker_num $worker_num \
       --seed $seed \
       --pretrain_task_level_type $TASK_TYPE \
       --pretrain_task_level_name $PRETRAIN_TASK_LEVEL_NAME \
       --per_gpu_train_batch_size $batch_size \
       --per_gpu_eval_batch_size $batch_size \
       --learning_rate $learning_rate \
       --num_train_epochs $num_train_epochs \
       --do_train \
       --do_eval \
       --do_test \
       --do_metrics \
       --evaluate_during_training \
       --best_metric_type $BEST_METRIC_TYPE \
       --logging_steps $logging_steps \
       --save_steps $save_steps \
       --gradient_accumulation_steps $gradient_accumulation_steps \
       --save_all \
       --start_eval_epoch 2 \
       --lr_update_steps 760000 \
       --dropout_prob 0.0 \
       --no_position_embeddings \
       --no_token_dropout \
       --scheduler_type step \
       --warmup_steps $warmup_steps   \
       --max_steps $max_steps \
       --beta1 0.9 \
       --beta2 0.98 \
       --weight_decay 0.01 \
       --no_use_embed_layer_norm \
       --loss_logging_steps $loss_logging_steps \
       --pretrained_model_name esm2_t36_3B_UR50D