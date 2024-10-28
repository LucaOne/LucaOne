# LLM for embedding     

### LucaOne(LucaGPLM) Embedding   
for `gene` or `prot`
```
cd ./src/llm/lucaone_gplm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/test_data/prot/test_prot.fasta \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0

# for DNA or RNA
cd ./src/llm/lucagplm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/test_data/gene/test_gene.fasta \
    --save_path ../../../embedding/lucaone/test_data/gene/test_gene \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
```