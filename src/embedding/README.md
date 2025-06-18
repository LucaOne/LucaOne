# LLM for embedding     

### LucaOne(LucaGPLM) Embedding     

**建议与说明:**         
1）尽量使用显存大进行embedding 推理，如：A100，H100，H200等，这样一次性能够处理较长的序列，LucaOne在A100下可以一次性处理`2800`左右长度的序列；   
2）对于超长序列，LucaOne会进行Overlap分片进行embedding，最后合并成完整的embedding，请设置`--embedding_complete`与`--embedding_complete_seg_overlap`；    
3）如果显卡不足以处理输入的序列长度，会调用CPU进行处理，这样速度会变慢，如果你的数据集中长序列不是很多，那么可以使用这种方式: `--gpu_id -1`；      
4）如果你的数据集中长序列很多，比如: 万条以上，那么再设置`--embedding_complete`与`--embedding_complete_seg_overlap`之外，再加上设置`--embedding_fixed_len_a_time`，表示一次性embedding的最大长度。
如果序列长度大于这个长度，基于这个长度进行分片embedding，最后进行合并。否则根据序列的实际长度；    
5）如果不设置`--embedding_complete`，那么根据设置的`--truncation_seq_length`的值对序列进行截断embedding；  
6）对于蛋白，因为绝大部分蛋白长度在1000以下，因此超长蛋白序列不会很多，因此可以将`--embedding_fixed_len_a_time`设置长一点或者`不设置`；    
7）对于DNA，因为很多任务的DNA序列很长，那么请设置`--embedding_fixed_len_a_time`。    
超长序列数据量越多，该值设置越小一点，比如在A100下设置为`2800`，否则设置大一点，如果GPU根据这个长度embedding失败，则会调用CPU。如果数据集数不大，则时间不会很久；          
8）对于RNA，因为大部分RNA不会很长，因此与蛋白处理方式一致，因此可以将`--embedding_fixed_len_a_time`设置长一点或者不设置；

**Suggestions and Instructions:**
1) Try to use a large GPU-memory machine for embedding reasoning, such as A100, H100, H200, etc., so that long sequences can be processed once.       
   LucaOne can process sequences of about `2800` in length at one time under A100;
2) For long sequences, LucaOne will do overlapped fragments in the sequence for embedding and finally merge them into a completed embedding matrix.        
   Please set `--embedding_complete` and `--embedding_complete_seg_overlap`;
3) If the GPU memory is not enough to process the longer sequence, it will use the CPU for embedding, so the speed will be reduced.       
   If your dataset is small, then you can set: `--gpu_id -1`;
4) If your dataset includes a lot of long sequences (more than 10,000 sequences), please set: `--embedding_complete`, `--embedding_complete_seg_overlap`, and `--embedding_fixed_len_a_time` (represent the maximum length for embedding at one-time).       
   If the sequence length is greater than the value of `--embedding_fixed_len_a_time`, fragment embedding is performed based on this value, and finally, the merge is performed; otherwise, according to the actual length of the sequence;
5) If `--embedding_complete` is not set, the code will truncate the sequence embedding according to the value of `--truncation_seq_length`;
6) For proteins, the length of most proteins is less than 1000; there are not many ultra-long protein sequences, so the value of `--embedding_fixed_len_a_time` can be set a large value or not be set;
7) For DNA, the DNA sequence of many tasks is very long; please set `--embedding_fixed_len_a_time`.  
   The larger the amount of ultra-long sequence, the smaller value should be set, such as `2800` under A100.      
   If the GPU embedding fails to process the longer sequence, the CPU will be called.      
   When the amount of dataset is not large, the spent time will not be long;
8) For RNA, most RNA is not very long, so the processing method can be consistent with the protein, so the `--embedding_fixed_len_a_time` can be set a larger value or not be set.


for `gene` or `prot`
```

# for DNA or RNA
cd ./src/embedding/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../ \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/test_data/gene/test_gene.fasta \
    --save_path ../../../embedding/lucaone/test_data/gene/test_gene \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
    
# for Protein
cd ./src/llm/lucaone_gplm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../ \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/test_data/prot/test_prot.fasta \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


```