--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-13 14:51:53
--********************************************************************--
-- 738,695,581
select count(1) from stg_lucaone_prot_colabfold_envdb_202108_seq;

-- 209,335,865
select count(1) from stg_lucaone_prot_colabfold_envdb_202108;

-- 738,695,581
select count(1) from stg_lucaone_prot_colabfold_envdb_202108_h;


-- colabfold_envdb fasta
-- 209,335,865
drop table if exists tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster;
create table if not exists tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster
AS
select t2.seq_id as seq_id, seq
from stg_lucaone_prot_colabfold_envdb_202108 t1
         join stg_lucaone_prot_colabfold_envdb_202108_h t2
              on t1.seq_id = t2.seq_index;


-- 每条序列的来源
SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_colabfold_envdb_db_from_cluster;
create table if not exists tmp_lucaone_v2_colabfold_envdb_db_from_cluster AS
select seq_id, seq_clean(seq) as seq,
       case
           when seq_id like 'tr|%' then SPLIT_PART(seq_id, '|', 1)
           when seq_id like 'sp|%' then SPLIT_PART(seq_id, '|', 1)
           when seq_id like '%.scaffolds.%' then "scaffolds"
           when seq_id like '%.fasta_scaffold%' then "fasta_scaffold"
           when seq_id like '%|scaffold%' then "scaffold"
           when seq_id like '%.fasta_contig%' then "fasta_contig"
           when seq_id like '%.fsa_nt_%' then "fsa_nt"
           when seq_id like '%FD_contig%' then "FD_contig"
           when seq_id like '%FD_k123%' then "FD_k123"
           when seq_id like '266_contig%' then "266_contig"
           when seq_id like '%OM-RGC%' then "OM-RGC"
           when seq_id like '%fsa_nt_gb%' then "gb"
           when seq_id like '%fsa_nt_gi%' then "gi"
           when seq_id like '%fsa_nt_emb%' then "emb"
           when seq_id like 'SRR%' or seq_id like '%|SRR%' then "SRR"
           when seq_id like 'ERR%' or seq_id like '%|ERR%' then "ERR"
           when instr(seq_id, '|gb|') > 0 then 'gb'
           when seq_id like '%|Ga%' then "Ga"
           when seq_id like 'UniRef%' then "UniRef"
           when seq_id like 'MGY%' then "MGY"
           when seq_id like 'UPI%' then "Uniparc"
           when seq_id like '%:Ga%' then "Ga"
           when seq_id like '%|NODE%' then "NODE"
           when seq_id like 'uvig_%' then "uvid"
           when seq_id like 'ivig_%' then "ivig"
           when seq_id like 'TARA_%' then "TARA"
           when seq_id like '%|JGI%' then "JGI"
           when seq_id like '%:JGI%' then "JGI"
           when seq_id like 'MGV-%' then "MGV"
           when seq_id like '%|contig_%' then 'contig'
           when seq_id like '%_contig_%' then 'contig'
           else null
           end as db,
       case
           when seq_id like 'tr|%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like 'sp|%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%.scaffolds.%' then SPLIT_PART(seq_id, '.', 3)
           when seq_id like '%.fasta_scaffold%' then SPLIT_PART(seq_id, '.', 2)
           when seq_id like '%.fasta_contig%' then SPLIT_PART(SPLIT_PART(seq_id, '.', 3), '_length', 1)
           when seq_id like '%|scaffold%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%.fsa_nt_%' then SPLIT_PART(seq_id, '.', 3)
           when seq_id like '%FD_contig%' then SPLIT_PART(seq_id, '_length', 1)
           when seq_id like '%FD_k123%' then seq_id
           when seq_id like '266_contig%' then SPLIT_PART(seq_id, '_length', 1)
           when seq_id like '%OM-RGC%' then seq_id
           when seq_id like '%fsa_nt_gb%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%fsa_nt_gi%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%fsa_nt_emb%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like 'ERR%' then seq_id
           when seq_id like '%|ERR%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like 'SRR%' then seq_id
           when seq_id like '%|SRR%' then SPLIT_PART(seq_id, '|', 2)
           when instr(seq_id, '|gb|') > 0 then SPLIT_PART(seq_id, '|', 4)
           when seq_id like '%|Ga%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like 'UniRef%' then SPLIT_PART(SPLIT_PART(seq_id, ' ', 1), '_', 2)
           when seq_id like 'MGY%' then SPLIT_PART(seq_id, ' ', 1)
           when seq_id like 'UPI%' then seq_id
           when seq_id like '%:Ga%' then SPLIT_PART(SPLIT_PART(seq_id, ' ', 1), ':', 2)
           when seq_id like '%|NODE%' then SPLIT_PART(SPLIT_PART(seq_id, '_length', 1), '|', 2)
           when seq_id like 'uvig_%' then seq_id
           when seq_id like 'ivig_%' then seq_id
           when seq_id like 'TARA_%' then seq_id
           when seq_id like '%|JGI%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%:JGI%' then SPLIT_PART(seq_id, ':', 2)
           when seq_id like 'MGV-%' then SPLIT_PART(seq_id, ' ', 1)
           when seq_id like '%|contig_%' then SPLIT_PART(seq_id, '|', 2)
           when seq_id like '%_contig_%' then SPLIT_PART(seq_id, '|', 2)
           else null end as db_seq_id,
       case when seq_id like '%UniRef%' then SPLIT_PART(SPLIT_PART(seq_id, "TaxID=", 2), " ", 1) else null end as taxid,
       case when seq_id like '%UniRef%' then SPLIT_PART(SPLIT_PART(seq_id, "RepID=", 2), " ", 1) else null end as repid
from tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster;
