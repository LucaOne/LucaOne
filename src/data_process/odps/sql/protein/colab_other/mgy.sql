--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-18 20:57:34
--********************************************************************--

-- 物种
-- SPLIT_PART(prot_id， "_", 1) in (select distinct genome from tmp_lucaone_v2_prot_taxonomy_info_mgnify where genome is null);
select count(distinct genome)
from  tmp_lucaone_v2_prot_taxonomy_info_mgnify;

-- feature, 492,354,203
select count(DISTINCT prot_id)
from  tmp_lucaone_v2_prot_fea_info_mgnify;

-- fasta信息,  只有26,600,681个，不够
select count(distinct seq_id)
from  tmp_lucaone_v2_prot_fasta_info_mgnify;

-- 更多的fatsta, 122,054,222
SET odps.sql.python.version=cp37;

-- 3,695,952
SET odps.sql.python.version=cp37;
select count(1)
from stg_lucaone_prot_colabfold_envdb_202108_seq
where seq_clean(seq) in (select seq_clean(seq) from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100);

-- 1,568,263
SET odps.sql.python.version=cp37;
select count(1)
from stg_lucaone_prot_colabfold_envdb_202108_seq
where seq_clean(seq) in (select seq_clean(seq) from tmp_lucaone_v2_prot_fasta_info_mgnify);

select *
from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100
         limit 10;

-- 106,103,527
select count(distinct prot_id)
from tmp_lucaone_v2_prot_fea_info_mgnify
where prot_id in (select seq_id from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100);

-- 1,863,441
SET odps.sql.python.version=cp37;
select count(1)
from (select seq_clean(seq) as seq from tmp_lucaone_v2_colabfold_envdb_db_from where db = "MGY")
where seq in (select seq_clean(seq) from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100);

-- 26,021,215
SET odps.sql.python.version=cp37;
select count(1)
from tmp_lucaone_v2_prot_fasta_info_mgnify
where seq_clean(seq) in (select seq_clean(seq) from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100);

-- 确实没有
SET odps.sql.python.version=cp37;
select *
from tmp_lucaone_v2_prot_fasta_info_mgnify
where seq_clean(seq) not in (select seq_clean(seq) from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100);
select *
from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100
where seq_id = 'MGYG000000193_02368';

-- seq id相同 seq不同
SET odps.sql.python.version=cp37;
create table if not exists tmp_lucaone_v2_prot_fasta_info_verify_mgnify as
select t1.seq_id as seq_id, t1.seq as seq_1, t2.seq as seq_2
from (
         select t2.seq_id as seq_id, seq_clean(t2.seq) as seq
         from tmp_lucaone_v2_prot_fea_info_mgnify t1
                  join tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100 t2
                       on t1.prot_id = t2.seq_id
     ) t1
         join (select seq_id, seq_clean(seq) as seq from tmp_lucaone_v2_prot_fasta_info_mgnify) t2
              on t1.seq_id = t2.seq_id;

-- 确实没有
select *
from tmp_lucaone_v2_prot_fasta_info_verify_mgnify
where seq_1 != seq_2;

-- tax信息
SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_mgnify_taxonomy_info;
create table if not exists tmp_lucaone_v2_mgnify_taxonomy_info as
select seq_id, seq, order_bio
from
    (
        select distinct seq_id, seq, genome
        from(
                select seq_id, seq_clean(seq) as seq, SPLIT_PART(seq_id, "_", 1) as genome
                from tmp_lucaone_v2_prot_fasta_info_mgnify_protein_catalogue_identity_100
                union all
                select seq_id, seq_clean(seq) as seq, SPLIT_PART(seq_id, "_", 1) as genome
                from tmp_lucaone_v2_prot_fasta_info_mgnify
            ) t
    ) t1
        join
    (
        select distinct genome, order as order_bio
        from tmp_lucaone_v2_prot_taxonomy_info_mgnify
    ) t2
    on t1.genome = t2.genome;





