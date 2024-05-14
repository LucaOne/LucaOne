--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-13 14:51:53
--********************************************************************--
-- seq_id与seq都来源于uniprot，有labels
-- seq_id与seq都来源于uniparc, 没有keywords
-- seq_id与seq都来源于uniref100，只有taxid。
-- seq_id与seq都来源于colab_other，只有order_bio
-- 其他，只有序列
-- 情况1: seq_id相等

SET odps.sql.python.version=cp37;
DROP TABLE IF EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster AS
SELECT  t1.seq_id AS ref_id
     ,t2.*
FROM    tmp_lucaone_v2_colabfold_envdb_db_from_cluster t1
            JOIN    tmp_lucaone_v2_uniprot_all_label_detail_v2 t2
                    ON      t1.db_seq_id = t2.prot_seq_accession
;

-- 情况2: seq_id不相等，但seq相等
DROP TABLE IF EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster AS
SELECT  t1.seq_id AS ref_id
     ,t2.*
FROM    (
            SELECT  *
            FROM    tmp_lucaone_v2_colabfold_envdb_db_from_cluster
            WHERE   seq_id NOT IN (
                SELECT  DISTINCT ref_id
                FROM    tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                WHERE   ref_id IS NOT NULL
            )
        ) t1
            JOIN    tmp_lucaone_v2_uniprot_all_label_detail_v2 t2
                    ON      t1.seq = t2.seq
;
-- 情况3: seq_id在uniparc中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2 t2
              on t1.db_seq_id = t2.prot_seq_accession;


-- 情况4: seq在uniparc中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2 t2
              on t1.seq = t2.seq;


-- 情况5: seq_id在uniref中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colab_uniref_label_detail_v2_cluster t2
              on t1.seq_id = t2.prot_seq_accession;


-- 情况6: seq在uniref中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colab_uniref_label_detail_v2_cluster t2
              on t1.seq = t2.seq;


-- 情况7: seq_id在colab_other中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colab_other_label_detail_v2_cluster t2
              on t1.db_seq_id = t2.prot_seq_accession;


-- 情况8: seq在uniref中
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_c8_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c8_v2_cluster AS
select t1.seq_id AS ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
         where seq_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_colab_other_label_detail_v2_cluster t2
              on t1.seq = t2.seq;

-- 情况9: 其他
drop table if  exists tmp_lucaone_v2_colabfold_envdb_label_detail_c9_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_c9_v2_cluster AS
select seq_id as ref_id,
       case when db_seq_id is not null then db_seq_id
            else md5(seq_id)
           end as prot_seq_accession,
       seq_clean(seq) as seq,
       '' as taxid,
       '' order_bio,
       '' as keywords,
       '' as prot_feature_name,
       '' as prot_feature_type,
       cast(-1 as bigint) as start_p,
       cast(-1 as bigint) as end_p
from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
where seq_id not in (
    select distinct ref_id
    from (
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster
             union ALL
             select distinct ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_c8_v2_cluster
         )
    where ref_id is not null
) ;

-- 所有情况合并
drop table if exists tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster AS
select *
from (
         select *, "c1" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster
         union ALL
         select *, "c2" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster
         union ALL
         select *, "c3" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster
         union ALL
         select *, "c4" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster
         union ALL
         select *, "c5" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster
         union ALL
         select *, "c6" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster
         union ALL
         select *, "c7" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster
         union ALL
         select *, "c8" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c8_v2_cluster
         union ALL
         select *, "c9" as source
         from tmp_lucaone_v2_colabfold_envdb_label_detail_c9_v2_cluster
     ) t;

inputs:
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c1_v2_cluster: 9024373 (2953000942 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c2_v2_cluster: 429220 (44037752 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c3_v2_cluster: 1037777 (128385587 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c4_v2_cluster: 5901 (998603 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c5_v2_cluster: 0 (0 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c6_v2_cluster: 57787 (16605844 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c7_v2_cluster: 482168 (69973190 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c8_v2_cluster: 587544 (46342973 bytes)
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_c9_v2_cluster: 203335245 (21515625136 bytes)
outputs:
        luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster: 214960015 (24384473549 bytes)