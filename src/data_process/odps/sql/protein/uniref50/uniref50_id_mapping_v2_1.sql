--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-02 14:11:03
--********************************************************************--

CREATE FUNCTION seq_clean AS 'seq_clean_udf.seq_clean' USING 'seq_clean_udf.py' -f ;

SET odps.sql.python.version=cp37;


-- 获取uniref50的id
drop table if exists luca_data2.tmp_lucaone_v2_uniref50_fasta_info;
create table if not exists luca_data2.tmp_lucaone_v2_uniref50_fasta_info
as
select seq_id, replace(seq_id, ">UniRef50_", "") as uniprot_id, seq_clean(seq) as seq
from luca_data2.stg_lucaone_prot_uniref50;

-- uniprot中存在的id
drop table if exists luca_data2.tmp_lucaone_v2_uniprot_fasta_info;
create table if not exists luca_data2.tmp_lucaone_v2_uniprot_fasta_info
as
select seq_id, seq_clean(seq) as seq
from
    (
        select *
        from tmp_lucaone_v2_uniprot_sprot_fasta_info
        union
        select *
        from tmp_lucaone_v2_uniprot_trembl_fasta_info
    ) tmp;

-- 有多少id不在uniprot中出现, 9,640,132
select count(uniprot_id)
from
    luca_data2.tmp_lucaone_v2_uniref50_fasta_info t1
where uniprot_id not in (select seq_id from luca_data2.tmp_lucaone_v2_uniprot_fasta_info);

-- uniref与uniprot join
drop table if exists tmp_lucaone_v2_uniref50_join_uniprot_fasta_info;
create table if not exists tmp_lucaone_v2_uniref50_join_uniprot_fasta_info as
select t1.seq_id as seq_id,
       t1.uniprot_id as uniref50_id,
       t1.seq as uniref50_seq,
       t2.seq_id as uniprot_id,
       t2.seq as uniprot_seq,
       LENGTH(t1.seq) as uniref50_seq_len,
       LENGTH(t2.seq) as uniprot_seq_len,
       instr(t1.seq, t2.seq) as instr_pos
from
    luca_data2.tmp_lucaone_v2_uniref50_fasta_info t1
        left outer join
    tmp_lucaone_v2_uniprot_fasta_info t2
    on t1.uniprot_id = t2.seq_id;

-- 没有返回，也就是说, 能够seq_id join上的都在uniprot中, seq也等值
select *
from tmp_lucaone_v2_uniref50_join_uniprot_fasta_info
where instr_pos = 0 or uniref50_seq_len != uniprot_seq_len;


-- id不在uniprot出现的，有266个seq在uniprot中出现
select t1.*, t2.*
from
    (
        select uniprot_id, seq
        from
            luca_data2.tmp_lucaone_v2_uniref50_fasta_info t1
        where uniprot_id not in (select seq_id from luca_data2.tmp_lucaone_v2_uniprot_fasta_info)
    ) t1
        join
    luca_data2.tmp_lucaone_v2_uniprot_fasta_info t2
    on t1.seq = t2.seq;


-- 9,639,866 seq也不出现
select count(distinct uniprot_id)
from
    (
        select uniprot_id, seq
        from
            luca_data2.tmp_lucaone_v2_uniref50_fasta_info t1
        where uniprot_id not in (select seq_id from luca_data2.tmp_lucaone_v2_uniprot_fasta_info)
    ) t
where seq not in (select seq from luca_data2.tmp_lucaone_v2_uniprot_fasta_info);


-- uniref50中所有不在uniprot中的id
drop table if exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id;
create table if not exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id
as
select uniprot_id as seq_id
from
    luca_data2.tmp_lucaone_v2_uniref50_fasta_info t1
        left outer join
    luca_data2.tmp_lucaone_v2_uniprot_fasta_info t2
    on t1.uniprot_id = t2.seq_id
where t2.seq_id is null;

-- uniref50中与uniparc id等值的
drop table if exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id_equal;
create table if not exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id_equal
as
select *
from luca_data2.tmp_lucaone_v2_uniref50_uniparc_id
where seq_id like 'UPI%';

-- 9,628,938
SELECT count(1)
from tmp_lucaone_v2_uniref50_uniparc_id_equal;

-- uniref50中与去要去mapping uniparc id的
drop table if exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id_mapping;
create table if not exists luca_data2.tmp_lucaone_v2_uniref50_uniparc_id_mapping
as
select  *
from luca_data2.tmp_lucaone_v2_uniref50_uniparc_id
where seq_id not like 'UPI%';

-- 得到了mapping 11,194
select count(seq_id)
from luca_data2.tmp_lucaone_v2_uniref50_uniparc_id_mapping
where seq_id in (select uniprot_id from tmp_lucaone_v2_uniprot_uniparc_mapping );

###############
-- 开始
-- 情况1: seq_id相等
SET odps.sql.python.version=cp37;
drop table if exists luca_data2.tmp_lucaone_v2_uniprot_all_label_detail_v2;
create table if not exists luca_data2.tmp_lucaone_v2_uniprot_all_label_detail_v2
as
select *
from
    (
        select *
        from tmp_lucaone_v2_uniprot_sprot_label_detail_v2
        union
        select *
        from tmp_lucaone_v2_uniprot_trembl_label_detail_v2
    ) tmp;


DROP TABLE IF EXISTS tmp_lucaone_v2_uniref50_label_detail_c1_v2
;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c1_v2 AS
SELECT  t1.uniprot_id AS ref_id
     ,t2.*
FROM    tmp_lucaone_v2_uniref50_fasta_info t1
            JOIN    tmp_lucaone_v2_uniprot_all_label_detail_v2 t2
                    ON      t1.uniprot_id = t2.prot_seq_accession
;
-- 情况2: seq_id不相等，但seq相等
DROP TABLE IF EXISTS tmp_lucaone_v2_uniref50_label_detail_c2_v2
;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c2_v2 AS
SELECT  t1.uniprot_id AS ref_id
     ,t2.*
FROM    (
            SELECT  *
            FROM    tmp_lucaone_v2_uniref50_fasta_info
            WHERE   uniprot_id NOT IN (
                SELECT  DISTINCT ref_id
                FROM    tmp_lucaone_v2_uniref50_label_detail_c1_v2
                WHERE   ref_id IS NOT NULL
            )
        ) t1
            JOIN    tmp_lucaone_v2_uniprot_all_label_detail_v2 t2
                    ON      t1.seq = t2.seq
;
-- 情况3: seq_id在uniparc中
drop table if  exists tmp_lucaone_v2_uniref50_label_detail_c3_v2;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c3_v2 AS
select t1.uniprot_id as ref_id, t2.*
from (
         select *
         from tmp_lucaone_v2_uniref50_fasta_info
         where uniprot_id not in (
             select distinct ref_id
             from (
                      select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c1_v2
                      union ALL
                      select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c2_v2
                  )
             where ref_id is not null
         )
     ) t1
         join tmp_lucaone_v2_uniref_uniparc_label_detail_v2 t2
              on t1.uniprot_id = t2.prot_seq_accession;

-- 情况4: 映射得到的uniparc的id
drop table if  exists tmp_lucaone_v2_uniref50_label_detail_c4_v2;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c4_v2 AS
select uniprot_id as ref_id, t2.*
from  (
          select *
          from (
                   select seq_id, tt1.uniprot_id as uniprot_id, tt2.uniparc_id as uniparc_id, seq
                   from tmp_lucaone_v2_uniref50_fasta_info tt1
                            join tmp_lucaone_v2_uniprot_uniparc_mapping tt2
                                 on tt1.uniprot_id = tt2.uniprot_id
               ) tmp
          where uniprot_id not in (
              select distinct ref_id
              from (
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c1_v2
                       union ALL
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c2_v2
                       union ALL
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c3_v2
                   )
              where ref_id is not null
          )

      ) t1
          join tmp_lucaone_v2_uniref_uniparc_label_detail_v2 t2
               on t1.uniparc_id = t2.prot_seq_accession;


-- 情况5: uniparc id不在的，但是seq在的
drop table if  exists tmp_lucaone_v2_uniref50_label_detail_c5_v2;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c5_v2 AS
select uniprot_id as ref_id, t2.*
from  (
          select *
          from tmp_lucaone_v2_uniref50_fasta_info
          where uniprot_id not in (
              select distinct ref_id
              from (
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c1_v2
                       union ALL
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c2_v2
                       union ALL
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c3_v2
                       union ALL
                       select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c4_v2
                   )
              where ref_id is not null
          )

      ) t1
          join tmp_lucaone_v2_uniref_uniparc_label_detail_v2 t2
               on t1.seq = t2.seq;

-- 情况6: 其他
drop table if  exists tmp_lucaone_v2_uniref50_label_detail_c6_v2;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_c6_v2 AS
select seq_id as ref_id,
       seq_id as prot_seq_accession,
       seq_clean(seq) as seq,
       '' as taxid,
       '' order_bio,
       '' as keywords,
       '' as prot_feature_name,
       '' as prot_feature_type,
       -1 as start_p,
       -1 as end_p
from tmp_lucaone_v2_uniref50_fasta_info
where uniprot_id not in (
    select distinct ref_id
    from (
             select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c1_v2
             union ALL
             select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c2_v2
             union ALL
             select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c3_v2
             union ALL
             select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c4_v2
             union ALL
             select distinct ref_id from tmp_lucaone_v2_uniref50_label_detail_c5_v2
         )
    where ref_id is not null
) ;

-- 所有情况合并
drop table if exists tmp_lucaone_v2_uniref50_label_detail_all_v2;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref50_label_detail_all_v2 AS
select *
from (
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c1_v2
         union ALL
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c2_v2
         union ALL
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c3_v2
         union ALL
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c4_v2
         union ALL
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c5_v2
         union ALL
         select *
         from tmp_lucaone_v2_uniref50_label_detail_c6_v2
     ) t;