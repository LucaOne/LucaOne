--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 16:34:13
--********************************************************************--
-- 189680
--
SELECT *
FROM    tmp_mgv_contig_info
;

-- 11837198
SELECT  *
FROM    tmp_mgv_pc_info
;

-- 95164
SELECT  *
FROM    tmp_mgv_pc_functions
;

-- 11837198
-- join  seq_id.split("_")[0] == tmp_mgv_contig_info.contig_id
-- ictv_order
SELECT  *
FROM  tmp_mgv_proteins
;

select *
from  luca_data2.tmp_mgv_pc_functions;



SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_mgv_taxonomy_info;
create table if not exists tmp_lucaone_v2_mgv_taxonomy_info as
select seq_id, seq_clean(seq) as seq, ictv_order as order_bio
from
    (
        select *, split_part(seq_id, "_", 1) as contig_id
        from tmp_mgv_proteins
    ) t1
        left join
    (
        select distinct contig_id, ictv_order
        from tmp_mgv_contig_info
    ) t2
    on t1.contig_id = t2.contig_id;


