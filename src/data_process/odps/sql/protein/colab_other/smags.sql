--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 16:34:59
--********************************************************************--
-- 4,485,410
SELECT  *
FROM    tmp_SMAGs_v1_EggNog
            limit 1000
;

-- 10,207,435
SELECT  *
FROM    tmp_SMAGs_proteins
            limit 1000
;


-- 只有界粒度，join query_name == seq_id


SELECT distinct predicted_taxonomic_group
FROM tmp_SMAGs_v1_EggNog;

SELECT distinct tax_scope
FROM tmp_SMAGs_v1_EggNog;

SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_SMAGs_taxonomy_info;
create table if not exists tmp_lucaone_v2_SMAGs_taxonomy_info as
select seq_id, seq_clean(seq) as seq, predicted_taxonomic_group, tax_scope
from
    tmp_SMAGs_proteins t1
        left join
    tmp_SMAGs_v1_EggNog t2
    on t1.seq_id = t2.query_name;