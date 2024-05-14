--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 16:33:48
-- GPD 只能获取物种信息
--********************************************************************--
-- query_name
-- taxonomy 是 class级别， 3982970
SELECT  COUNT(1)
FROM   luca_data2.tmp_gpd_proteome_orthology_assignment
;

SELECT  *
FROM    luca_data2.tmp_gpd_proteome_orthology_assignment
            LIMIT   10
;

-- 7474086
SELECT  COUNT(1)
FROM    luca_data2.tmp_gpd_pc_info
;

-- 7581807
SELECT  COUNT(1)
FROM    luca_data2.tmp_gpd_proteins
;


-- 0, 证实是class级别
SELECT  COUNT(DISTINCT best_tax_level)
FROM    luca_data2.tmp_gpd_proteome_orthology_assignment
WHERE   best_tax_level NOT IN (
    SELECT  DISTINCT class
    FROM    tmp_lucaone_v2_taxid_mapping
)
;


SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_gpd_taxonomy_info;
create table if not exists tmp_lucaone_v2_gpd_taxonomy_info as
select seq_id, seq, class
from
    (
        select seq_id, seq_clean(seq) as seq
        from tmp_gpd_proteins
    ) t1
        left join
    (
        select distinct query_name, best_tax_level as class
        from tmp_gpd_proteome_orthology_assignment
    ) t2
    on t1.seq_id = t2.query_name;


select *
from tmp_lucaone_v2_gpd_taxonomy_info
         limit 10;

