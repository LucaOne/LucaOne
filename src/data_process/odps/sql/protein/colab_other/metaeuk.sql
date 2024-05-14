--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 16:34:36
--********************************************************************--
--12111301
SELECT  *
FROM    luca_data2.tmp_MetaEuk_preds_Tara_vs_euk_profiles_proteins
;

-- 6158526
SELECT  *
FROM    luca_data2.tmp_MetaEuk_preds_Tara_vs_euk_profiles_uniqs_proteins
;

-- 12111301
SELECT  *
FROM    luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
;

SELECT  *
FROM    tmp_lucaone_v2_taxid_mapping
where species = 'Clostridium argentinense';

-- lineage_info要解析
SELECT  distinct tax_level, lineage_info
FROM    luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles;


SELECT  distinct size(SPLIT(lineage_info, ":"))
FROM    luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
;

drop table if exists tmp_MetaEuk_tax_level10;
create table if not exists tmp_MetaEuk_tax_level10
as
SELECT  REPLACE(SPLIT_PART(lineage_info, ":", 1), "_", " ") as level1,
        REPLACE(SPLIT_PART(lineage_info, ":", 2), "_", " ")  as level2,
        REPLACE(SPLIT_PART(lineage_info, ":", 3), "_", " ")  as level3,
        REPLACE(SPLIT_PART(lineage_info, ":", 4), "_", " ") as level4,
        REPLACE(SPLIT_PART(lineage_info, ":", 5), "_", " ")  as level5,
        REPLACE(SPLIT_PART(lineage_info, ":", 6), "_", " ") as level6,
        REPLACE(SPLIT_PART(lineage_info, ":", 7), "_", " ") as level7,
        REPLACE(SPLIT_PART(lineage_info, ":", 8), "_", " ")  as level8,
        REPLACE(SPLIT_PART(lineage_info, ":", 9), "_", " ") as level9,
        REPLACE(SPLIT_PART(lineage_info, ":", 10), "_", " ")  as level10
FROM    luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
where  size(SPLIT(lineage_info, ":")) = 10;

drop table if exists tmp_MetaEuk_tax_level11;
create table if not exists tmp_MetaEuk_tax_level11
as
SELECT  concat(REPLACE(SPLIT_PART(lineage_info, ":", 1), "_", " "),
               REPLACE(SPLIT_PART(lineage_info, ":", 2), "_", " ")) as level1,
        REPLACE(SPLIT_PART(lineage_info, ":", 3), "_", " ")  as level2,
        REPLACE(SPLIT_PART(lineage_info, ":", 4), "_", " ")  as level3,
        REPLACE(SPLIT_PART(lineage_info, ":", 5), "_", " ") as level4,
        REPLACE(SPLIT_PART(lineage_info, ":", 6), "_", " ")  as level5,
        REPLACE(SPLIT_PART(lineage_info, ":", 7), "_", " ") as level6,
        REPLACE(SPLIT_PART(lineage_info, ":", 8), "_", " ") as level7,
        REPLACE(SPLIT_PART(lineage_info, ":", 9), "_", " ")  as level8,
        REPLACE(SPLIT_PART(lineage_info, ":", 10), "_", " ") as level9,
        REPLACE(SPLIT_PART(lineage_info, ":", 11), "_", " ")  as level10
FROM   luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
where  size(SPLIT(lineage_info, ":")) = 11;

select *
FROM   luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
where  size(SPLIT(lineage_info, ":")) is null;

-- 第一层为species，第二层为属 genus，第三层为 科 family， 第四层为 目 order
-- 第五层为 纲 class 第六层为 门 phylum 第七层为 界 superkingdom
select distinct level5
from tmp_MetaEuk_tax_level11
where level5  in (
    select distinct order from tmp_lucaone_v2_taxid_mapping
where order is not null);

select *
FROM   luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
where  size(SPLIT(lineage_info, ":")) = 11;

drop table if exists tmp_lucaone_v2_taxonomy_tree_detail;
create table if not exists tmp_lucaone_v2_taxonomy_tree_detail
as
select *,
       REPLACE(SPLIT_PART(lineage, ",", 2), "_", " ") as superkingdom,
       REPLACE(SPLIT_PART(lineage, ",", 3), "_", " ")  as phylum,
       REPLACE(SPLIT_PART(lineage, ",", 4), "_", " ")  as class,
       REPLACE(SPLIT_PART(lineage, ",", 5), "_", " ") as order_bio,
       REPLACE(SPLIT_PART(lineage, ",", 6), "_", " ")  as family,
       REPLACE(SPLIT_PART(lineage, ",", 7), "_", " ") as genus,
       REPLACE(SPLIT_PART(lineage, ",", 8), "_", " ") as species
from tmp_lucaone_v2_taxonomy_tree
where rank = 'species';

select *
from tmp_lucaone_v2_taxonomy_tree_detail;

select distinct taxid
from tmp_lucaone_v2_taxid_mapping
where taxid not in (
    select distinct taxid from tmp_lucaone_v2_taxonomy_tree where taxid is not null);



create table if not exists tmp_MetaEuk_tax_info
as
select *
from(
        SELECT  *,
                REPLACE(SPLIT_PART(lineage_info, ":", 1), "_", " ") as level1,
                REPLACE(SPLIT_PART(lineage_info, ":", 2), "_", " ")  as level2,
                REPLACE(SPLIT_PART(lineage_info, ":", 3), "_", " ")  as level3,
                REPLACE(SPLIT_PART(lineage_info, ":", 4), "_", " ") as level4,
                REPLACE(SPLIT_PART(lineage_info, ":", 5), "_", " ")  as level5,
                REPLACE(SPLIT_PART(lineage_info, ":", 6), "_", " ") as level6,
                REPLACE(SPLIT_PART(lineage_info, ":", 7), "_", " ") as level7,
                REPLACE(SPLIT_PART(lineage_info, ":", 8), "_", " ")  as level8,
                REPLACE(SPLIT_PART(lineage_info, ":", 9), "_", " ") as level9,
                REPLACE(SPLIT_PART(lineage_info, ":", 10), "_", " ")  as level10
        FROM    luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
        where  size(SPLIT(lineage_info, ":")) = 10
        union all
        SELECT  *, concat(REPLACE(SPLIT_PART(lineage_info, ":", 1), "_", " "),
            REPLACE(SPLIT_PART(lineage_info, ":", 2), "_", " ")) as level1,
            REPLACE(SPLIT_PART(lineage_info, ":", 3), "_", " ")  as level2,
            REPLACE(SPLIT_PART(lineage_info, ":", 4), "_", " ")  as level3,
            REPLACE(SPLIT_PART(lineage_info, ":", 5), "_", " ") as level4,
            REPLACE(SPLIT_PART(lineage_info, ":", 6), "_", " ")  as level5,
            REPLACE(SPLIT_PART(lineage_info, ":", 7), "_", " ") as level6,
            REPLACE(SPLIT_PART(lineage_info, ":", 8), "_", " ") as level7,
            REPLACE(SPLIT_PART(lineage_info, ":", 9), "_", " ")  as level8,
            REPLACE(SPLIT_PART(lineage_info, ":", 10), "_", " ") as level9,
            REPLACE(SPLIT_PART(lineage_info, ":", 11), "_", " ")  as level10
        FROM   luca_data2.tmp_Tax_assignment_MetaEuk_preds_Tara_vs_euk_profiles
        where  size(SPLIT(lineage_info, ":")) = 11
    ) t;


SET odps.sql.python.version=cp37;
drop table if exists tmp_lucaone_v2_MetaEuk_taxonomy_info;
create table if not exists tmp_lucaone_v2_MetaEuk_taxonomy_info as
select seq_id, seq_clean(seq) as seq, level4 as order_bio
from
    tmp_MetaEuk_preds_Tara_vs_euk_profiles_proteins t1
        left join
    tmp_MetaEuk_tax_info t2
    on t1.header = t2.header;
