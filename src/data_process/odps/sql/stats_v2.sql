--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2024-04-08 13:38:36
--********************************************************************--
show tables;

-- select * from tmp_lucaone_v2_taxid_mapping limit 10;

-- lucaone_data.tmp_lucaone_v2_data_gene2_01: 136311178 (46409958163 bytes)
-- lucaone_data.tmp_lucaone_v2_data_gene2_02: 1181133873 (396835450638 bytes)
-- lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2: 1317445051 (428391528983 bytes)

create table if not exists lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2 as
select *
from
    (
        select *
        from lucaone_data.tmp_lucaone_v2_data_gene2_01
        union ALL
        select *
        from lucaone_data.tmp_lucaone_v2_data_gene2_02
    ) t;

-- lucaone_data.tmp_lucaone_v2_data_prot_01: 62150523 (12173480301 bytes)
-- lucaone_data.tmp_lucaone_v2_data_prot_02: 252170925 (59002595454 bytes)
-- lucaone_data.tmp_lucaone_v2_data_prot_03: 208966064 (21432104675 bytes)
-- ucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2: 523287512 (90585481507 bytes)


create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2 as
select *
from
    (
        select *
        from lucaone_data.tmp_lucaone_v2_data_prot_01
        union all
        select *
        from lucaone_data.tmp_lucaone_v2_data_prot_02
        union all
        select *
        from lucaone_data.tmp_lucaone_v2_data_prot_03
    ) t;

SET odps.sql.python.version=cp37;
add table tmp_lucaone_v2_taxid_mapping as lucaone_v2_taxid_mapping_resource -f;
CREATE FUNCTION data_stats_v2 AS 'data_stats_v2.data_stats_v2' USING 'data_stats_v2.py,lucaone_v2_taxid_mapping_resource,gene_type_span_level_label_v2.txt,gene_taxonomy_seq_level_label_v2.txt,prot_homo_span_level_label_v2.txt,prot_site_span_level_label_v2.txt,prot_domain_span_level_label_v2.txt,prot_taxonomy_seq_level_label_v2.txt,prot_keyword_seq_level_label_v2.txt' -f ;


-- gene tax-superkingdom
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_superkingdom;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_superkingdom as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "superkingdom") as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene tax-phylum
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_phylum;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_phylum as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "phylum") as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene tax-class
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_class_name;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_class_name as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "class_name") as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene seq
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_seq;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_seq as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "seq", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene token
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_token;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_token as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "token", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene order
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_taxonomy;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_taxonomy as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "taxonomy", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- gene gene_type
drop table if exists lucaone_v2_data_gene_lucaone_v2_stats_gene_type;
create table if not exists lucaone_v2_data_gene_lucaone_v2_stats_gene_type as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "gene_type", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_gene_lucaone_v2
group by 1;

-- prot tax-superkingdom
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_superkingdom;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_superkingdom as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "superkingdom") as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;

-- prot tax-phylum
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_phylum;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_phylum as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "phylum") as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;

-- prot tax-class
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_class_name;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_class_name as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "class_name") as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;

-- prot seq
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_seq;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_seq as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "seq", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;

-- prot token
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_token;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_token as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "token", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;


-- prot  order
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_taxonomy;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_taxonomy as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "taxonomy", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;

-- prot keyword
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_keyword;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_keyword as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "keyword", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;


-- prot site
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_site;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_site as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "site", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;


-- prot homo
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_homo;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_homo as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "homo", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;


-- prot domain
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_domain;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_domain as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "domain", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;



-- prot structure
drop table if exists lucaone_v2_data_prot_lucaone_v2_stats_structure;
create table if not exists lucaone_v2_data_prot_lucaone_v2_stats_structure as
select data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "structure", NULL) as res
from lucaone_data.tmp_lucaone_v2_data_prot_lucaone_v2
group by 1;


