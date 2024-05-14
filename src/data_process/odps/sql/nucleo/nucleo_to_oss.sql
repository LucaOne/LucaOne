--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-24 19:34:39
--********************************************************************--
-- ori_data
DROP TABLE IF EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_01
;

CREATE TABLE IF NOT EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_01 AS
SELECT  *
FROM    luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_${max_segment_len}_final
;

--ori_data
DROP TABLE IF EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_02
;

CREATE TABLE IF NOT EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_02 AS
SELECT  *
FROM    luca_data2.tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_final
;

-- data
DROP TABLE IF EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_01_external
;

CREATE EXTERNAL TABLE IF NOT EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_01_external
(
    obj_id     STRING
    ,obj_type  STRING
    ,obj_seq   STRING
    ,obj_label STRING
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES ('odps.text.option.use.quote' = 'true','delimiter' = ',','quoteChar' = '"','odps.text.option.header.lines.count' = '1','odps.sql.text.option.flush.header' = 'true')
LOCATION 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_data/gene2/rna/'
;

--
INSERT OVERWRITE TABLE lucaone_data.tmp_lucaone_v2_data_gene2_01_external
SELECT  *
FROM    lucaone_data.tmp_lucaone_v2_data_gene2_01
;

--
DROP TABLE IF EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_02_external
;

CREATE EXTERNAL TABLE IF NOT EXISTS lucaone_data.tmp_lucaone_v2_data_gene2_02_external
(
    obj_id     STRING
    ,obj_type  STRING
    ,obj_seq   STRING
    ,obj_label STRING
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES ('odps.text.option.use.quote' = 'true','delimiter' = ',','quoteChar' = '"','odps.text.option.header.lines.count' = '1','odps.sql.text.option.flush.header' = 'true')
LOCATION 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_data/gene2/genomic/'
;

--
INSERT OVERWRITE TABLE lucaone_data.tmp_lucaone_v2_data_gene2_02_external
SELECT  *
FROM    lucaone_data.tmp_lucaone_v2_data_gene2_02
;



-- dataset

--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-23 15:11:22
--********************************************************************--
--rna
set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_dev;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_dev
as
select *, "rna" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene2_01
    ) tmp
order by rand_v
    limit 10000;


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_test;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_test
as
select *, "rna" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene2_01
        where obj_id not in (
            select obj_id
            from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_dev
            where obj_id is not null
        )
    ) tmp
order by rand_v
    limit 10000;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_train;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_train
as
select *, "rna" as obj_source
from lucaone_data.tmp_lucaone_v2_data_gene2_01
where obj_id not in (
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_dev
    where obj_id is not null
    union all
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_test
    where obj_id is not null
);

-- genomic
set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_dev;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_dev
as
select *, "genomic" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene2_02
    ) tmp
order by rand_v
    limit 10000;


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_test;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_test
as
select *, "genomic" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene2_02
        where obj_id not in (
            select obj_id
            from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_dev
            where obj_id is not null
        )
    ) tmp
order by rand_v
    limit 10000;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_train;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_train
as
select *, "genomic" as obj_source
from lucaone_data.tmp_lucaone_v2_data_gene2_02
where obj_id not in (
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_dev
    where obj_id is not null
    union all
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_test
    where obj_id is not null
);


-- odps->oss
-- train
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_train_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_train_external
(
    obj_id string,
    obj_type string,
    obj_seq string,
    obj_label string,
    obj_source string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/train/gene2/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_gene2_train_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_train
        union all
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_train
    ) tmp;

-- dev
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_dev_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_dev_external
(
    obj_id string,
    obj_type string,
    obj_seq string,
    obj_label string,
    obj_source string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/dev/gene2/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_gene2_dev_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_dev
        union all
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_dev
    ) tmp;

-- test
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene2_test_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_gene2_test_external
(
    obj_id string,
    obj_type string,
    obj_seq string,
    obj_label string,
    obj_source string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/test/gene2/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_gene2_test_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_rna_rand_test
        union all
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_gene2_genomic_rand_test
    ) tmp;

