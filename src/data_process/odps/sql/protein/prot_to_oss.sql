--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-23 21:34:14
--********************************************************************--
-- data_ori
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_01;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_01 as
select *
from luca_data2.tmp_lucaone_v2_uniref50_all_detail_v2_final;

drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_02;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_02 as
select *
from luca_data2.tmp_lucaone_v2_uniprot_all_detail_v2_final;

drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_03;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_03 as
select *
from luca_data2.tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster_final;

-- data
-- uniref50(done) 62,150,523
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_01_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_01_external
(
obj_id string,
obj_type string,
obj_seq string,
obj_label string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_data/prot/uniref50/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_01_external
select *
from lucaone_data.tmp_lucaone_v2_data_prot_01;

-- uniprot(done) 252,170,925
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_02_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_02_external
(
obj_id string,
obj_type string,
obj_seq string,
obj_label string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_data/prot/uniprot/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_02_external
select *
from lucaone_data.tmp_lucaone_v2_data_prot_02;

-- colabfold_envdb(done) 208,966,064
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_03_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_03_external
(
obj_id string,
obj_type string,
obj_seq string,
obj_label string
)
stored by 'com.aliyun.odps.CsvStorageHandler'
with serdeproperties (
'odps.text.option.use.quote'='true',
'delimiter'=',',
'quoteChar'='"',
'odps.text.option.header.lines.count' = '1',
'odps.sql.text.option.flush.header' = 'true'
)
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_data/prot/colabfold_envdb/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_03_external
select *
from lucaone_data.tmp_lucaone_v2_data_prot_03;


-- uniref50
set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_dev;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_dev
as
select *, "uniref50" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_01
    ) tmp
order by rand_v
    limit 10000;


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_test;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_test
as
select *, "uniref50" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_01
        where obj_id not in (
            select obj_id
            from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_dev
            where obj_id is not null
        )
    ) tmp
order by rand_v
    limit 10000;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_train;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_train
as
select *, "uniref50" as obj_source
from lucaone_data.tmp_lucaone_v2_data_prot_01
where obj_id not in (
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_dev
    where obj_id is not null
    union all
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_test
    where obj_id is not null
);


-- uniprot
set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_dev;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_dev
as
select *, "uniprot" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_02
    ) tmp
order by rand_v
    limit 10000;


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_test;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_test
as
select *, "uniprot" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_02
        where obj_id not in (
            select obj_id
            from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_dev
            where obj_id is not null
        )
    ) tmp
order by rand_v
    limit 10000;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_train;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_train
as
select *, "uniprot" as obj_source
from lucaone_data.tmp_lucaone_v2_data_prot_02
where obj_id not in (
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_dev
    where obj_id is not null
    union all
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_test
    where obj_id is not null
);



-- colabfold_envdb
set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_dev;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_dev
as
select *, "colabfold_envdb" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_03
    ) tmp
order by rand_v
    limit 10000;


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_test;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_test
as
select *, "colabfold_envdb" as obj_source
from(
        select *, rand(unix_timestamp()) as rand_v
        from lucaone_data.tmp_lucaone_v2_data_prot_03
        where obj_id not in (
            select obj_id
            from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_dev
            where obj_id is not null
        )
    ) tmp
order by rand_v
    limit 10000;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_train;
create table if not exists lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_train
as
select *, "colabfold_envdb" as obj_source
from lucaone_data.tmp_lucaone_v2_data_prot_03
where obj_id not in (
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_dev
    where obj_id is not null
    union all
    select obj_id
    from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_test
    where obj_id is not null
);

-- odps->oss

-- train(done 523,227,512)
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_train_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_train_external
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
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/train/prot/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_train_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_train
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_train
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_train
    ) tmp;

-- dev(done 30,000)
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_dev_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_dev_external
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
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/dev/prot/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_dev_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_dev
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_dev
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_dev
    ) tmp;


-- test (done 30,000)
drop table if exists lucaone_data.tmp_lucaone_v2_data_prot_test_external;
create external table if not exists lucaone_data.tmp_lucaone_v2_data_prot_test_external
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
location 'oss://oss-cn-beijing-internal.aliyuncs.com/lucaone-data/lucaone_v2_dataset/test/prot/';

insert overwrite table lucaone_data.tmp_lucaone_v2_data_prot_test_external
select *
from(
        select obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniref50_rand_test
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_uniprot_rand_test
        union all
        select  obj_id, obj_type, obj_seq, obj_label, obj_source
        from lucaone_data.tmp_lucaone_v2_data_prot_colabfold_envdb_rand_test
    ) tmp;
