--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-12-22 17:38:09
--********************************************************************--
select count(distinct SPLIT_PART(obj_id, '_r_', 1))
from tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_1278_final
where instr(obj_id, '_r_') <= 0;

select count(distinct seq_id)
from tmp_lucaone_v2_refseq_100w_all_detail_v2_1278
where instr(seq_id, '_r_') <= 0;

select count(distinct seq_id)
from tmp_lucaone_v2_refseq_100w_all_detail_v2_1278
where instr(seq_id, '_r_') > 0;

select count(distinct seq_id)
from tmp_lucaone_v2_refseq_100w_all_detail_rna_1278;

show tables;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len < 100;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 100 and seq_len < 1000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 1000 and seq_len < 10000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 10000 and seq_len < 100000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 100000 and seq_len < 1000000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 1000000 and seq_len < 10000000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 10000000 and seq_len < 100000000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 100000000 and seq_len < 1000000000;

select count(seq_id) as total
from luca_data2.tmp_lucaone_v2_refseq_seq_info
where seq_len >= 1000000000;


desc tmp_lucaone_v2_refseq_seq_info;

-- 最终的表

-- RNA 136,311,178
select count(1)
from lucaone_data.tmp_lucaone_v2_data_gene2_01;

-- DNA 1,181,133,873
select count(1)
from lucaone_data.tmp_lucaone_v2_data_gene2_02;

-- uniref 62,150,523
select count(1)
from lucaone_data.tmp_lucaone_v2_data_prot_01;

-- uniprot 252,170,925
select count(1)
from lucaone_data.tmp_lucaone_v2_data_prot_02;

-- colab 208,966,064
select count(1)
from lucaone_data.tmp_lucaone_v2_data_prot_03;


select *
from lucaone_data.tmp_lucaone_v2_data_prot_02
         limit 10;


SET odps.sql.python.version=cp37;

CREATE FUNCTION data_stats AS 'data_stats.data_stats' USING 'data_stats.py' -f ;
-- RNA 136,311,178
select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from lucaone_data.tmp_lucaone_v2_data_gene2_01
group by 1;

-- DNA 1,181,133,873
select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from lucaone_data.tmp_lucaone_v2_data_gene2_02
group by 1;

-- uniref 62,150,523
select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from lucaone_data.tmp_lucaone_v2_data_prot_01
group by 1;

-- uniprot 252,170,925
select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from lucaone_data.tmp_lucaone_v2_data_prot_02
group by 1;

-- colab 208,966,064
select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from lucaone_data.tmp_lucaone_v2_data_prot_03
group by 1;


select data_stats(obj_id, obj_type, obj_seq, obj_label) as res
from (
         select *
         from lucaone_data.tmp_lucaone_v2_data_prot_01
         union all
         select *
         from lucaone_data.tmp_lucaone_v2_data_prot_02
         union all
         select *
         from lucaone_data.tmp_lucaone_v2_data_prot_03
     ) tmp
group by 1;


select *
from tmp_lucaone_v2_taxid_mapping
         limit 10;

select *
from lucaone_data.tmp_lucaone_v2_data_gene2_02
         limit 10;

desc tmp_lucaone_v2_taxid_mapping;

-- 统计物种
SET odps.sql.python.version=cp37;

CREATE FUNCTION extract_taxid AS 'extract_taxid_udf.extract_taxid' USING 'extract_taxid_udf.py' -f ;
-- RNA 136,311,178
create table if not exists tmp_lucaone_v2_data_gene2_01_species_stats
as
select species, count(obj_id) as cnt
from(
        select species, obj_id, tax_id, taxid
        from
            (
                select obj_id, extract_taxid(obj_type, obj_label) as tax_id
                from lucaone_data.tmp_lucaone_v2_data_gene2_01
            ) t1
                join
            tmp_lucaone_v2_taxid_mapping t2
            on t1.tax_id = t2.taxid
    ) t
where tax_id is not null and taxid is not null
group by species
order by cnt desc;


-- DNA 1,181,133,873
create table if not exists tmp_lucaone_v2_data_gene2_02_species_stats
as
select species, count(obj_id) as cnt
from(
        select species, obj_id, tax_id, taxid
        from
            (
                select obj_id, extract_taxid(obj_type, obj_label) as tax_id
                from lucaone_data.tmp_lucaone_v2_data_gene2_02
            ) t1
                join
            tmp_lucaone_v2_taxid_mapping t2
            on t1.tax_id = t2.taxid
    ) t
where tax_id is not null and taxid is not null
group by species
order by cnt desc;

-- RNA + DNA
create table if not exists tmp_lucaone_v2_data_gene2_species_stats
as
select species, sum(cnt) as cnt
from(
        select species, count(obj_id) as cnt
        from(
                select species, obj_id, tax_id, taxid
                from
                    (
                        select obj_id, extract_taxid(obj_type, obj_label) as tax_id
                        from lucaone_data.tmp_lucaone_v2_data_gene2_01
                    ) t1
                        join
                    tmp_lucaone_v2_taxid_mapping t2
                    on t1.tax_id = t2.taxid
            ) t
        where tax_id is not null and taxid is not null
        group by species

        union ALL
        select species, count(obj_id) as cnt
        from(
                select species, obj_id, tax_id, taxid
                from
                    (
                        select obj_id, extract_taxid(obj_type, obj_label) as tax_id
                        from lucaone_data.tmp_lucaone_v2_data_gene2_02
                    ) t1
                        join
                    tmp_lucaone_v2_taxid_mapping t2
                    on t1.tax_id = t2.taxid
            ) t
        where tax_id is not null and taxid is not null
        group by species
    ) t
group by species
order by cnt desc;


-- prot
create table if not exists tmp_lucaone_v2_data_prot_02_species_stats
as
select species, count(obj_id) as cnt
from(
        select species, obj_id, tax_id, taxid
        from
            (
                select obj_id, extract_taxid(obj_type, obj_label) as tax_id
                from (
                         select *
                         from lucaone_data.tmp_lucaone_v2_data_prot_01
                         union all
                         select *
                         from lucaone_data.tmp_lucaone_v2_data_prot_02
                         union all
                         select *
                         from lucaone_data.tmp_lucaone_v2_data_prot_03
                     ) tmp
            ) t1
                join
            tmp_lucaone_v2_taxid_mapping t2
            on t1.tax_id = t2.taxid
    ) t
where tax_id is not null and taxid is not null
group by species
order by cnt desc;

select count(distinct species)
from tmp_lucaone_v2_data_gene2_species_stats
where cnt >= 10;

select count(distinct species)
from tmp_lucaone_v2_data_prot_02_species_stats
where cnt >= 10;

select count(distinct species)
from tmp_lucaone_v2_data_gene2_species_stats
where cnt >= 1;

select count(distinct species)
from tmp_lucaone_v2_data_prot_02_species_stats
where cnt >= 1;

select *
from tmp_lucaone_v2_data_prot_02_species_stats;

-- 169861
select count(distinct species)
from
    (
        select species, sum(cnt) as cnt
        from(
                select species, cnt
                from tmp_lucaone_v2_data_gene2_species_stats
                union all
                select species, cnt
                from tmp_lucaone_v2_data_prot_02_species_stats
            ) t
        GROUP by species
        HAVING sum(cnt) >= 10
    ) tmp;


drop table lucaone_v2_species_seq_count_stats;
create table lucaone_v2_species_seq_count_stats AS
select species, sum(cnt) as seq_cnt
from(
        select species, cnt
        from tmp_lucaone_v2_data_gene2_species_stats
        where species is not null
        union all
        select species, cnt
        from tmp_lucaone_v2_data_prot_02_species_stats
        where species is not null
    ) t
GROUP by species
HAVING sum(cnt) >= 10;

select count(1) from lucaone_v2_species_seq_count_stats
where species is not null;

-- 1125050
select count(distinct species)
from
    (
        select species, sum(cnt) as cnt
        from(
                select species, cnt
                from tmp_lucaone_v2_data_gene2_species_stats
                union all
                select species, cnt
                from tmp_lucaone_v2_data_prot_02_species_stats
            ) t
        GROUP by species
    ) tmp;


-- 168508
select count(distinct species)
from
    (
        select species, sum(cnt) as cnt
        from(
                select species, cnt
                from tmp_lucaone_v2_data_gene2_species_stats
                where cnt >= 10
                union all
                select species, cnt
                from tmp_lucaone_v2_data_prot_02_species_stats
                where cnt >= 10
            ) t
        GROUP by species
    ) tmp;

-- RNA 136,311,178
select 'refseq_rna' as db, sum(LENGTH(obj_seq))
from lucaone_data.tmp_lucaone_v2_data_gene2_01;

-- DNA 1,181,133,873
select 'refseq_dna' as db, sum(LENGTH(obj_seq))
from lucaone_data.tmp_lucaone_v2_data_gene2_02;

-- uniref 62,150,523
select 'uniref50_prot' as db, sum(LENGTH(obj_seq))
from lucaone_data.tmp_lucaone_v2_data_prot_01;

-- uniprot 252,170,925
select 'uniprot_prot' as db, sum(LENGTH(obj_seq))
from lucaone_data.tmp_lucaone_v2_data_prot_02;


-- colab 208,966,064
select 'colabfold_prot'  as db, sum(LENGTH(obj_seq))
from lucaone_data.tmp_lucaone_v2_data_prot_03;
