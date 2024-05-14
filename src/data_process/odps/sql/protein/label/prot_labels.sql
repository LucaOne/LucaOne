--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 15:31:01
--********************************************************************--

-- prot_binding_span_level_label_v2.txt
-- prot_homo_span_level_label_v2.txt
-- prot_site_span_level_label_v2.txt
-- prot_domain_span_level_label_v2.txt
-- prot_taxonomy_seq_level_label_v2.txt
-- prot_keyword_seq_level_label_v2.txt


-- span:homo:label list:(3438->3443)
drop table if exists prot_homo_span_level_label_v2;
create table if not exists prot_homo_span_level_label_v2
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_sprot_label_detail_v2
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_trembl_label_detail_v2
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniref_uniparc_label_detail_v2
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2
        where prot_feature_type = "Homologous_superfamily"
    ) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;



-- span:site:label list: binding
--drop table if exists prot_binding_span_level_label_v2;
create table if not exists prot_binding_span_level_label_v2
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_sprot_label_detail
        where prot_feature_type = 'BINDING'
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_trembl_label_detail
        where prot_feature_type = 'BINDING'
    ) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;


-- span:site:label list: Site(922->946)
drop table if exists prot_site_span_level_label_v2;
create table if not exists prot_site_span_level_label_v2
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_sprot_label_detail_v2
        where prot_feature_type = 'Site'
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_trembl_label_detail_v2
        where prot_feature_type = 'Site'
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniref_uniparc_label_detail_v2
        where prot_feature_type = "Site"
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2
        where prot_feature_type = "Site"
    ) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;

-- span:domain:label list: Domain(13717)
drop table if exists prot_domain_span_level_label_v2;
create table if not exists prot_domain_span_level_label_v2
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_sprot_label_detail_v2
        where prot_feature_type = 'Domain'
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniprot_trembl_label_detail_v2
        where prot_feature_type = 'Domain'
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_uniref_uniparc_label_detail_v2
        where prot_feature_type = "Domain"
        union ALL
        select distinct prot_feature_name
        from tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2
        where prot_feature_type = "Domain"
    ) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;


-- seq:taxonomy:label list(1651->2196):
drop table if exists prot_taxonomy_seq_level_label_v2;
create table if not exists prot_taxonomy_seq_level_label_v2
AS
select distinct order_bio as label
from(
        select distinct order_bio
        from tmp_lucaone_v2_uniprot_sprot_label_detail
        union ALL
        select distinct order_bio
        from tmp_lucaone_v2_uniprot_trembl_label_detail
        union ALL
        select distinct order_bio
        from tmp_lucaone_v2_uniref_uniparc_label_detail_v2
        union ALL
        select distinct order_bio
        from tmp_lucaone_v2_colabfold_envdb_uniparc_label_detail_v2
        union ALL
        select distinct order_bio
        from tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2
    ) t
where order_bio is not NULL and length(order_bio) > 0;



set odps.sql.python.version=cp37;
create function split_2_multi_rows as 'split_2_multi_rows_udf.split_2_multi_rows' using 'split_2_multi_rows_udf.py' -f;

-- seq:keyword:label list:(1179)
drop table if exists prot_keyword_seq_level_label_v2;
create table if not exists prot_keyword_seq_level_label_v2
AS
select distinct keyword as label
from(
        select DISTINCT keyword
        from(
                select split_2_multi_rows(keywords, ";") as keyword
                from tmp_lucaone_v2_uniprot_sprot_label_detail
                union ALL
                select split_2_multi_rows(keywords, ";") as keyword
                from tmp_lucaone_v2_uniprot_trembl_label_detail
            ) tmp
        where length(keyword) > 0
    ) t
where keyword is not NULL and length(keyword) > 0;

--show instances;


select ref_id, count(1)
from tmp_lucaone_v2_colabfold_envdb_label_detail_c9_v2_cluster
group by ref_id
having COUNT (1) > 1;


select ref_id, count(1)
from tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster
group by ref_id
having COUNT (1) > 1;

select *
from tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster
where ref_id = 'ERR1711939_300896|ERR868362_k119_2687573|-|462|2.318e-129|2|40756|41383|41383[41383]:41273[41273]:111[111]|41229[41238]:40756[40756]:474[465]   ';

select count(1)
from tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster;

select *
from luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster
where ref_id not in (select seq_id from tmp_lucaone_v2_colabfold_envdb_fasta_info);

tmp_lucaone_v2_colabfold_envdb_db_from_cluster

select *
from luca_data2.tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster
where seq_id not in (select ref_id from tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2_cluster where ref_id is not null);

select count(distinct seq_id)
from luca_data2.tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster;

select count(distinct seq_id)
from luca_data2.tmp_lucaone_v2_colabfold_envdb_fasta_info_cluster;

show instances;

rna
20231122125706673gqk670xe8kf
dna.

20231122143127683gafe79691i6


select *
from lucaone_data.tmp_lucaone_v2_refseq_fasta_info_100w
where seq_id != "seq_id";


create table if not exists luca_data2.tmp_lucaone_v2_refseq_fasta_info_100w_bak
as
select *
from luca_data2.tmp_lucaone_v2_refseq_fasta_info_100w;

limit 10;

-- 279
select assembly_accession, seq_id
from luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
where seq_len <= 1000000 and seq_id not in (select seq_id from tmp_lucaone_v2_refseq_fasta_info_100w_bak where seq_id is not null);

-- 0
select count(seq_id)
from luca_data2.tmp_lucaone_v2_refseq_seq_info_rna
where seq_len <= 1000000 and seq_id not in (select seq_id from lucaone_data.tmp_lucaone_v2_refseq_rna_fasta_info where seq_id is not null);


select *
from luca_data2.tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_1022_final
         limit 10;

select *
from luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022_final
         limit 10;

select *
from tmp_lucaone_v2_refseq_100w_all_detail_rna_1022_step2
where seq_id like 'NM_000733.4%';

select *, length(seq), substr(seq, 1022, 1361 - 1022)
from tmp_lucaone_v2_refseq_rna_fasta_info
where seq_id like 'NM_000733.4%';

select *
from luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022
where seq_id  like 'NM_000733.4%';

select *
from luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022_final
where obj_id  like 'NM_000733.4%';


inputs:
        luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_rna_1022_step2: 163777456 (1231107459 bytes)
        luca_data2.tmp_lucaone_v2_refseq_rna_fasta_info: 51664008 (26624256293 bytes)
outputs:
        luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022: 164159016 (30209765679 bytes)




drop table if exists tmp_lucaone_v2_refseq_rna_all_detail_v2_${max_segment_len}_final;
create table if not exists tmp_lucaone_v2_refseq_rna_all_detail_v2_${max_segment_len}_final
as
select distinct concat(seq_id, "_", gene_idx, "_", seq_start_p, "_", seq_end_p, "_", molecule_type) as obj_id,
                "gene" as obj_type,
                seq_segment as obj_seq,
                labels as obj_label
from tmp_lucaone_v2_refseq_rna_all_detail_v2_${max_segment_len};

inputs:
        luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022: 164159016 (30407166246 bytes)
outputs:
        luca_data2.tmp_lucaone_v2_refseq_rna_all_detail_v2_1022_final: 163777456 (30557743628 bytes)


select *
from luca_data2.tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_1022_final
where obj_seq is null or LENGTH(obj_seq) == 0;


select *
from tmp_lucaone_v2_refseq_rna_all_detail_v2_1022
where replace(seq_id, "_r_", "") not in (select seq_id from luca_data2.tmp_lucaone_v2_refseq_dna_rna_fasta_info_100w);

select *
from tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_1022
where replace(seq_id, "_r_", "") not in (select seq_id from luca_data2.tmp_lucaone_v2_refseq_dna_rna_fasta_info_100w);


set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists lucaone_data.tmp_lucaone_v2_data_gene_rna_rand;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene_rna_rand
as
select *,  row_number() over(partition by 1 order by rand_v) as rank_id
from(
        select *, "rna" as obj_source, rand() as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene_01
    )  tmp;

set odps.sql.executionengine.enable.rand.time.seed=true;
drop table if exists  lucaone_data.tmp_lucaone_v2_data_gene_genomic_rand;
create table if not exists lucaone_data.tmp_lucaone_v2_data_gene_genomic_rand
as
select *,  row_number() over(partition by 1 order by rand_v) as rank_id
from(
        select *, "genomic" as obj_source, rand() as rand_v
        from lucaone_data.tmp_lucaone_v2_data_gene_02
    )  tmp;