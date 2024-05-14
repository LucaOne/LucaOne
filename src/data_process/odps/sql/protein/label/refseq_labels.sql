--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-09 15:52:29
--********************************************************************--
-- span:label list:
drop table if exists gene_type_span_level_label_v2;
create table if not exists gene_type_span_level_label_v2
AS
select distinct feature_type as label
from(
        select case
                   when feature_type = "exon" then "CDS"
                   when feature_type = "intron_manual" then "intron"
                   else feature_type
                   end as feature_type
        from tmp_lucaone_v2_refseq_label_detail_v2
        union ALL
        select case
                   when feature_type = "exon" then "CDS"
                   when feature_type = "intron_manual" then "intron"
                   else feature_type
                   end as feature_type
        from tmp_lucaone_v2_refseq_label_detail_rna
    ) t
where feature_type is not NULL and length(feature_type) > 0;


-- seq:taxonomy:label list:
drop table if exists gene_taxonomy_seq_level_label_v2;
create table if not exists gene_taxonomy_seq_level_label_v2
AS
select distinct label
from (
         select distinct order_bio as label
         from tmp_lucaone_v2_refseq_label_detail_v2
         where order_bio is not NULL and length(order_bio) > 0
         union all
         select distinct order_bio as label
         from tmp_lucaone_v2_refseq_label_detail_rna
         where order_bio is not NULL and length(order_bio) > 0
     ) t;
