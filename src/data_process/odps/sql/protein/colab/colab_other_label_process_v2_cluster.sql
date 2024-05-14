--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-19 12:17:50
--********************************************************************--
-- MetaEuk tmp_lucaone_v2_MetaEuk_taxonomy_info
-- MGV tmp_lucaone_v2_mgv_taxonomy_info
-- MGY tmp_lucaone_v2_mgnify_taxonomy_info
-- 只有这三个有order种族信息。


SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_colab_other_label_detail_v2_cluster ;
CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_colab_other_label_detail_v2_cluster AS
SELECT  prot_seq_accession
     ,seq_clean(seq) as seq
     ,taxid
     ,order_bio
     ,keywords
     ,'' as prot_feature_name
     ,'' as prot_feature_type
     ,cast(-1 as bigint) as start_p
     ,cast(-1 as bigint) as end_p
FROM  (
          SELECT  seq_id AS prot_seq_accession
               ,seq
               ,'' AS taxid
               ,order_bio
               ,'' as keywords
          FROM
              (
                  select distinct seq_id, seq, order_bio
                  from tmp_lucaone_v2_MetaEuk_taxonomy_info
                  union all
                  select distinct seq_id, seq, order_bio
                  from tmp_lucaone_v2_mgv_taxonomy_info
                  union all
                  select distinct seq_id, seq, order_bio
                  from tmp_lucaone_v2_mgnify_taxonomy_info
              ) tmp
          where order_bio is not null and length(order_bio) > 0
      ) t1;

SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_colab_uniref_label_detail_v2_cluster ;
CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_colab_uniref_label_detail_v2_cluster AS
SELECT  prot_seq_accession
     ,seq_clean(seq) as seq
     ,taxid
     ,order_bio
     ,keywords
     ,'' as prot_feature_name
     ,'' as prot_feature_type
     ,cast(-1 as bigint) as start_p
     ,cast(-1 as bigint) as end_p
FROM  (
          SELECT  seq_id AS prot_seq_accession
               ,seq
               ,t1.taxid as taxid
               ,order_bio
               ,'' as keywords
          FROM
              (
                  select distinct db_seq_id as seq_id, seq, taxid
                  from tmp_lucaone_v2_colabfold_envdb_db_from_cluster
                  where taxid is not null and length(taxid) > 0
              ) t1
                  JOIN
              (
                  SELECT  DISTINCT taxid, order_bio
                  FROM    tmp_lucaone_v2_taxid_mapping_final
                  WHERE   order_bio IS NOT NULL

              ) t2
              on t1.taxid = t2.taxid
      ) t;