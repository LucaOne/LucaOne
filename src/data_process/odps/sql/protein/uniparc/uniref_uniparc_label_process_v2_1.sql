--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-16 15:52:32
--********************************************************************--
-- fasta信息
-- tmp_lucaone_v2_prot_fasta_info_uniref_uniparc
-- domain信息
-- tmp_lucaone_v2_prot_domain_info_uniref_uniparc
-- xref信息
-- tmp_lucaone_v2_prot_xref_info_uniref_uniparc
-- xref_2_property 信息
-- tmp_lucaone_v2_prot_xref_2_property_info_uniref_uniparc
CREATE FUNCTION interpro_db_priority_v2 AS 'interpro_db_priority_2_udf.interpro_db_priority' USING 'interpro_db_priority_2_udf.py' -f ;

SET odps.stage.mapper.mem=12288;
--set odps.stage.mapper.split.size=128;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;
-- domain信息数据库优先级计算

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_prot_domain_info_uniref_uniparc_db_priority_v2 ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_prot_domain_info_uniref_uniparc_db_priority_v2 AS
SELECT  uniparc_id AS prot_seq_accession
     ,origin_db
     ,interpro_accession
     ,interpro_name
     ,interpro_feature_type
     ,case    WHEN interpro_feature_type IN ('Homologous_superfamily') THEN 'Homologous_superfamily'
              WHEN interpro_feature_type IN ('Active_site', 'Binding_site', 'Conserved_site', 'PTM') THEN 'Site'
              WHEN interpro_feature_type IN ('Domain') THEN 'Domain'
              ELSE interpro_feature_type
    END AS alg_feature_type
     ,start_p
     ,end_p
     ,priority
FROM    (
            SELECT  DISTINCT uniparc_id
                           ,database AS origin_db
                           ,interpro_id
                           ,interpro_name
                           ,domain_start AS start_p
                           ,domain_end AS end_p
                           ,interpro_db_priority(DATABASE) AS priority
            FROM    luca_data2.tmp_lucaone_v2_prot_domain_info_uniref_uniparc
        ) t1
            JOIN    (
    SELECT  DISTINCT interpro_accession
                   ,interpro_feature_type
    FROM    luca_data2.tmp_lucaone_v2_interpro_entry_info
) t2
                    ON      t1.interpro_id = t2.interpro_accession
;
-- 每一种alg_feature_type，选择db最高优先级的db

DROP TABLE IF EXISTS tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_db_v2 ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_db_v2 AS
SELECT  DISTINCT prot_seq_accession
               ,alg_feature_type
               ,origin_db
FROM    (
            SELECT  *
                 ,ROW_NUMBER() OVER (PARTITION BY prot_seq_accession,alg_feature_type ORDER BY priority ASC ) AS rank_id
            FROM    tmp_lucaone_v2_prot_domain_info_uniref_uniparc_db_priority_v2
        ) t
WHERE   rank_id = 1
;
-- 选择

DROP TABLE IF EXISTS tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_v2 ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_v2 AS
SELECT  t1.*
FROM    tmp_lucaone_v2_prot_domain_info_uniref_uniparc_db_priority_v2 t1
            JOIN    tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_db_v2 t2
WHERE   t1.prot_seq_accession = t2.prot_seq_accession
  AND     t1.alg_feature_type = t2.alg_feature_type
  AND     t1.origin_db = t2.origin_db
;

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_uniref_uniparc_label_detail_v2 ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_uniref_uniparc_label_detail_v2 AS
SELECT  t1.prot_seq_accession AS prot_seq_accession
     ,seq_clean(seq) AS seq
     ,taxid
     ,order_bio
     ,keywords
     ,prot_feature_name
     ,prot_feature_type
     ,start_p
     ,end_p
FROM    (
            SELECT  ttt1.uniparc_id AS prot_seq_accession
                 ,seq
                 ,ttt2.taxid AS taxid
                 ,order_bio
                 ,'' AS keywords
            FROM    (
                        SELECT  DISTINCT uniparc_id
                                       ,seq_len
                                       ,seq
                        FROM    luca_data2.tmp_lucaone_v2_prot_fasta_info_uniref_uniparc
                    ) ttt1
                        LEFT OUTER JOIN (
                SELECT  DISTINCT uniparc_id
                               ,property_value AS taxid
                FROM    luca_data2.tmp_lucaone_v2_prot_xref_2_property_info_uniref_uniparc
                WHERE   property_type = 'NCBI_taxonomy_id'
            ) ttt2
                                        ON      ttt1.uniparc_id = ttt2.uniparc_id
                        LEFT OUTER JOIN (
                SELECT  DISTINCT taxid
                               , order_bio
                FROM    luca_data2.tmp_lucaone_v2_taxid_mapping_final
                WHERE   order_bio IS NOT NULL
            ) ttt3
                                        ON      ttt2.taxid = ttt3.taxid
        ) t1
            LEFT OUTER JOIN (
    SELECT  *
    FROM    (
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_v2
                WHERE   alg_feature_type = "Homologous_superfamily"
                UNION ALL
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_v2
                WHERE   alg_feature_type = 'Site'
                UNION ALL
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_uniref_uniparc_interpro_db_priority_selected_v2
                WHERE   alg_feature_type = 'Domain'
            ) tmp
) t2
                            ON      t1.prot_seq_accession = t2.prot_seq_accession
;



