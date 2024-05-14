--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-08 23:03:23
--description: 处理uniprot/trembl的信息
--********************************************************************--
CREATE FUNCTION extract_position AS 'extract_position_udf.extract_position' USING 'extract_position_udf.py,biopython-1.80-cp37.zip' -f ;

CREATE FUNCTION uniprot_feature_parse AS 'uniprot_feature_parse_udf.uniprot_feature_parse' USING 'uniprot_feature_parse_udf.py,biopython-1.80-cp37.zip' -f ;

SET odps.stage.mapper.mem=12288;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS tmp_lucaone_v2_uniprot_trembl_label_detail_v2 ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniprot_trembl_label_detail_v2 AS
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
            SELECT  seq_id AS prot_seq_accession
                 ,seq
                 ,ttt01.taxid AS taxid
                 ,order_bio
                 ,keywords
            FROM    (
                        SELECT  *
                        FROM    (
                                    SELECT  tmp01.seq_id AS seq_id
                                         ,seq
                                         ,taxid
                                         ,keywords
                                    FROM    tmp_lucaone_v2_uniprot_trembl_fasta_info tmp01
                                                LEFT OUTER JOIN tmp_lucaone_v2_uniprot_trembl_seq_info tmp02
                                                                ON      tmp01.seq_id = tmp02.seq_id
                                ) tmp0
                    ) ttt01
                        LEFT OUTER JOIN (
                SELECT  DISTINCT taxid
                               ,order_bio
                FROM    tmp_lucaone_v2_taxid_mapping_final
                WHERE   order_bio IS NOT NULL
            ) ttt02
                                        ON      ttt01.taxid = ttt02.taxid
        ) t1
            LEFT OUTER JOIN (
    SELECT  *
    FROM    (
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_interpro_db_priority_selected
                WHERE   alg_feature_type = "Homologous_superfamily"
                UNION ALL
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_interpro_db_priority_selected
                WHERE   alg_feature_type = 'Site'
                UNION ALL
                SELECT  prot_seq_accession
                     ,interpro_accession AS prot_feature_name
                     ,alg_feature_type AS prot_feature_type
                     ,start_p - 1 AS start_p
                     ,end_p AS end_p
                FROM    tmp_lucaone_v2_interpro_db_priority_selected
                WHERE   alg_feature_type = 'Domain'
            ) tmp
) t2
                            ON      t1.prot_seq_accession = t2.prot_seq_accession
;

SELECT  count(DISTINCT prot_seq_accession)
     ,count(1)
FROM    tmp_lucaone_v2_uniprot_trembl_label_detail_v2
;


