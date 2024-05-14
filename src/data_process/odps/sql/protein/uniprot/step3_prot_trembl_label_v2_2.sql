--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-08 23:03:23
--description: 处理uniprot/trembl的信息
--********************************************************************--
-- 对每个蛋白的label进行组装
-- label文件需要先从sql文件protein_label_list中导入到本地，然后以资源的信息上传（这样做是为了与后面算法label文件保存一致）
-- prot_site_span_level_label_v2.txt
-- prot_homo_span_level_label_v2.txt
-- prot_domain_span_level_label_v2.txt
-- prot_taxonomy_seq_level_label_v2.txt
-- prot_keyword_seq_level_label_v2.txt

SET odps.sql.python.version=cp37;

CREATE FUNCTION trembl_label_process_v2 AS 'trembl_label_process_v2_udf.trembl_label_process' USING 'trembl_label_process_v2_udf.py,prot_domain_span_level_label_v2.txt,prot_site_span_level_label_v2.txt,prot_homo_span_level_label_v2.txt,prot_taxonomy_seq_level_label_v2.txt,prot_keyword_seq_level_label_v2.txt,biopython-1.80-cp37.zip' -f ;
-- seq_id, taxid, order_bio, keywords, prot_feature_name,  prot_feature_type, start_p, end_p

DROP TABLE IF EXISTS tmp_lucaone_v2_uniprot_trembl_all_detail_step1_v2 ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniprot_trembl_all_detail_step1_v2 AS
SELECT  prot_seq_accession AS seq_id
     ,seq
     ,trembl_label_process_v2(
        prot_seq_accession
    ,taxid
    ,order_bio
    ,keywords
    ,prot_feature_name
    ,prot_feature_type
    ,start_p
    ,end_p
    ) AS labels
FROM    tmp_lucaone_v2_uniprot_trembl_label_detail_v2
GROUP BY prot_seq_accession, seq
;
-- 对每个蛋白不存在的那部分label使用占位符

SET odps.sql.python.version=cp37;

CREATE FUNCTION prot_label_fill_v2 AS 'prot_label_fill_v2_udf.prot_label_fill' USING 'prot_label_fill_v2_udf.py' -f ;

DROP TABLE IF EXISTS tmp_lucaone_v2_uniprot_trembl_all_detail_step2_v2 ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniprot_trembl_all_detail_step2_v2 AS
SELECT  seq_id
     ,seq
     ,prot_label_fill_v2(seq_id, labels) AS labels
FROM    tmp_lucaone_v2_uniprot_trembl_all_detail_step1_v2
;
-- 验证span不能超过序列长度，超过的则去掉

SET odps.sql.python.version=cp37;

CREATE FUNCTION span_verify_v2 AS 'span_verify_v2_udf.span_verify' USING 'span_verify_v2_udf.py' -f ;

DROP TABLE IF EXISTS tmp_lucaone_v2_uniprot_trembl_all_detail_v2 ;

CREATE TABLE tmp_lucaone_v2_uniprot_trembl_all_detail_v2 AS
SELECT  seq_id
     ,seq
     ,labels
FROM    (
            SELECT  seq_id
                 ,seq
                 ,labels
                 ,span_verify_v2(seq_id, seq, labels) AS flag
            FROM    tmp_lucaone_v2_uniprot_trembl_all_detail_step2_v2
        ) t1
WHERE   flag IS NULL
;
