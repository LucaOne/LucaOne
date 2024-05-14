--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-01 10:58:51
--********************************************************************--
-- 1) 复合CDS中间区间是intron_manual
SET odps.sql.python.version=cp37;

CREATE FUNCTION intron_manual_parse AS 'intron_manual_parse_udf.intron_manual_parse' USING 'intron_manual_parse_udf.py,biopython-1.80-cp37.zip' -f ;

CREATE FUNCTION nucleo_cds_clean AS 'nucleo_cds_clean_udf.nucleo_cds_clean' USING 'nucleo_cds_clean_udf.py,biopython-1.80-cp37.zip' -f ;
-- 人工 内含子

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2 ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2  AS
SELECT  *
FROM    (
            SELECT  assembly_accession
                 ,seq_id
                 ,"intron_manual" AS feature_type
                 ,strand
                 ,intron_manual_parse(feature_type,start_end) AS start_end
                 ,0 AS feature_complete
                 ,qualifiers
                 ,insert_date
            FROM    luca_data2.tmp_lucaone_v2_refseq_fea_info_v2
            WHERE   feature_type IN ("CDS", "exon")
              AND     start_end LIKE "CompoundLocation%"
            UNION ALL
            SELECT  *
            FROM    luca_data2.tmp_lucaone_v2_refseq_fea_info_v2
        ) tmp
;

--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-24 00:53:53
--********************************************************************--
SET odps.sql.python.version=cp37;

CREATE FUNCTION refseq_nucleo_split_c1 AS 'refseq_nucleo_split_udaf_c1.refseq_nucleo_split' USING 'refseq_nucleo_split_udaf_c1.py,biopython-1.80-cp37.zip' -f ;

CREATE FUNCTION refseq_extract_split_info AS 'refseq_nucleo_extract_udtf.refseq_extract_split_info' USING 'refseq_nucleo_extract_udtf.py,biopython-1.80-cp37.zip' -f ;

SET odps.stage.mapper.mem=12288;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_c1_v2_${max_segment_len} ;

-- 使用gene片段分片,seq_id, feature_type, start_end, seq_len, max_len
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_c1_v2_${max_segment_len} AS
SELECT  refseq_extract_split_info(seq_id,split_info) AS (seq_id,strand,start_p,end_p,fragment_type,gene_idx,split_type,segment_len)
FROM    (
            SELECT  seq_id
                 ,refseq_nucleo_split_c1(seq_id,feature_type,start_end,seq_len, ${max_segment_len}) AS split_info
            FROM    (
                        SELECT  t1.*
                             ,seq_len
                        FROM    (
                                    SELECT  *
                                    FROM    luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                                    WHERE   seq_id IN (
                                        SELECT DISTINCT seq_id
                                        FROM luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                                        WHERE feature_type IN ('gene') and seq_id is not null
                                    )
                                ) t1
                                    JOIN    (
                            SELECT  DISTINCT seq_id
                                           ,seq_len
                            FROM    luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
                            where seq_len <= ${max_len}
                        ) t2
                                            ON      t1.seq_id = t2.seq_id
                    ) tt
            GROUP BY seq_id
        ) tmp
;

CREATE FUNCTION refseq_nucleo_split_c2 AS 'refseq_nucleo_split_udaf_c2.refseq_nucleo_split' USING 'refseq_nucleo_split_udaf_c2.py,biopython-1.80-cp37.zip' -f ;

CREATE FUNCTION refseq_extract_split_info AS 'refseq_nucleo_extract_udtf.refseq_extract_split_info' USING 'refseq_nucleo_extract_udtf.py,biopython-1.80-cp37.zip' -f ;

SET odps.stage.mapper.mem=12288;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_c2_v2_${max_segment_len};

-- 无gene有feature 合并 无feature的, seq_id, molecule_type, feature_type, start_end, seq_len, max_len)
-- inputs:
--         luca_data2.tmp_lucaone_v2_refseq_fea_info_clean: 561797146 (4725376099 bytes)
--         luca_data2.tmp_lucaone_v2_refseq_seq_info: 96493950 (405750982 bytes)
-- outputs:
--         luca_data2.tmp_lucaone_v2_refseq_nucleo_split_c2: 74795545 (369060675 bytes)

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_c2_v2_${max_segment_len} AS
SELECT  refseq_extract_split_info(seq_id,split_info) AS (seq_id,strand,start_p,end_p,fragment_type,gene_idx,split_type,segment_len)
FROM    (
            SELECT  seq_id
                 ,refseq_nucleo_split_c2(
                    seq_id
                ,molecule_type
                ,feature_type
                ,start_end
                ,seq_len
                ,${max_segment_len}
                ) AS split_info
            FROM    (
                        SELECT  t1.*
                             ,seq_len
                             ,molecule_type
                        FROM    (
                                    SELECT  *
                                    FROM    luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                                    WHERE   seq_id NOT IN (
                                        SELECT DISTINCT seq_id
                                        FROM luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                                        WHERE feature_type IN ('gene') and seq_id is not null)
                                ) t1
                                    JOIN    (
                            SELECT  DISTINCT seq_id
                                           ,seq_len
                                           ,molecule_type
                            FROM    luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
                            where seq_len <= ${max_len}
                        ) t2
                                            ON      t1.seq_id = t2.seq_id
                    ) tmp1
            GROUP BY seq_id
            UNION ALL
            SELECT  seq_id
                 ,refseq_nucleo_split_c2(
                    seq_id
                ,molecule_type
                ,feature_type
                ,start_end
                ,seq_len
                ,${max_segment_len}
                ) AS split_info
            FROM    (
                        SELECT  DISTINCT seq_id
                                       ,seq_len
                                       ,molecule_type
                                       ,NULL AS feature_type
                                       ,NULL AS start_end
                        FROM    luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
                        where seq_id not in (
                            SELECT  DISTINCT seq_id
                            FROM luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                            where seq_id is not null
                        ) and seq_len <= ${max_len}
                    ) tmp2
            GROUP BY seq_id
        ) tttt
;

DROP TABLE IF EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_v2_${max_segment_len};

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_refseq_nucleo_100w_split_v2_${max_segment_len} AS
SELECT  *
FROM    (
            SELECT  *
            FROM    tmp_lucaone_v2_refseq_nucleo_100w_split_c1_v2_${max_segment_len}
            UNION ALL
            SELECT  *
            FROM    tmp_lucaone_v2_refseq_nucleo_100w_split_c2_v2_${max_segment_len}
        ) tmp
;

--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-02 11:16:32
--********************************************************************--
DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_refseq_100w_label_detail_v2_${max_segment_len} ;

CREATE FUNCTION extract_position_v3 AS 'extract_position_udf_v3.extract_position' USING 'extract_position_udf_v3.py,biopython-1.80-cp37.zip' -f ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_refseq_100w_label_detail_v2_${max_segment_len} AS
SELECT  seq_id
     ,gene_idx
     ,strand
     ,fragment_type
     ,split_type
     ,segment_len
     ,feature_type
     ,start_end
     ,seq_start_p
     ,seq_end_p
     ,molecule_type
     ,taxid
     ,order_bio
     ,start_p_split
     ,end_p_split
FROM    (
            SELECT  t1.seq_id AS seq_id
                 ,gene_idx
                 ,t1.strand AS strand
                 ,fragment_type
                 ,split_type
                 ,segment_len
                 ,molecule_type
                 ,t2.taxid AS taxid
                 ,seq_start_p
                 ,seq_end_P
                 ,feature_type
                 ,start_end
                 ,order_bio
                 ,start_p_split
                 ,end_p_split
            FROM    (
                        SELECT  DISTINCT seq_id
                                       ,gene_idx
                                       ,strand
                                       ,fragment_type
                                       ,split_type
                                       ,segment_len
                                       ,start_p AS seq_start_p
                                       ,end_p AS seq_end_P
                        FROM    luca_data2.tmp_lucaone_v2_refseq_nucleo_100w_split_v2_${max_segment_len}
                    ) t1
                        LEFT JOIN (
                SELECT  seq_id
                     ,molecule_type
                     ,tt1.taxid AS taxid
                     ,order_bio
                FROM    (
                            SELECT  DISTINCT seq_id
                                           ,molecule_type
                                           ,taxid
                            FROM    luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
                            UNION ALL
                            SELECT  DISTINCT concat(seq_id, "_r_") AS seq_id
                                           ,molecule_type
                                           ,taxid
                            FROM    luca_data2.tmp_lucaone_v2_refseq_seq_info_v2
                        ) tt1
                            LEFT JOIN (
                    SELECT  DISTINCT taxid
                                   ,order_bio
                    FROM    tmp_lucaone_v2_taxid_mapping_final
                    WHERE   order_bio IS NOT NULL
                ) tt2
                                      ON      tt1.taxid = tt2.taxid
            ) t2
                                  ON      t1.seq_id = t2.seq_id LEFT
                            JOIN    (
                SELECT  *
                     ,cast(split_part(start_end_split, ',', 1) AS BIGINT) AS start_p_split
                     ,cast(split_part(start_end_split, ',', 2) AS BIGINT) AS end_p_split
                FROM    (
                            SELECT  DISTINCT CASE    WHEN strand = 1 THEN seq_id
                                                     ELSE concat(seq_id, "_r_")
                                END AS seq_id
                                           ,strand
                                           ,feature_type
                                           ,extract_position_v3(start_end) AS start_end_split
                                           ,start_end
                            FROM    luca_data2.tmp_lucaone_v2_refseq_fea_info_intron_manual_v2
                            WHERE   feature_type IN ('CDS','exon','intron','intron_manual','tRNA','ncRNA','rRNA','misc_RNA','tmRNA','regulatory')
                        ) tttt3
            ) t3
                                    ON      t1.seq_id = t3.seq_id
                                        AND     t1.strand = t3.strand
        ) b

;
--WHERE   end_p_split IS NULL
--OR      start_p_split IS NULL
--OR      seq_start_p <= end_p_split - 1
--AND     seq_end_p - 1 >= start_p_split
-- 需要获取gene_type_span_level_label.txt， gene_taxonomy_seq_level_label.txt并上传资源

SET odps.sql.python.version=cp37;

SET odps.stage.mapper.split.size=10240;

CREATE FUNCTION refseq_labels_process AS 'refseq_labels_process_udf.refseq_labels_process' USING 'refseq_labels_process_udf.py,gene_type_span_level_label_v2.txt,gene_taxonomy_seq_level_label_v2.txt,biopython-1.80-cp37.zip' -f ;

DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}  ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}  AS
SELECT  seq_id
     ,gene_idx
     ,seq_start_p
     ,seq_end_p
     ,molecule_type
     ,refseq_labels_process(
        seq_id
    ,gene_idx
    ,seq_start_p
    ,seq_end_p
    ,feature_type
    ,start_end
    ,taxid
    ,order_bio
    ) AS labels
FROM    luca_data2.tmp_lucaone_v2_refseq_100w_label_detail_v2_${max_segment_len}
GROUP BY seq_id
       ,gene_idx
       ,seq_start_p
       ,seq_end_p
       ,molecule_type
;

-- 对部分label不存在的则进行占位符fill
set odps.sql.python.version=cp37;
create function gene_label_fill as 'gene_label_fill_udf.gene_label_fill' using 'gene_label_fill_udf.py' -f;

drop table if exists luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step1;
CREATE TABLE if not exists luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step1 AS
SELECT seq_id
     ,gene_idx
     ,seq_start_p
     ,seq_end_p
     ,molecule_type,
    gene_label_fill(seq_start_p, seq_end_p, labels) as labels
FROM  luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len};



-- 验证span不能超过序列长度（去掉那些超过的）
set odps.sql.python.version=cp37;
create function refseq_span_verify as 'refseq_span_verify_udf.refseq_span_verify' using 'refseq_span_verify_udf.py' -f;

drop table if exists luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step2;
create table if not exists luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step2 AS
select *
from (
         select *
              ,refseq_span_verify(seq_id, seq_end_p - seq_start_p, labels) as flag
         from luca_data2.tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step1
     ) tmp
where flag is null;

SET odps.sql.python.version=cp37;

SET odps.stage.mapper.split.size=10240;

set odps.sql.python.version=cp37;
CREATE FUNCTION seq_reverse AS 'seq_reverse_udf.seq_reverse' USING 'seq_reverse_udf.py' -f ;

--
drop table if exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len};
create table if not exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}
as
select seq_id as seq_id, gene_idx, seq_start_p, seq_end_p, molecule_type, labels,
       SUBSTR(seq, seq_start_p + 1, seq_end_p - seq_start_p) as seq_segment
from(
        select t1.*, seq
        from tmp_lucaone_v2_refseq_100w_all_detail_v2_${max_segment_len}_step2 t1
                 join (
            select distinct seq_id, seq_clean(seq) as seq
            from tmp_lucaone_v2_refseq_dna_fasta_info_100w
            union ALL
            select distinct concat(seq_id, "_r_") as seq_id, seq_reverse(seq) as seq
            from tmp_lucaone_v2_refseq_dna_fasta_info_100w
        ) t2
                      on t1.seq_id = t2.seq_id
    ) tmp;

drop table if exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_verfiy;
create table if not exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_verfiy
as
select *
from tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}
where seq_end_p - seq_start_p = LENGTH(seq_segment);

drop table if exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_final;
create table if not exists tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_final
as
select distinct concat(seq_id, "_", gene_idx, "_", seq_start_p, "_", seq_end_p, "_", molecule_type) as obj_id,
                "gene" as obj_type,
                seq_segment as obj_seq,
                labels as obj_label
from tmp_lucaone_v2_refseq_dna_100w_all_detail_v2_${max_segment_len}_verfiy;





