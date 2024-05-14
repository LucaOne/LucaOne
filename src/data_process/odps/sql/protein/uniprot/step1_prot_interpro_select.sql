--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-08 23:02:18
--description: 按照数据来源的进行优先级选择，优先级信息在udf：interpro_db_priority_udf中
--********************************************************************--
DROP TABLE IF EXISTS luca_data2.tmp_lucaone_v2_interpro_db_priority ;

CREATE TABLE IF NOT EXISTS luca_data2.tmp_lucaone_v2_interpro_db_priority
(
    prot_seq_accession string
    ,interpro_accession string
    ,original_accession string
    ,interpro_feature_name string
    ,interpro_feature_type string
    ,alg_feature_type string
    ,start_p bigint
    ,end_p BIGINT
    ,origin_db string
    ,priority BIGINT
)
;

CREATE FUNCTION interpro_db_priority AS 'interpro_db_priority_udf.interpro_db_priority' USING 'interpro_db_priority_udf.py' -f ;

SET odps.stage.mapper.mem=12288;
--set odps.stage.mapper.split.size=128;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;

INSERT OVERWRITE TABLE luca_data2.tmp_lucaone_v2_interpro_db_priority
SELECT prot_seq_accession
        ,t1.interpro_accession as interpro_accession
        ,original_accession
        ,interpro_feature_name
        ,interpro_feature_type
        , case when interpro_feature_type in ('Homologous_superfamily') then 'Homologous_superfamily'
               when interpro_feature_type in ('Active_site', 'Binding_site', 'Conserved_site', 'PTM') then 'Site'
               when interpro_feature_type in ('Domain') then 'Domain'
               else interpro_feature_type
        end as alg_feature_type
        ,start_p
        ,end_p
        ,origin_db
        ,CAST(priority AS BIGINT) AS priority
FROM    (
            SELECT  prot_seq_accession
                    ,interpro_accession
                    ,interpro_feature_name
                    ,original_accession
                    ,start_p
                    ,end_p
                    ,SPLIT_PART(origin_db_priority,';',1) AS origin_db
                    ,SPLIT_PART(origin_db_priority,';',2) AS priority
            FROM    (
                        SELECT  distinct
                                prot_seq_accession
                                ,interpro_accession
                                ,interpro_feature_name
                                ,original_accession
                                ,start_p
                                ,end_p
                                ,interpro_db_priority(original_accession) AS origin_db_priority
                        FROM    luca_data2.tmp_lucaone_v2_interpro_fea_info
                    ) tmp
        ) t1
LEFT OUTER JOIN (
                    SELECT  DISTINCT interpro_accession
                            ,interpro_feature_type
                    FROM    tmp_lucaone_v2_interpro_entry_info
                ) t2
ON      t1.interpro_accession = t2.interpro_accession
;



-- 每一种alg_feature_type，选择db最高优先级的db
DROP TABLE IF EXISTS tmp_lucaone_v2_interpro_db_priority_selected_db
;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_interpro_db_priority_selected_db AS
SELECT  distinct prot_seq_accession, alg_feature_type, origin_db
FROM    (
            SELECT  *
                    ,ROW_NUMBER() OVER (PARTITION BY prot_seq_accession,alg_feature_type ORDER BY priority ASC ) AS rank_id
            FROM    tmp_lucaone_v2_interpro_db_priority
        ) t
WHERE   rank_id = 1
;

-- 选择
DROP TABLE IF EXISTS tmp_lucaone_v2_interpro_db_priority_selected
;
CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_interpro_db_priority_selected AS
SELECT  t1.*
FROM   tmp_lucaone_v2_interpro_db_priority t1
join tmp_lucaone_v2_interpro_db_priority_selected_db t2
where t1.prot_seq_accession = t2.prot_seq_accession and t1.alg_feature_type = t2.alg_feature_type and t1.origin_db = t2.origin_db
;
