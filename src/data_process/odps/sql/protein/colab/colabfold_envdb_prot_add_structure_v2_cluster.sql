--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-19 17:23:01
--********************************************************************--
CREATE FUNCTION prot_structure_add AS 'prot_structure_add_udf.prot_structure_add' USING 'prot_structure_add_udf.py,biopython-1.80-cp37.zip' -f ;

SET odps.stage.mapper.mem=12288;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;


DROP TABLE IF EXISTS tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster_final ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster_final AS
select seq_id as obj_id, "prot" as obj_type, seq_clean(seq) as obj_seq , prot_structure_add(labels, coord_list) as obj_label
from tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster tt1
         left join (
    SELECT protein_id, coord_list
    FROM (
             SELECT protein_id, coord_list, ROW_NUMBER()over(PARTITION BY protein_id ORDER BY priority DESC) AS rn
             FROM (
                      SELECT protein_id, coord_list, 1 AS priority
                      FROM (
                               SELECT protein_id,
                                      REPLACE(REPLACE(coord_list,'-1,','[-100,-100,-100],'),'-1]','[-100,-100,-100]]') AS coord_list,
                                      ROW_NUMBER()over(PARTITION BY protein_id ORDER BY score DESC ) AS rn
                               FROM tmp_lucaone_v2_prot_structure_info
                               WHERE source = 'pdb'
                           ) a
                      WHERE rn = 1
                      UNION ALL
                      SELECT protein_id, coord_list AS coord_list, 0 AS priority
                      FROM (
                               SELECT protein_id,
                                      REPLACE(REPLACE(coord_list,'-1,','[-100,-100,-100],'),'-1]','[-100,-100,-100]]') AS coord_list
                               FROM tmp_lucaone_v2_prot_structure_info
                               WHERE source = 'alphafold'
                           )b
                  ) t1
         ) t2
    WHERE rn = 1
) tt2
                   on seq_id = protein_id;

inputs:
        luca_data2.tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster: 208966064 (21297677059 bytes)
        luca_data2.tmp_lucaone_v2_prot_structure_info: 1089354 (6321454249 bytes)
outputs:
        luca_data2.tmp_lucaone_v2_colabfold_envdb_all_detail_v2_cluster_final: 208966064 (20942904644 bytes)