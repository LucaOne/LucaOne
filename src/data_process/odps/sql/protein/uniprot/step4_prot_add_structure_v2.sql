--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-10 12:40:21
--********************************************************************--


CREATE FUNCTION prot_structure_add AS 'prot_structure_add_udf.prot_structure_add' USING 'prot_structure_add_udf.py,biopython-1.80-cp37.zip' -f ;

SET odps.stage.mapper.mem=12288;

SET odps.stage.reducer.mem=12288;

SET odps.sql.python.version=cp37;

DROP TABLE IF EXISTS tmp_lucaone_v2_uniprot_all_detail_v2_final ;

CREATE TABLE IF NOT EXISTS tmp_lucaone_v2_uniprot_all_detail_v2_final AS
select seq_id as obj_id, "prot" as obj_type, seq_clean(seq) as obj_seq, prot_structure_add(labels, coord_list) as obj_label
from(
        select seq_id, seq, labels
        from tmp_lucaone_v2_uniprot_sprot_all_detail_v2
        union all
        select seq_id, seq, labels
        from tmp_lucaone_v2_uniprot_trembl_all_detail_v2
    )  tt1
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
