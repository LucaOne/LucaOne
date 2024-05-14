--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2023-11-19 17:38:16
--********************************************************************--

drop table if exists tmp_lucaone_v2_taxid_mapping_final;
create table if not exists tmp_lucaone_v2_taxid_mapping_final AS
select *
from(
        select taxid, order_bio, priority, ROW_NUMBER() OVER (PARTITION BY taxid ORDER BY priority ASC ) AS rank_id
        from
            (
                select taxid, order as order_bio, 1 as priority
                from tmp_lucaone_v2_taxid_mapping
                where order is not null
                union all
                SELECT taxid, order_bio, 2 as priority
                from(
                    select taxid, trim(SPLIT_PART(lineage, ',', 5)) as order_bio
                    from tmp_lucaone_v2_taxonomy_tree
                    )
                where order_bio is not null
            ) tmp
    ) t
where rank_id = 1;

