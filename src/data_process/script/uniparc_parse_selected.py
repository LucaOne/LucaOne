#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/25 13:52
@project: LucaOne
@file: uniprot_parse_selected.py
@desc: uniprot parse selected
'''
import gzip
import io, os, csv
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqFeature import *
from xml.dom.minidom import parseString


def dict_to_row(item, cols):
    row = []
    for col in cols:
        row.append(item[col])
    return row


def csv_writer(dataset, handle, header):
    '''
    csv 写，适合大文件
    :param dataset: 数据
    :param handle: 文件
    :param header: 头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'w')
    try:
        writer = csv.writer(handle)
        if header:
            if isinstance(header, str):
                header = [header]
            writer.writerow(header)
        for row in dataset:
            if isinstance(row, dict):
                row = dict_to_row(row, header)
            elif isinstance(row, str):
                row = [row]
            writer.writerow(row)
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()


def write_fasta(filepath, sequences):
    '''
    write fasta file
    :param filepath: savepath
    :param sequences: fasta sequence(each item: [id, seq])
    :return:
    '''

    if sequences:
        with open(filepath, "w") as output_handle:
            if len(sequences[0]) > 1 and isinstance(sequences[0][0], str):
                for row in sequences:
                    protein_id = row[0]
                    seq = row[1]
                    sequence = SeqRecord(Seq(seq, None), id=protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id, description="")
                    SeqIO.write(sequence, output_handle, "fasta")
            else:
                for sequence in sequences:
                    SeqIO.write(sequence, output_handle, "fasta")


def gunzip(filepath, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filepath, 'rb') as pr, open(os.path.join(save_dir, os.path.basename(filepath).replace(".gz", "")), 'wb') as pw:
        pw.write(gzip.decompress(pr.read()))


def get_one_entry(handle):
    try:
        handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
        comment = ""
        for line in handle:
            row = line.strip()
            if len(row) == 0:
                continue
            if row.startswith("<entry"):
                comment = row
            elif row == "</entry>":
                comment += row
                xml_tree = parseString(comment)
                comment = ""
                yield xml_tree
            elif len(comment) > 0:
                comment += row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def csv_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle)
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def load_selected_ids(filepath):
    selected_ids = set()
    for row in csv_reader(filepath, header=True, header_filter=True):
        seq_id = row[0].strip()
        selected_ids.add(seq_id)
    print("selected_ids: %d" % len(selected_ids))
    return selected_ids


def parse_one(filepath, selected_ids, db_ref_idx, property_idx, time_str,
              chunk_size=1000000, chunk_idx=0, save_dir=None, prefix=None):
    # uniparc_fasta, 一个entry一行
    t_uniparc_fasta = []

    # uniparc_xref，一个entry一行
    t_uniparc_xref = []

    # uniparc_XRef_2_Property， 一个entry多行
    t_uniparc_xref_2_property = []

    # uniparc_domain， 一个entry多行
    t_uniparc_domain = []
    delete_flag = False
    if filepath.endswith(".gz"):
        delete_flag = True
        gunzip(filepath, os.path.join(os.path.dirname(filepath), "gunzip"))
        filepath = os.path.join(os.path.dirname(filepath), "gunzip", os.path.basename(filepath).replace(".gz", ""))

    cnt = 0
    for dom_tree in get_one_entry(filepath):
        entry = dom_tree.documentElement
        children = entry.childNodes
        cur_uniparc_xref = []
        cur_uniparc_xref_2_property = []
        cur_uniparc_domain = []
        flag = False
        for child in children:
            if child.nodeName == "accession":
                uniparc_id = child.firstChild.nodeValue
                if uniparc_id not in selected_ids:
                    flag = True
                    break
            elif child.nodeName == "dbReference":
                db_ref_idx += 1
                db_type = child.getAttribute('type')
                db_id = child.getAttribute('id')
                version_i = child.getAttribute('version_i')
                active = child.getAttribute('active')
                version = child.getAttribute('version')
                created = child.getAttribute('created')
                last = child.getAttribute('last')
                child_children = child.childNodes
                cur_uniparc_xref.append([None, db_ref_idx, db_type, db_id, version_i, active, version, created, last])
                for child_child in child_children:
                    property_idx += 1
                    property_type = child_child.getAttribute('type')
                    property_value = child_child.getAttribute('value')
                    cur_uniparc_xref_2_property.append([None, db_ref_idx, property_idx, property_type, property_value])
            elif child.nodeName == "signatureSequenceMatch":
                database = child.getAttribute('database')
                database_id = child.getAttribute('id')
                interpro_id = None
                interpro_name = None
                domain_start = None
                domain_end = None
                child_children = child.childNodes
                for child_child in child_children:
                    if child_child.nodeName == "ipr":
                        interpro_id = child_child.getAttribute('id')
                        interpro_name = child_child.getAttribute('name')
                    elif child_child.nodeName == "lcn":
                        domain_start = int(child_child.getAttribute('start'))
                        domain_end = int(child_child.getAttribute('end'))
                cur_uniparc_domain.append([None, database, database_id, interpro_id, interpro_name, domain_start, domain_end])
            elif child.nodeName == "sequence":
                seq = child.firstChild.nodeValue
                seq_len = child.getAttribute('length')
                seq_checksum = child.getAttribute('checksum')
            else:
                print(child.nodeName)
        if flag:
            continue
        if uniparc_id not in selected_ids:
            continue
        t_uniparc_fasta.append([uniparc_id, seq_len, seq, seq_checksum, time_str])
        new_cur_uniparc_xref = []
        for item in cur_uniparc_xref:
            new_cur_uniparc_xref.append([uniparc_id, *item[1:], time_str])
        cur_uniparc_xref = new_cur_uniparc_xref
        new_cur_uniparc_xref_2_property = []
        for item in cur_uniparc_xref_2_property:
            new_cur_uniparc_xref_2_property.append([uniparc_id, *item[1:], time_str])
        cur_uniparc_xref_2_property = new_cur_uniparc_xref_2_property
        new_cur_uniparc_domain = []
        for item in cur_uniparc_domain:
            new_cur_uniparc_domain.append([uniparc_id, *item[1:], time_str])
        cur_uniparc_domain = new_cur_uniparc_domain
        t_uniparc_xref.extend(cur_uniparc_xref)
        t_uniparc_xref_2_property.extend(cur_uniparc_xref_2_property)
        t_uniparc_domain.extend(cur_uniparc_domain)
        cnt += 1
        if len(t_uniparc_fasta) % chunk_size == 0:
            chunk_idx += 1
            write_chunk(t_uniparc_fasta, t_uniparc_xref, t_uniparc_xref_2_property, t_uniparc_domain,
                        chunk_idx, save_dir, prefix=prefix)
            # uniparc_fasta, 一个entry一行
            t_uniparc_fasta = []

            # uniparc_xref，一个entry一行
            t_uniparc_xref = []

            # uniparc_XRef_2_Property， 一个entry多行
            t_uniparc_xref_2_property = []

            # uniparc_domain， 一个entry多行
            t_uniparc_domain = []

    if len(t_uniparc_fasta) > 0:
        chunk_idx += 1
        write_chunk(t_uniparc_fasta, t_uniparc_xref, t_uniparc_xref_2_property, t_uniparc_domain,
                    chunk_idx, save_dir, prefix=prefix)
    if delete_flag and os.path.exists(filepath):
        os.remove(filepath)
    return cnt, chunk_idx, db_ref_idx, property_idx


def write_chunk(chunk_t_uniparc_fasta, chunk_t_uniparc_xref, chunk_t_uniparc_xref_2_property, chunk_t_uniparc_domain, chunk_idx, save_dir, prefix=""):
    for dir_name in ["fasta", "xref", "xref_2_property", "domain"]:
        dirpath = os.path.join(save_dir, dir_name)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    csv_writer(
        chunk_t_uniparc_fasta,
        os.path.join(save_dir, "fasta", f"{prefix}_fasta_{chunk_idx}.csv"),
        header=["uniparc_id", "seq_len", "seq", "seq_checksum", "insert_date"]
    )

    csv_writer(
        chunk_t_uniparc_xref,
        os.path.join(save_dir, "xref", f"{prefix}_xref_{chunk_idx}.csv"),
        header=["uniparc_id", "db_ref_idx", "db_type", "db_id", "version_i", "active", "version", "created", "last", "insert_date"]
    )

    csv_writer(
        chunk_t_uniparc_xref_2_property,
        os.path.join(save_dir, "xref_2_property", f"{prefix}_xref_2_property_{chunk_idx}.csv"),
        header=["uniparc_id", "db_ref_idx", "property_idx", "pproperty_type", "property_value", "insert_date"]
    )

    csv_writer(
        chunk_t_uniparc_domain,
        os.path.join(save_dir, "domain", f"{prefix}_domain_{chunk_idx}.csv"),
        header=["uniparc_id", "database", "database_id", "interpro_id", "interpro_name", "domain_start", "domain_end", "insert_date"]
    )


def parse(filedir, save_dir, prefix, chunk_size=1000000, part_range="1_10", selected_ids_filepath=None):
    chunk_idx = 0
    seq_cnt = 0
    db_ref_idx, property_idx = 0, 0
    import time
    time_str = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    filename_list = os.listdir(filedir)
    filename_list = sorted(filename_list)
    selected_ids = load_selected_ids(selected_ids_filepath)
    print("selected_ids: %d" % len(selected_ids))
    part_ranges = part_range.split("_")
    prefix = prefix + "_" + part_range
    for filename in filename_list:
        if ".xml." not in filename:
            continue
        num = filename.split("_")[-1].replace(".xml.gz", "")
        if int(num[1:]) < int(part_ranges[0]) or int(num[1:]) > int(part_ranges[1]):
            continue
        print("parse filename: %s" % filename)
        cur_size, chunk_idx, db_ref_idx, property_idx = parse_one(os.path.join(filedir, filename),
                                                                  selected_ids,
                                                                  db_ref_idx, property_idx, time_str,
                                                                  chunk_size=chunk_size, chunk_idx=chunk_idx, save_dir=save_dir, prefix=prefix)
        print("cur_size=", cur_size, "db_ref_idx=", db_ref_idx, "property_idx=", property_idx)
        seq_cnt += cur_size

    print("total: %d" % seq_cnt)


if __name__ == "__main__":
    '''
    http://47.89.241.130:8880/terminals/4_15
    '''
    '''
    parse(
        "/backup/uniparc/p1-50",
        "/backup/uniparc/colabfold_envdb_selected_uniparc_parsed",
        "uniparc",
        chunk_size=1000000,
        part_range="1_10",
        selected_ids_filepath="data/tmp_lucaone_v2_colabfold_envdb_selected_uniparc_ids.csv"
    )
    161-170 done 171-180 done 181-190 done
    '''
    '''
    http://47.114.61.66:8880/terminals/2-6
    '''
    parse(
        "/mnt2/workspace/data/uniparc/p1-200",
        "/mnt2/workspace/data/uniparc/uniref_selected_uniparc_parsed_2",
        "uniparc",
        chunk_size=1000000,
        part_range="1_200",
        selected_ids_filepath="data/tmp_lucaone_v2_uniref_selected_uniparc_ids_2.csv"
    )

'''

tunnel download tmp_lucaone_v2_colabfold_envdb_selected_uniparc_ids /bio/sanyuan.hy/data/tmp_lucaone_v2_colabfold_envdb_selected_uniparc_ids.csv -cf true -fd "," -rd "\n" -h true;
tunnel download tmp_lucaone_v2_colabfold_envdb_selected_uniprot_ids /bio/sanyuan.hy/data/tmp_lucaone_v2_colabfold_envdb_selected_uniprot_ids.csv -cf true -fd "," -rd "\n" -h true;

create table if not exists tmp_lucaone_v2_uniref_selected_uniparc_ids(
    uniparc_id string
);
tunnel upload /mnt2/workspace/data/uniref_selected_uniparc_ids.csv tmp_lucaone_v2_uniref_selected_uniparc_ids -cf true -fd "," -rd "\n" -h true;

# http://47.114.61.66:8880/terminals/12
tunnel download tmp_lucaone_v2_uniref_selected_uniparc_ids_2 /mnt2/workspace/data/tmp_lucaone_v2_uniref_selected_uniparc_ids_2.csv -cf true -fd "," -rd "\n" -h true;


create table if not exists luca_data2.tmp_lucaone_v2_prot_fasta_info_uniref_uniparc(
    uniparc_id string,
    seq_len bigint,
    seq string,
    seq_checksum string,
    insert_date string
);

tunnel upload /mnt3/sanyuan.hy/data/uniprot/uniparc_parsed/fasta luca_data2.tmp_lucaone_v2_prot_fasta_info_uniref_uniparc -cf true -fd "," -rd "\n" -h true;

#
create table if not exists luca_data2.tmp_lucaone_v2_prot_xref_info_uniref_uniparc(
    uniparc_id string,
    db_ref_idx bigint,
    db_type string,
    db_id string,
    version_i string,
    active string,
    version string,
    created string,
    last string,
    insert_date string
);

tunnel upload /mnt3/sanyuan.hy/data/uniprot/uniparc_parsed/xref luca_data2.tmp_lucaone_v2_prot_xref_info_uniref_uniparc -cf true -fd "," -rd "\n" -h true;


#
create table if not exists luca_data2.tmp_lucaone_v2_prot_xref_2_property_info_uniref_uniparc(
    uniparc_id string,
    db_ref_idx bigint,
    property_idx bigint,
    property_type string,
    property_value string,
    insert_date string
);

tunnel upload /mnt3/sanyuan.hy/data/uniprot/uniparc_parsed/xref_2_property luca_data2.tmp_lucaone_v2_prot_xref_2_property_info_uniref_uniparc -cf true -fd "," -rd "\n" -h true;

# 
create table if not exists luca_data2.tmp_lucaone_v2_prot_domain_info_uniref_uniparc(
    uniparc_id string,
    database string,
    database_id string,
    interpro_id string,
    interpro_name string,
    domain_start bigint,
    domain_end bigint,
    insert_date string
);

tunnel upload /mnt3/sanyuan.hy/data/uniprot/uniparc_parsed/domain luca_data2.tmp_lucaone_v2_prot_domain_info_uniref_uniparc -cf true -fd "," -rd "\n" -h true;

##### colabfold_envdb

create table if not exists luca_data2.tmp_lucaone_v2_prot_fasta_info_colabfold_envdb_uniparc(
    uniparc_id string,
    seq_len bigint,
    seq string,
    seq_checksum string,
    insert_date string
);

tunnel upload /mnt2/workspace/data/uniparc/colabfold_envdb_selected_uniparc_parsed_all/fasta luca_data2.tmp_lucaone_v2_prot_fasta_info_colabfold_envdb_uniparc -cf true -fd "," -rd "\n" -h true;

#
create table if not exists luca_data2.tmp_lucaone_v2_prot_xref_info_colabfold_envdb_uniparc(
    uniparc_id string,
    db_ref_idx bigint,
    db_type string,
    db_id string,
    version_i string,
    active string,
    version string,
    created string,
    last string,
    insert_date string
);

tunnel upload /mnt2/workspace/data/uniparc/colabfold_envdb_selected_uniparc_parsed_all/xref luca_data2.tmp_lucaone_v2_prot_xref_info_colabfold_envdb_uniparc -cf true -fd "," -rd "\n" -h true;


#
create table if not exists luca_data2.tmp_lucaone_v2_prot_xref_2_property_info_colabfold_envdb_uniparc(
    uniparc_id string,
    db_ref_idx bigint,
    property_idx bigint,
    property_type string,
    property_value string,
    insert_date string
);

tunnel upload /mnt2/workspace/data/uniparc/colabfold_envdb_selected_uniparc_parsed_all/xref_2_property luca_data2.tmp_lucaone_v2_prot_xref_2_property_info_colabfold_envdb_uniparc -cf true -fd "," -rd "\n" -h true;

# 
create table if not exists luca_data2.tmp_lucaone_v2_prot_domain_info_colabfold_envdb_uniparc(
    uniparc_id string,
    database string,
    database_id string,
    interpro_id string,
    interpro_name string,
    domain_start bigint,
    domain_end bigint,
    insert_date string
);

tunnel upload /mnt2/workspace/data/uniparc/colabfold_envdb_selected_uniparc_parsed_all/domain luca_data2.tmp_lucaone_v2_prot_domain_info_colabfold_envdb_uniparc -cf true -fd "," -rd "\n" -h true;

'''