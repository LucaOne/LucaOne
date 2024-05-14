import csv, os
import io, textwrap, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqFeature import *

import gzip
import re
from datetime import datetime
import time
import pymysql
import sys
import pandas as pd
import hashlib
import json


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
            writer.writerow(header)
        for row in dataset:
            if isinstance(row, dict):
                row = dict_to_row(row, header)
            writer.writerow(row)
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()


def parse_one(filepath):
    t_nucleo_fasta_all = []
    t_nucleo_sequence_all = []
    t_nucleo_feature_all = []
    with gzip.open(filepath, "rt") as rfp:
        for rec in SeqIO.parse(rfp, "genbank"):
            t_nucleo_sequence = {}

            t_nucleo_fasta = [rec.id, str(rec.seq)]

            t_nucleo_sequence["seq_id"] = rec.id
            t_nucleo_sequence["seq_name"] = rec.name
            t_nucleo_sequence["summary"] = rec.description
            t_nucleo_sequence["seq_len"] = len(rec)
            t_nucleo_sequence["dbxrefs"] = rec.dbxrefs
            t_nucleo_sequence["date"] = datetime.strptime(rec.annotations["date"], '%d-%b-%Y').strftime('%Y-%m-%d')
            t_nucleo_sequence["molecule_type"] = rec.annotations.get("molecule_type")
            t_nucleo_sequence["topology"] = rec.annotations.get("topology")
            t_nucleo_sequence["data_file_division"] = rec.annotations.get("data_file_division")
            t_nucleo_sequence["accessions"] = rec.annotations.get("accessions")
            t_nucleo_sequence["sequence_version"] = rec.annotations.get("sequence_version")
            t_nucleo_sequence["keywords"] = rec.annotations.get("keywords")
            t_nucleo_sequence["source"] = rec.annotations.get("source")
            t_nucleo_sequence["organism"] = rec.annotations.get("organism")
            t_nucleo_sequence["taxonomy"] = rec.annotations.get("taxonomy")
            t_nucleo_sequence["references"] = rec.annotations.get("references")
            t_nucleo_sequence["comment"] = rec.annotations.get("comment")
            t_nucleo_sequence["structured_comment"] = rec.annotations.get("structured_comment")
            t_nucleo_sequence["contig"] = rec.annotations.get("contig")
            insert_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t_nucleo_sequence["insert_date"] = insert_date

            seq_dbxrefs_dict = {}
            for dbxref in rec.dbxrefs:
                regex_match = re.match("^(.+):(.+)$", dbxref)
                if regex_match:
                    db_name, ref = regex_match.groups()
                    seq_dbxrefs_dict[db_name] = ref
            # t_nucleo_sequence["assembly_accession"] = dbxrefs_dict.get("Assembly")

            # t_nucleo_sequence["assembly_accession"] = re.findall(r"^/mnt/yanyan/refseq_data/(GCF_[0-9]+\.?[0-9]*)_.+$", fname).pop()
            fname = filepath.split("/")[-1]
            assembly = re.findall(r"^(GCF_[0-9]+\.?[0-9]*)_.+$", fname).pop()
            # print(fname, Assembly)
            '''
            if "Assembly" in seq_dbxrefs_dict:
                t_nucleo_sequence["assembly_accession"] = seq_dbxrefs_dict["Assembly"]
                if Assembly != t_nucleo_sequence["assembly_accession"]:
                    print(fname, Assembly, t_nucleo_sequence["assembly_accession"])
            else:
                t_nucleo_sequence["assembly_accession"] = Assembly
            '''
            t_nucleo_sequence["assembly_accession"] = assembly
            t_nucleo_sequence["biosample"] = seq_dbxrefs_dict.get("BioSample")
            t_nucleo_sequence["bioproject"] = seq_dbxrefs_dict.get("BioProject")

            for ft in rec.features:
                t_nucleo_feature = {}
                t_nucleo_feature["assembly_accession"] = t_nucleo_sequence["assembly_accession"]
                t_nucleo_feature["seq_id"] = rec.id
                t_nucleo_feature["feature_type"] = ft.type
                t_nucleo_feature["strand"] = ft.strand
                t_nucleo_feature["start_end"] = repr(ft.location)
                t_nucleo_feature["insert_date"] = insert_date

                partial_3end, partial_5end = False, False
                if type(ft.location.start) is BeforePosition:
                    partial_3end = True
                if type(ft.location.end) is AfterPosition:
                    partial_5end = True
                if partial_3end and partial_5end:
                    t_nucleo_feature["feature_complete"] = 35
                elif partial_3end:
                    t_nucleo_feature["feature_complete"] = 3
                elif partial_5end:
                    t_nucleo_feature["feature_complete"] = 5
                else:
                    t_nucleo_feature["feature_complete"] = 0

                qualifiers = {}
                for key in ft.qualifiers:
                    qualifiers[key] = ft.qualifiers[key]
                t_nucleo_feature["qualifiers"] = qualifiers if len(qualifiers) > 0 else None

                if ft.type == "source":
                    feature_dbxrefs_dict = {}
                    for dbxref in ft.qualifiers.get("db_xref"):
                        regex_match = re.match("^(.+):(.+)$", dbxref)
                        db_name, ref = regex_match.groups()
                        feature_dbxrefs_dict[db_name] = ref
                    t_nucleo_sequence["taxid"] = feature_dbxrefs_dict.get("taxon")

                t_nucleo_feature_all.append(t_nucleo_feature)
            if len(t_nucleo_fasta) > 0:
                t_nucleo_fasta_all.append(t_nucleo_fasta)
            if len(t_nucleo_sequence) > 0:
                t_nucleo_sequence_all.append(t_nucleo_sequence)
    return t_nucleo_fasta_all, t_nucleo_sequence_all, t_nucleo_feature_all


def write_chunk(chunk_t_fasta, chunk_t_nucleo_sequence, chunk_t_nucleo_features, chunk_idx, save_dir, prefix=""):
    for dirname in ["fasta", "seq_info", "fea_info"]:
        dirpath = os.path.join(save_dir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    csv_writer(
        chunk_t_nucleo_sequence,
        os.path.join(save_dir, "seq_info", f"{prefix}_seq_{chunk_idx}.csv"),
        header=["assembly_accession", "seq_id", "seq_name", "summary", "seq_len", "dbxrefs", "taxid",
                "molecule_type", "topology", "data_file_division", "accessions", "sequence_version", "keywords",
                "source", "organism", "taxonomy", "references", "comment",
                "structured_comment", "contig", "biosample", "bioproject", "date", "insert_date"]
    )
    csv_writer(
        chunk_t_fasta,
        os.path.join(save_dir, "fasta", f"{prefix}_fasta_{chunk_idx}.csv"),
        header=["seq_id", "seq"]
    )
    '''
    write_fasta(
        os.path.join(save_dir, "fasta", f"{prefix}_{chunk_idx}.fasta"),
        chunk_t_fasta
    )
    '''
    csv_writer(
        chunk_t_nucleo_features,
        os.path.join(save_dir, "fea_info", f"{prefix}_fea_{chunk_idx}.csv"),
        header=["assembly_accession", "seq_id", "feature_type", "strand",
                "start_end", "feature_complete", "qualifiers", "insert_date"]
    )


def parse(dirpath, save_dir, prefix, chunk_size=1000000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    err_wfp = open(os.path.join(save_dir, "err_parse_all.txt"), "w")

    chunk_t_nucleo_fasta = []
    chunk_t_nucleo_sequence = []
    chunk_t_nucleo_features = []
    dir_num = 0
    seq_cnt = 0
    total_cnt = 0
    chunk_idx = 0
    total_fasta_cnt = 0
    total_seq_cnt = 0
    total_fea_cnt = 0
    for dirname in os.listdir(dirpath):
        file_dir = os.path.join(dirpath, dirname)
        filepath_list = set()
        for filename in os.listdir(file_dir):
            if "genomic.gbff" in filename or "rna.gbff" in filename:
                filepath = os.path.join(file_dir, filename)
                filepath_list.add(filepath)
        if len(filepath_list) == 0:
            err_wfp.write("%s not contains gbff file\n" % dirname)
            err_wfp.flush()
            continue
        print("process filenames: %s" % filepath_list)
        dir_num += 1
        for filepath in filepath_list:
            try:
                cur_nucleo_fasta_all, cur_nucleo_sequence_all, cur_nucleo_feature_all = parse_one(filepath)
                total_fasta_cnt += len(cur_nucleo_fasta_all)
                total_seq_cnt += len(cur_nucleo_sequence_all)
                total_fea_cnt += len(cur_nucleo_feature_all)
                chunk_t_nucleo_fasta.extend(cur_nucleo_fasta_all)
                chunk_t_nucleo_sequence.extend(cur_nucleo_sequence_all)
                chunk_t_nucleo_features.extend(cur_nucleo_feature_all)

                cur_size = len(cur_nucleo_fasta_all)
                seq_cnt += cur_size
                total_cnt += cur_size
                # print(seq_cnt)
                if seq_cnt >= chunk_size:
                    chunk_idx += 1
                    write_chunk(chunk_t_nucleo_fasta, chunk_t_nucleo_sequence, chunk_t_nucleo_features,
                                chunk_idx, save_dir, prefix=prefix)
                    chunk_t_nucleo_fasta = []
                    chunk_t_nucleo_sequence = []
                    chunk_t_nucleo_features = []
                    seq_cnt = 0
            except Exception as e:
                print(e)
                print(filepath, seq_cnt, e)
                err_wfp.write("%s,%d,%s\n" % (filepath, seq_cnt, e))
                err_wfp.flush()
    print("total_fasta_cnt: %d" % total_fasta_cnt)
    print("total_seq_cnt: %d" % total_seq_cnt)
    print("total_fea_cnt: %d" % total_fea_cnt)
    print("dir_num: %d" % dir_num)
    print("total: %d" % total_cnt)
    err_wfp.close()
    if len(chunk_t_nucleo_fasta) > 0 or len(chunk_t_nucleo_sequence) > 0 or len(chunk_t_nucleo_features) > 0:
        chunk_idx += 1
        write_chunk(chunk_t_nucleo_fasta, chunk_t_nucleo_sequence, chunk_t_nucleo_features,
                    chunk_idx, save_dir, prefix=prefix)

parse("/mnt2/yanyan/refseq_data/",
      "/mnt3/sanyuan.hy/data/refseq/parsed_all/",
      "refseq",
      chunk_size=100000)