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
    with gzip.open(filepath, "rt") as rfp:
        for rec in SeqIO.parse(rfp, "genbank"):
            t_nucleo_fasta = [rec.id, str(rec.seq)]
            t_nucleo_fasta_all.append(t_nucleo_fasta)
    return t_nucleo_fasta_all


def write_chunk(chunk_t_fasta, chunk_idx, save_dir, prefix=""):
    for dirname in ["fasta"]:
        dirpath = os.path.join(save_dir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    csv_writer(
        chunk_t_fasta,
        os.path.join(save_dir, "fasta", f"{prefix}_fasta_{chunk_idx}.csv"),
        header=["seq_id", "seq"]
    )


def get_not_exists_assembly_seq_ids(filepath):
    assembly_accession_set = set()
    assembly_accession_seq_id_set = []
    for row in csv_reader(filepath, header=True, header_filter=True):
        assembly = row[0].strip()
        seq_id = row[1].strip()
        assembly_accession_seq_id_set.append([assembly, seq_id])
        assembly_accession_set.add(assembly)
    print("not exists assembly_accession: %d" % len(assembly_accession_set))
    print("not exists assembly_accession_seq_ids: %d" % len(assembly_accession_seq_id_set))
    return assembly_accession_set, assembly_accession_seq_id_set


def parse(dirpath, save_dir, prefix, chunk_size=1000000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    err_wfp = open(os.path.join(save_dir, "err_parse_fasta_nonexists.txt"), "w")
    not_exists_assembly_set, not_exists_assembly_seq_id_set = get_not_exists_assembly_seq_ids("./data/dna_fasta_not_exists_assembly_seq_ids.csv")
    chunk_t_nucleo_fasta = []
    dir_num = 0
    seq_cnt = 0
    total_cnt = 0
    chunk_idx = 0
    total_fasta_cnt = 0
    for dirname in os.listdir(dirpath):
        file_dir = os.path.join(dirpath, dirname)
        yes = False
        for assembly_no in not_exists_assembly_set:
            if assembly_no in dirname:
                yes = True
                break
        if not yes:
            continue
        print("dir name: %s" % dirname)
        filepath_list = set()
        for filename in os.listdir(file_dir):
            if "genomic.gbff" in filename:
                filepath = os.path.join(file_dir, filename)
                filepath_list.add(filepath)
        if len(filepath_list) == 0:
            err_wfp.write("%s not contains genomic.gbff file\n" % dirname)
            err_wfp.flush()
            continue

        dir_num += 1
        for filepath in filepath_list:
            fname = filepath.split("/")[-1]
            assembly = re.findall(r"^(GCF_[0-9]+\.?[0-9]*)_.+$", fname).pop()
            '''
            只需要目标的assembly
            '''
            if assembly not in not_exists_assembly_set:
                continue
            print("process filenames: %s" % filepath)
            try:
                cur_nucleo_fasta_all = parse_one(filepath)
                if len(cur_nucleo_fasta_all) == 0:
                    err_wfp.write("%s not contains nucleo sequence \n" % assembly)
                    err_wfp.flush()
                total_fasta_cnt += len(cur_nucleo_fasta_all)
                chunk_t_nucleo_fasta.extend(cur_nucleo_fasta_all)

                cur_size = len(cur_nucleo_fasta_all)
                seq_cnt += cur_size
                total_cnt += cur_size
                # print(seq_cnt)
                if seq_cnt >= chunk_size:
                    chunk_idx += 1
                    write_chunk(chunk_t_nucleo_fasta, chunk_idx, save_dir, prefix=prefix)
                    chunk_t_nucleo_fasta = []
                    seq_cnt = 0
            except Exception as e:
                print(e)
                print(filepath, seq_cnt, e)
                err_wfp.write("%s,%d,%s\n" % (filepath, seq_cnt, e))
                err_wfp.flush()
    print("total_fasta_cnt: %d" % total_fasta_cnt)
    print("dir_num: %d" % dir_num)
    print("total: %d" % total_cnt)
    err_wfp.close()
    if len(chunk_t_nucleo_fasta) > 0:
        chunk_idx += 1
        write_chunk(chunk_t_nucleo_fasta, chunk_idx, save_dir, prefix=prefix)


parse("/mnt2/yanyan/refseq_data/",
      "/mnt3/sanyuan.hy/data/refseq/fasta_parsed_nonexists/",
      "refseq_append",
      chunk_size=100000
      )

'''
total_fasta_cnt: 279
dir_num: 1
total: 279

create table if not exists luca_data2.tmp_lucaone_v2_refseq_fasta_info_100w_nonexists(
    seq_id string, 
    seq string
);

tunnel upload /mnt3/sanyuan.hy/data/refseq/fasta_parsed_nonexists/fasta/ luca_data2.tmp_lucaone_v2_refseq_fasta_info_100w_nonexists -cf true -fd "," -rd "\n" -h true;
'''