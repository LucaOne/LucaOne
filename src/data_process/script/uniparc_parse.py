#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/25 13:52
@project: LucaOne
@file: uniprot_parse.py
@desc: uniprot parse
'''
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
import xml


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


def fasta_reader(handle, width=None):
    """
    Reads a FASTA file, yielding header, sequence pairs for each sequence recovered 适合大文件
    args:
        :handle (str, pathliob.Path, or file pointer) - fasta to read from
        :width (int or None) - formats the sequence to have max `width` character per line.
                               If <= 0, processed as None. If None, there is no max width.
    yields:
        :(header, sequence) tuples
    returns:
        :None
    """
    FASTA_STOP_CODON = "*"

    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    width = width if isinstance(width, int) and width > 0 else None
    try:
        for is_header, group in itertools.groupby(handle, lambda line: line.startswith(">")):
            if is_header:
                header = group.__next__().strip()
            else:
                seq = ''.join(line.strip() for line in group).strip().rstrip(FASTA_STOP_CODON)
                if width is not None:
                    seq = textwrap.fill(seq, width)
                yield header, seq
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


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


def parse_one(filepath, db_ref_idx, property_idx, time_str):
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
        '''
        infile = gzip.open(filepath)
        content = infile.read()
        # parse xml file content
        dom_tree = xml.dom.minidom.parseString(content)
        '''
        delete_flag = True
        gunzip(filepath, os.path.join(os.path.dirname(filepath), "gunzip"))
        filepath = os.path.join(os.path.dirname(filepath), "gunzip", os.path.basename(filepath).replace(".gz", ""))
    dom_tree = xml.dom.minidom.parse(filepath)
    collection = dom_tree.documentElement
    entry_list = collection.getElementsByTagName("entry")

    for entry in entry_list:
        children = entry.childNodes
        cur_uniparc_xref = []
        cur_uniparc_xref_2_property = []
        cur_uniparc_domain = []
        for child in children:
            if child.nodeName == "accession":
                uniparc_id = child.firstChild.nodeValue
                # print(["uniparc_id", uniparc_id])
            elif child.nodeName == "dbReference":
                db_ref_idx += 1
                db_type = child.getAttribute('type')
                db_id = child.getAttribute('id')
                version_i = child.getAttribute('version_i')
                active = child.getAttribute('active')
                version = child.getAttribute('version')
                created = child.getAttribute('created')
                last = child.getAttribute('last')
                # print(["xref", db_ref_idx, db_type, db_id, version_i, active, version, created, last])
                cur_uniparc_xref.append([None, db_ref_idx, db_type, db_id, version_i, active, version, created, last])
                child_children = child.childNodes
                for child_child in child_children:
                    property_idx += 1
                    property_type = child_child.getAttribute('type')
                    property_value = child_child.getAttribute('value')
                    cur_uniparc_xref_2_property.append([None, db_ref_idx, property_idx, property_type, property_value])
                    # print(["xref_2_property", db_ref_idx, property_idx, property_type, property_value])
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
                        domain_start = child_child.getAttribute('start')
                        domain_end = child_child.getAttribute('end')
                # print(["domain", database, database_id, interpro_id, interpro_name, domain_start, domain_end])
                cur_uniparc_domain.append([None, database, database_id, interpro_id, interpro_name, domain_start, domain_end])
            elif child.nodeName == "sequence":
                seq = child.firstChild.nodeValue
                seq_len = child.getAttribute('length')
                seq_checksum = child.getAttribute('checksum')
            else:
                print(child.nodeName)
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

    if delete_flag:
        os.remove(filepath)
    return t_uniparc_fasta, t_uniparc_xref, t_uniparc_xref_2_property, t_uniparc_domain, db_ref_idx, property_idx


def write_chunk(chunk_t_uniparc_fasta, chunk_t_uniparc_xref, chunk_t_uniparc_xref_2_property, chunk_t_uniparc_domain, chunk_idx, save_dir, prefix=""):
    for dirname in ["fasta", "xref", "xref_2_property", "domain"]:
        dirpath = os.path.join(save_dir, dirname)
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


def parse(filedir, save_dir, prefix, chunk_size=1000000):
    chunk_t_uniparc_fasta = []
    chunk_t_uniparc_xref = []
    chunk_t_uniparc_xref_2_property = []
    chunk_t_uniparc_domain = []
    chunk_idx = 1
    seq_cnt = 0
    cur_seq_cnt = 0
    db_ref_idx, property_idx = 0, 0
    import time
    time_str = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    filename_list = os.listdir(filedir)
    filename_list = sorted(filename_list)
    for filename in filename_list:
        if ".xml." not in filename:
            continue
        num = filename.split("_")[-1].replace(".xml.gz", "")
        if int(num[1:]) < 101:
            continue
        try:
            t_uniparc_fasta, t_uniparc_xref, t_uniparc_xref_2_property, t_uniparc_domain, db_ref_idx, property_idx\
                = parse_one(os.path.join(filedir, filename), db_ref_idx, property_idx, time_str)
            chunk_t_uniparc_fasta.append(t_uniparc_fasta)
            chunk_t_uniparc_xref.append(t_uniparc_xref)
            chunk_t_uniparc_xref_2_property.extend(t_uniparc_xref_2_property)
            chunk_t_uniparc_domain.extend(t_uniparc_domain)
            seq_cnt += len(t_uniparc_fasta)
            cur_seq_cnt += len(t_uniparc_fasta)
            print(db_ref_idx, property_idx)

            if cur_seq_cnt >= chunk_size:
                write_chunk(chunk_t_uniparc_fasta, chunk_t_uniparc_xref, chunk_t_uniparc_xref_2_property, chunk_t_uniparc_domain,
                            chunk_idx, save_dir, prefix=prefix)
                chunk_t_uniparc_fasta = []
                chunk_t_uniparc_xref = []
                chunk_t_uniparc_xref_2_property = []
                chunk_t_uniparc_domain = []
                chunk_idx += 1
                cur_seq_cnt = 0

        except StopIteration:
            break
        except Exception as e:
            print(filename, seq_cnt, e)
            pass
        print("total: %d" % seq_cnt)
        if len(chunk_t_uniparc_fasta) > 0:
            write_chunk(chunk_t_uniparc_fasta, chunk_t_uniparc_xref, chunk_t_uniparc_xref_2_property, chunk_t_uniparc_domain,
                        chunk_idx, save_dir, prefix=prefix)
            chunk_t_uniparc_fasta = []
            chunk_t_uniparc_xref = []
            chunk_t_uniparc_xref_2_property = []
            chunk_t_uniparc_domain = []
            chunk_idx += 1


parse(
    "/mnt3/sanyuan.hy/data/uniprot/uniparc",
    "/mnt3/sanyuan.hy/data/uniprot/uniparc_parsed",
    "uniparc",
    chunk_size=1000000
)