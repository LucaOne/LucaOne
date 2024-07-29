import os, sys
import csv, json
import argparse
import csv, os
import io, textwrap, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import json

from Bio import pairwise2 as pw2
from Bio.PDB.PDBParser import PDBParser
from Bio.Seq import Seq
from Bio import PDB

pdb_parser = PDBParser(PERMISSIVE=1)


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
        header = None
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


def read_fasta(fasta_filepath):
    fasta_info = {}
    # >sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-001R PE=4 SV=1
    for row in fasta_reader(fasta_filepath):
        seq_id = row[0].strip().split("|")[1]
        seq = row[1].upper().strip()
        fasta_info[seq_id] = seq
    return fasta_info


def read_structure_info(structure_mapping_filepth):
    structure_info = {}
    for row in csv_reader(structure_mapping_filepth, header_filter=True, header=True):
        protein_id, structure_filename, chain, source = row
        if protein_id not in structure_info:
            structure_info[protein_id] = {}
            structure_info[protein_id][source] = {}
            structure_info[protein_id][source][chain] = set()
        elif source not in structure_info[protein_id]:
            structure_info[protein_id][source] = {}
            structure_info[protein_id][source][chain] = set()
        elif chain not in structure_info[protein_id][source]:
            structure_info[protein_id][source][chain] = set()
        structure_info[protein_id][source][chain].add(structure_filename)
    return structure_info


def un_gz(filename):
    f_name = filename.replace(".gz", "")
    g_file = gzip.GzipFile(filename)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


aa_d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
            'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
            'SEC': 'U', 'PYL': 'O'}


def get_aa_coord(protein_seq, pdb_id, pdb_filepath, chain=None):
    global pdb_parser
    data = pdb_parser.get_structure(pdb_id, pdb_filepath)
    models = list(data.get_models())
    chains = list(models[0].get_chains())
    for chain_idx in range(len(chains)):
        cur_chain = chains[chain_idx].get_id()
        if chain is not None and chain != cur_chain:
            continue
        # print("chain_idx: %d, name: %s" % (chain_idx, chains[chain_idx].get_id()))
        residue_list = list(chains[chain_idx].get_residues())
        chain_seq = ""
        chain_xyz = []
        chain_xyz_seq = ""
        for res_idx, residue in enumerate(residue_list):
            full_id = residue.get_full_id()
            if full_id[3][0] != ' ':
                continue
            aa = aa_d3to1.get(residue.get_resname(), "X")
            chain_seq += aa
            atoms = list(residue.get_atoms())
            flag = False
            for atom in atoms:
                name = atom.get_name()
                # 使用CA的位置作为每个氨基酸的位置
                if name == "CA":
                    coord = atom.get_coord()
                    chain_xyz.append([coord[0], coord[1], coord[2]])
                    flag = True
            if flag:
                chain_xyz_seq += aa
        if len(chain_xyz_seq) < 1:
            print("pdb_filepath:", pdb_filepath)
            print("protein_seq:", protein_seq)
            print("chain_seq:", chain_seq)
            print("chain_xyz_seq:", chain_xyz_seq)
            return None, None
        # alignments = pw2.align.globalxx(Seq(protein_seq), Seq(chain_xyz_seq))
        alignments = pw2.align.globalms(Seq(protein_seq), Seq(chain_xyz_seq), 2, -1, -.5, -.1)
        protein_xyzs = []
        seqA = alignments[0].seqA
        seqB = alignments[0].seqB
        score = alignments[0].score
        xyz_idx = 0
        for ch_idx, ch in enumerate(seqA):
            if ch == "-":
                xyz_idx += 1
            elif seqB[ch_idx] == "-":
                protein_xyzs.append(-1)
            else:
                protein_xyzs.append(chain_xyz[xyz_idx])
                xyz_idx += 1
        return protein_xyzs, score
    return None, None

# http://47.93.21.181/
swiss_prot_fasta_filepath = "/mnt/sanyuan.hy/data/uniprot/swiss-prot/uniprot_sprot.fasta"
swiss_prot_fasta = read_fasta(swiss_prot_fasta_filepath)
print("swiss_prot_fasta size: %d" % len(swiss_prot_fasta))

trembl_fasta_filepath = "/mnt/sanyuan.hy/data/uniprot/TrEMBL/uniprot_trembl.fasta"
trembl_fasta = read_fasta(trembl_fasta_filepath)
print("trembl_fasta size: %d" % len(trembl_fasta))

structure_mapping_filepth = "/mnt/sanyuan.hy/data/uniprot/sanyuan_protein_structure_list.csv"
structure_info = read_structure_info(structure_mapping_filepth)
print("structure size: %d" % len(structure_info))

pdb_dirpath = "/mnt/sanyuan.hy/data/pdb/pdb"
alphafold_dirpath = "/mnt/sanyuan.hy/data/alphafold/pdb"
done = 0
had = 0
with open("/mnt/sanyuan.hy/data/uniprot/sanyuan_protein_structure_detail_info.csv", "w") as wfp1:
    with open("/mnt/sanyuan.hy/data/uniprot/sanyuan_protein_structure_info.csv", "w") as wfp2:
        writer1 = csv.writer(wfp1)
        writer2 = csv.writer(wfp2)
        writer1.writerow(["protein_id", "protein_db", "structure_id", "chain", "source", "score", "aa", "aa_idx" "coord_x", "coord_y", "coord_z"])
        writer2.writerow(["protein_id", "protein_db", "structure_id", "chain", "source", "score", "seq", "coord_list"])
        for item1 in structure_info.items():
            protein_id = item1[0]
            done += 1
            if protein_id in swiss_prot_fasta:
                aa_seq = swiss_prot_fasta[protein_id]
                protein_db = "swiss-prot"
            elif protein_id in trembl_fasta:
                aa_seq = trembl_fasta[protein_id]
                protein_db = "trembl"
            else:
                print("not exists protein: %s" % protein_id)
                continue
            if done % 10000 == 0:
                print("done: %d/%d" %(done, len(structure_info)))
            for item2 in item1[1].items():
                source = item2[0]
                for item3 in item2[1].items():
                    chain = item3[0]
                    for structure_filename in item3[1]:
                        if "pdb" == source:
                            structure_filepath = os.path.join(pdb_dirpath, structure_filename)
                            structure_id = structure_filename.split(".")[0].replace("pdb", "")
                        else:
                            structure_filepath = os.path.join(alphafold_dirpath, structure_filename)
                            chain = None
                            structure_id = structure_filename.split("-")[1]

                        new_structure_filepath = structure_filepath.replace(".gz", "")
                        if not os.path.exists(new_structure_filepath):
                            un_gz(structure_filepath)
                        structure_filepath = new_structure_filepath
                        # print(aa_seq, structure_id, structure_filepath, chain)
                        coord_list, score = get_aa_coord(aa_seq, structure_id, structure_filepath, chain=chain)
                        if coord_list is None or len(coord_list) < 1:
                            continue
                        new_coord_list = []
                        had += 1
                        for aa_idx, coord in enumerate(coord_list):
                            if coord == -1:
                                coord_x = None
                                coord_y = None
                                coord_z = None
                                new_coord_list.append(coord)
                            else:
                                coord_x, coord_y, coord_z = float(coord[0]), float(coord[1]), float(coord[2])
                                new_coord_list.append([coord_x, coord_y, coord_z])

                            writer1.writerow([protein_id, protein_db, structure_id, chain, source, score,
                                              aa_seq[aa_idx], aa_idx, coord_x, coord_y, coord_z])
                        writer2.writerow([
                            protein_id, protein_db, structure_id, chain, source, score, aa_seq, json.dumps(
                                new_coord_list,
                                ensure_ascii=False
                            )]
                        )
print("done: %d, had: %d" % (done, had))