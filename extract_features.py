import pickle
import os
import argparse
import itertools
from collections import Counter
import pandas as pd
from Bio import SeqIO
import esm
import torch
# Project --> Terminal --> pip install fair-esm
from collections import OrderedDict
import numpy as np
from Bio import SeqIO
import pandas as pd



# 定义函数将txt文件转换为csv文件
def txt_to_csv(txt_file, csv_file):
    # 读取txt文件
    df = pd.read_csv(txt_file, sep='\t', header=None)
    # 将数据写入csv文件
    df.to_csv(csv_file, index=False, header=True)


def fastaTOcsv(fasta_file, csv_file):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        header = record.description
        subcellular_localization = header.split('_')[-1]
        data.append({'Sequence': sequence, 'Subcellular_Localization': subcellular_localization})
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea = []
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string = string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip()
            string = string.replace(" ", "")

    fea.append(string)
    f.close()
    return fea


def read_fasta(filename):
    sequences = {}
    with open(filename, 'r') as file:
        current_id = None
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = current_sequence
                current_id = line[1:]
                current_sequence = ''
            else:
                current_sequence += line
        if current_id:
            sequences[current_id] = current_sequence
    return sequences

def write_pro_seq(sequences, output_file):
    with open(output_file, 'w') as file:
        for i, (protein, sequence) in enumerate(sequences.items(), start=1):
            file.write(f"pro{i}, {sequence}\n")

def extract_ESM_features(input_file_name, save_file_name):
    esm_embeddings_dim = 320
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet("./ESM_models/esm2_t6_8M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    fw = open(current_path+"/features/"+save_file_name, "w")

    with open(current_path+'/data/'+input_file_name, 'r') as fr_profastas:
        readlines = fr_profastas.readlines()
    for idx, eachfasta in enumerate(readlines):
        thisline = readlines[idx].strip()
        thisinfo_list = thisline.split(",")
        thisPro_id = thisinfo_list[0]
        thisFasta = thisinfo_list[1]
        thisTemp = [(thisPro_id, thisFasta)]
        batch_labels, batch_strs, batch_tokens = batch_converter(thisTemp)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]
        fea_embeddings = []
        fea_embeddings = torch.FloatTensor(fea_embeddings).view(-1, 320)

        for idx, tokens_len in enumerate(batch_lens):
            esm_representation = token_representations[idx, 1: tokens_len - 1]
            fea_embeddings = torch.cat([fea_embeddings, esm_representation], 0)
        selected_data = fea_embeddings[:33, :]
        flatten_data = selected_data.view(-1)
        flatten_data_np = flatten_data.numpy()
        reshaped_array = flatten_data_np.reshape(1, -1)
        flatten_array_ = reshaped_array.flatten()
        np.savetxt(fw, [flatten_array_], fmt='%f', delimiter=',')
    fw.close()


def extract_sequences(fasta_file):
    sequences = []
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_name = record.id
            sequence = str(record.seq)
            sequences.append((seq_name, sequence))
    return sequences

def main(data_path, file_name):
    #eactracting the DR features
    fw = open(current_path+'/features_code/PseinOne2/command_protein.sh', 'w')
    fw.write(
        'python '+current_path+'/features_code/PseinOne2/nac.py '+current_path+'/data/'+file_name+' Protein DR -out '+path_fea + file_name[:-6]+'_DR.txt\n'
        )
    fw.close()
    os.system("bash "+current_path+'/features_code/PseinOne2/command_protein.sh')
    print('DR end.......')
    txt_to_csv(path_fea + file_name[:-6] + '_DR.txt', path_fea + file_name[:-6] + '_DR.csv')
    print("DR is done ......")

    # eactracting the ESM features and flatten
    sequences = read_fasta(data_path + file_name)
    write_pro_seq(sequences, current_path+'/data/'+file_name[:-6]+"_ESM_data.txt")
    extract_ESM_features(file_name[:-6]+"_ESM_data.txt", file_name[:-6]+"_ESM_features.txt")
    esm_fea_df = pd.read_csv(path_fea + file_name[:-6] + '_ESM_features.txt', header=None)
    esm_fea_df.to_csv(path_fea + file_name[:-6] + '_ESM_features.csv', index=False)
    print("ESM-flattening-1024 is done ......")





if __name__ == '__main__':
    current_path = os.getcwd()
    path_fea = current_path+"/features/"
    model_path = current_path+"/model/"
    parser = argparse.ArgumentParser(description="predict the localization Nucleolus or Nucleoplasm for lncRNA")
    parser.add_argument('--fastafile', action='store', dest='fastafile', required=True, \
                        help='fasta file needs including sequence')
    args = vars(parser.parse_args())
    fasta_file = args['fastafile']
    output_file_DR = args['DR_test']
    output_file_ESM = args['ESM_test']
    data_path = os.path.dirname(fasta_file) + "/"
    file_name = os.path.basename(fasta_file)
    main(data_path, file_name)



