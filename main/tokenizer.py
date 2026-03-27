import torch
from torch.utils.data import Dataset
import numpy as np
np.random.seed(40)
import random
random.seed(40)
from smiles_graph import *
from torch.nn import functional as F

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class SimpleTokenizer:
    def __init__(self, vocab_file,max_length=300):
        self.vocab = torch.load(vocab_file)
        self.inverse_vocab = list(self.vocab.keys())
        self.max_length=max_length
    def tokenize(self, text):
        tokens = text.split(' ')
        indexes = [self.get_index(word) for word in tokens]
        indexes=np.array(indexes)
        return torch.from_numpy(indexes)
    def get_index(self,word):
        if word in self.inverse_vocab:
            return self.vocab[word]
        else:
            return self.vocab['[UNK]']
    def detokenize(self, indexes):
        words = [self.inverse_vocab[index] for index in indexes]
        return words
def Pad(src,padding=300):
    src=src.split(' ')
    if len(src) > padding:
        src+=['[BOS]']+src[:padding]+['[EOS]']
    else:
        src=['[BOS]']+src +['[EOS]']
    src=' '.join(src)
    return src

def pad_matrix(matrix_batch,max_len1):
    batch_size=len(matrix_batch)
    padd_matrix = torch.zeros((batch_size,max_len1,max_len1))
    for i , matrix in enumerate(matrix_batch):
        n=matrix.size(0)
        padd_matrix[i,:n,:n] = matrix
    return padd_matrix

def collate_fn(batch):
    src_batch, tgt_batch,matrix_energy_batch,matrix_bondlength_batch= zip(*batch)
    max_len1 = max(len(x) for x in src_batch)     
    max_len2 = max(len(x) for x in tgt_batch)
    padded_src_batch = [torch.cat([x, torch.zeros(max_len1 - len(x), dtype=torch.long)], dim=0) for x in src_batch]
    padded_tgt_batch = [torch.cat([x, torch.zeros(max_len2 - len(x), dtype=torch.long)], dim=0) for x in tgt_batch]
    padded_matrix_energy=pad_matrix(matrix_energy_batch,max_len1)
    padded_matrix_bondlength = pad_matrix(matrix_bondlength_batch, max_len1)
    src_batch = torch.stack(padded_src_batch)
    tgt_batch = torch.stack(padded_tgt_batch)
    return src_batch,tgt_batch,padded_matrix_energy,padded_matrix_bondlength

class NLPDataset(Dataset):
    def __init__(self, src_file,tgt_file,tokenizer, max_length=300):
        self.src_file = src_file
        self.tgt_file = tgt_file        
        self.tokenizer = tokenizer        
        self.max_length = max_length
        self.src_lines = open(src_file, 'r', encoding='utf-8').readlines()
        self.tgt_lines = open(tgt_file, 'r', encoding='utf-8').readlines()
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx].strip()
        if src_text[0]=='.':
            src_energy_matrix = torch.from_numpy(get_energy_matrix(''.join(src_text[2:].split(' '))))
            padded_energy_matrix = F.pad(src_energy_matrix, (2, 1, 2, 1), mode='constant', value=0)
            src_bondlength_matrix = torch.from_numpy(get_bondlength_matrix(''.join(src_text[2:].split(' '))))
            padded_bondlength_matrix = F.pad(src_bondlength_matrix, (2, 1, 2, 1), mode='constant', value=0)
        else:
            src_energy_matrix = torch.from_numpy(get_energy_matrix(''.join(src_text.split(' '))))
            padded_energy_matrix = F.pad(src_energy_matrix,(1,1,1,1),mode='constant',value=0)
            src_bondlength_matrix = torch.from_numpy(get_bondlength_matrix(''.join(src_text.split(' '))))
            padded_bondlength_matrix = F.pad(src_bondlength_matrix, (1, 1, 1, 1), mode='constant', value=0)
        padded_energy_matrix.diagonal().fill_(1)
        padded_bondlength_matrix.diagonal().fill_(1)
        tgt_text = self.tgt_lines[idx].strip()
        line_src= Pad(src_text, padding=self.max_length)
        line_tgt= Pad(tgt_text, padding=self.max_length)
        tokens_src = self.tokenizer.tokenize(line_src)
        tokens_tgt = self.tokenizer.tokenize(line_tgt)

        return tokens_src , tokens_tgt,padded_energy_matrix,padded_bondlength_matrix

    
if __name__=="__main__":
    pass


