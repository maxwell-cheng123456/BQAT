import torch
import math

def PositionalEncoding(dim, seq_length):
    # 初始化一个足够长的位置编码矩阵，维度为 seq_length x dim
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    # 计算位置编码
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe




if __name__=='__main__':
    position=PositionalEncoding(128,200)

    print(position.shape)