import decoder as decoder
import encoder as encoder
import prior_encoder as prior_encoder
from position import PositionalEncoding
import torch
import torch.nn as nn

class Local_encoder_block(nn.Module):
    def __init__(self,  d_model=128,hidden_size=64, heads=8, blocks=1):
        super().__init__()
        self.Prior_Encoder1 = prior_encoder.Prior_Encoder(d_model, d_model // heads, heads, 1)
        self.Prior_Encoder2 = prior_encoder.Prior_Encoder(d_model, d_model // heads, heads, 1)
        self.Encoder = encoder.Encoder(d_model, d_model // heads, heads, 1)
    def forward(self,x):
        x,matrix1,matrix2=x
        out1=self.Prior_Encoder1(x,matrix1)
        out2=self.Prior_Encoder2(x, matrix2)
        out3=self.Encoder(x)
        out=(out1+out2)*1+out3*1
        return out,matrix1,matrix2
class New_Encoder(nn.Module):
    def __init__(self,  d_model=128,hidden_size=128//8, heads=8, blocks=6):
        super().__init__()
        self.New_Encoder =  nn.Sequential(
            *[Local_encoder_block(d_model, hidden_size, heads) for i in range(blocks)])
    def forward(self, x, matrix1,matrix2):
        x = (x,matrix1,matrix2)
        x= self.New_Encoder(x)
        return x[0]
class Prior_Transformer(nn.Module):
    def __init__(self, vocab_length, d_model=128, heads=8, blocks=4, max_length=300, device=None):
        super().__init__()
        self.position = PositionalEncoding(d_model, max_length + 1).unsqueeze(0).to(device)
        self.position1 = PositionalEncoding(d_model, max_length + 1).unsqueeze(0).to(device)
        self.embedding = nn.Embedding(vocab_length, d_model)
        self.new_encoder = New_Encoder(d_model, d_model // heads, heads, blocks)
        self.Decoder = decoder.Decoder(d_model, d_model // heads, heads, blocks, device)
        self.fc = nn.Linear(d_model, vocab_length)
    def forward(self, x, x1,matrix1,matrix2):
        x = self.embedding(x) + self.position[:, :x.shape[1]]
        x1 = self.embedding(x1) + self.position1[:, :x1.shape[1]]
        feature =self.new_encoder(x,matrix1,matrix2)
        out = self.Decoder(x1, feature)
        return self.fc(out)

if __name__=="__main__":
    vocab_length=80
    d_model = 32
    heads = 8
    blocks = 4
    max_length = 300
    device=torch.device('cpu')
    model=Prior_Transformer(vocab_length=80, d_model=128, heads=8, blocks=6, max_length=300, device=device)
    print(model)
    src=torch.randint(0,80,(4,15)).long()
    tgt=torch.randint(0,80,(4,15)).long()
    matrix=torch.randint(0,4,(4,15,15)).float()
    out=model(src,tgt,matrix,matrix)
    print(out.shape)
