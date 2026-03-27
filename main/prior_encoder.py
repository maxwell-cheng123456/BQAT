import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, embedding_dim=100, hidden_size=8, head=4):
        super().__init__()

        self.dim = embedding_dim
        self.hidden_size = hidden_size
        self.head = head
        self.size = head * hidden_size

        self.Wq = nn.Linear(embedding_dim, self.size, bias=self)
        self.Wk = nn.Linear(embedding_dim, self.size, bias=self)
        self.Wv = nn.Linear(embedding_dim, self.size, bias=self)

        self.W = nn.Linear(self.size, self.dim, bias=self)
        self.Layer_nor1 = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim * 4)
        self.ffn2 = nn.Linear(self.dim * 4, self.dim)
        self.active = nn.ReLU()
        self.Layer_norm_ffn = nn.LayerNorm(self.dim)
        self.att_drop = nn.Dropout(0.2)
        self.state_drop = nn.Dropout(0.2)

    def SelfAttention(self, x,matrix):

        new_size = x.size()[:-1] + (self.head, self.hidden_size)  # b n h s
        Q = self.Wq(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        K = self.Wk(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        V = self.Wv(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s

        attention_score = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.dim))
        attention_score = torch.mul(attention_score,matrix.unsqueeze(1))
        attention_score = nn.Softmax(dim=3)(attention_score)
        O = torch.matmul(attention_score, V)
        O = O.permute(0, 2, 1, 3)
        size2 = O.size()[:-2] + (self.size,)
        O = O.reshape(*size2)
        O = self.W(O)  # b * n * d
        O = self.state_drop(O)
        O = self.Layer_nor1(O + x)
        return O
    def FFN(self, x):
        hidden = self.active(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output)
        output = self.Layer_norm_ffn(output + x)
        return output
    def forward(self, x):
        x,matrix=x
        x = self.SelfAttention(x,matrix)
        x = self.FFN(x)
        return x


class Prior_Encoder(nn.Module):
    def __init__(self, embedding_dim=100, hidden_size=8, head=4, blocks=4):
        super().__init__()
        self.attentions = nn.Sequential(
            *[Attention(embedding_dim, hidden_size=hidden_size, head=head) for i in range(blocks)])
    def forward(self, x,matrix):
        x=(x,matrix)
        x = self.attentions(x)
        return x

if __name__ == "__main__":
    x = torch.rand(4,20,128)
    matrix = torch.rand(4,20,20)
    model = Prior_Encoder(embedding_dim=128, hidden_size=16, head=8, blocks=1)
    print(model)
    y = model(x,matrix)
    print(y.shape)







