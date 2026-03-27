import torch
import torch.nn as nn

class D_Attention(nn.Module):
    def __init__(self, embedding_dim=128, hidden_size=16, head=8,device=torch.device('cuda')):#didden_size表示dim_v
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

        self.Wq_cross = nn.Linear(embedding_dim, self.size, bias=self)
        self.Wk_cross = nn.Linear(embedding_dim, self.size, bias=self)
        self.Wv_cross = nn.Linear(embedding_dim, self.size, bias=self)
        self.W_cross = nn.Linear(self.size, self.dim, bias=self)
        self.Layer_nor2 = nn.LayerNorm(self.dim)

        self.device=device

    def create_masked_attention_mask(self,seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.float()
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.device)
    def MaskedAttention(self, x):
        seq_len=x.shape[1]
        new_size = x.size()[:-1] + (self.head, self.hidden_size)  # b n h s
        Q = self.Wq(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        K = self.Wk(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        V = self.Wv(x).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        mask=self.create_masked_attention_mask(seq_len)
        attention_score =( torch.matmul(Q, K.transpose(2, 3)) +mask)/ torch.sqrt(torch.tensor(self.dim))
        attention_score = nn.Softmax(dim=3)(attention_score)
        attention_score = self.att_drop(attention_score)
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

    def cross_attention(self,x,feature):
        global a
        new_size_1 = x.size()[:-1] + (self.head, self.hidden_size)  # b n h s
        new_size = feature.size()[:-1] + (self.head, self.hidden_size)  # b n h s
        Q = self.Wq_cross(x).view(*new_size_1).permute(0, 2, 1, 3)  # b h n s
        K = self.Wk_cross(feature).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        V = self.Wv_cross(feature).view(*new_size).permute(0, 2, 1, 3)  # b h n s
        attention_score = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.dim))
        attention_score = nn.Softmax(dim=3)(attention_score)
        attention_score = self.att_drop(attention_score)  # sublayer 之间要 Dropout
        O = torch.matmul(attention_score, V)
        O = O.permute(0, 2, 1, 3)
        size2 = O.size()[:-2] + (self.size,)
        O = O.reshape(*size2)
        O = self.W_cross(O)  # b * n * d
        O = self.state_drop(O)
        O = self.Layer_nor2(O + x)
        return O
    def forward(self, inputs):
        x, feature = inputs
        x = self.MaskedAttention(x)
        if feature!=None:
            x=self.cross_attention(x,feature)
        x = self.FFN(x)
        return x,feature
class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, hidden_size=16, head=8,blocks=2,device=torch.device('cuda')):
        super().__init__()
        self.attentions = nn.Sequential(
            *[D_Attention(embedding_dim, hidden_size=hidden_size, head=head,device=device) for i in range(blocks)])
    def forward(self, x, feature = None):
        inputs = (x, feature)
        inputs = self.attentions(inputs)
        return inputs[0]
if __name__ == "__main__":
    torch.manual_seed(10)
    model=Decoder(2,2,1,1).cuda()
    x=torch.rand(1,2,2).cuda()
    y=model(x)
    print(y)








