import torch
vocab=torch.load('./vocab.pt')
vocab['[ERROR]']=79
torch.save(vocab,'vocab_with_error.pt')
print(vocab)
