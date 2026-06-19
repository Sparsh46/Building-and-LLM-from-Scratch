import torch
from torch.utils.data import DataLoader,Dataset
import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")

with open("the-verdict.txt","r",encoding="utf-8") as f :
    text=f.read()

vocab_size=50257
dimension=128

token_emb = torch.nn.Embedding(vocab_size, dimension)

class DatasetV2(Dataset):
     
    def __init__(self,text,max_length,stride,tokenizer):
        self.inp=[]
        self.op=[]

        data=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

        for i in range(0,len(data)-max_length,stride):
            ic=data[i:max_length+i]
            tc=data[i+1:i+max_length+1]
            self.inp.append(torch.tensor(ic))
            self.op.append(torch.tensor(tc))

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, idx) :
        return self.inp[idx],self.op[idx]
    
def dataloaderV2(text,max_length=256,stride=10,
               batch_size=10,shuffle=True,drop_last=True,num_workers=0):
    
    tokenizer=tiktoken.get_encoding("gpt2")

    Ds=DatasetV2(text,max_length,stride,tokenizer)

    Dl=DataLoader(
        Ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return Dl

max_length=4
top=dataloaderV2(text,max_length=4,stride=max_length,batch_size=8,shuffle=False)

sparsh=iter(top)
inputs,targets=next(sparsh)
print(inputs)

token_emb=token_emb(inputs)

print(token_emb.shape)


context_length=max_length
positional_embeddings_layer = torch.nn.Embedding(context_length, dimension)
pos_emb=positional_embeddings_layer(torch.arange(context_length))

print(pos_emb.shape)

input_embeddings=pos_emb+token_emb

print(input_embeddings.shape)

