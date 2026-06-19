with open("the-verdict.txt","r",encoding="utf-8") as f :
    text=f.read()

import tiktoken
tokenizer=tiktoken.get_encoding('gpt2')

data=tokenizer.encode(text)

import torch
from torch.utils.data import DataLoader,Dataset

class DatasetV1(Dataset):
     
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

def dataloaderV1(text,max_length=256,stride=10,
               batch_size=10,shuffle=True,drop_last=True,num_workers=0):
    
    tokenizer=tiktoken.get_encoding("gpt2")

    Ds=DatasetV1(text,max_length,stride,tokenizer)

    Dl=DataLoader(
        Ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return Dl

sparsh=dataloaderV1(text,max_length=4,stride=1,batch_size=8,shuffle=False)
pop=iter(sparsh)
inputs,outputs=next(pop)
print(inputs.shape)


vocab_size=50256
dim=128
vector_emb_layer=torch.nn.Embedding(vocab_size,dim)
vector_emb=vector_emb_layer(inputs)

max_length=4
context_length=max_length
positional_embeddings_layer=torch.nn.Embedding(context_length,dim)

pos_embs=positional_embeddings_layer(torch.arange(context_length))

inputs=vector_emb+pos_embs
print(inputs.shape)

