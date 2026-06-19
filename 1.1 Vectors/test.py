import torch 
import torch.nn as nn

class attentionv1(nn.Module):
    def __init__(self,din,dout,context_length,dropout,num_heads,qkv_bias=False) :
        super().__init__()
        assert dout%num_heads==0

        self.dout=dout
        self.num_heads=num_heads
        self.head_dim=dout//num_heads

        self.queryW=nn.Linear(din,dout,bias=qkv_bias)
        self.keyW=nn.Linear(din,dout,bias=qkv_bias)
        self.valueW=nn.Linear(din,dout,bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),
                       diagonal=1)
        )

    def forward(self,input):
        b,num_tokens,d_in=input.shape

        query=self.queryW(input)
        key=self.queryW(input)
        value=self.queryW(input)

        query=query.view(b,num_tokens,self.num_heads,self.head_dim)  
        key=key.view(b,num_tokens,self.num_heads,self.head_dim)
        value=value.view(b,num_tokens,self.num_heads,self.head_dim)   

        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)

        attention_vals=query @ key.transpose(2,3)

        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]
        attention_vals.masked_fill_(mask_bool,-torch.inf)

        attn_weight=torch.softmax(attention_vals/key.shape[-1]**0.5,dim=-1)

        context_v=(attn_weight@value).transpose(1,2)

        context_v=context_v.contiguous().view(b,num_tokens,self.dout)

        return context_v
    
torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)
batch=torch.stack((inputs,inputs),dim=0)
b,context_length,din=batch.shape
dout=6
mha=attentionv1(din,dout,context_length,0.0,2)
cv=mha(batch)
print(cv)



        




        