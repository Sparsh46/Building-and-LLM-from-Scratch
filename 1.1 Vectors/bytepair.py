import importlib
import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")

text = (
    "Hello do you like tea ? <|endoftext|> in the sunlight terraces"
    "of someunknownplace"
)

integers=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
strings=tokenizer.decode(integers)

x=tokenizer.encode("sidjaisjd hhhfjnfj pucji")
y=tokenizer.decode(x)
print(x)
print(y)

with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()

z=tokenizer.encode(raw_text)
print(z[:100])






