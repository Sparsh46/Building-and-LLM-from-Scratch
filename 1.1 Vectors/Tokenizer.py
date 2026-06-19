with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()

print("total number of characters ", len(raw_text))

import re
text="Hello, what a. beautiful, world we live in"
result=re.split(r'(\s)',text)
print(result)

result=re.split(r'([,.]|\s)',text)
print(result)

filter_res=[]
for item in result:
    if item.strip():
        filter_res.append(item)

result=filter_res
print(result)

