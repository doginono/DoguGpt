#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#chat = open("../_chat.txt", encoding="utf-8")
with open("C:/Users/dogut/OneDrive/Masaüstü/HeaderFolder/_chat.txt",  encoding="utf-8") as f:
    chat = [line.strip('\n') for line in f]


# ## Clean Data

# In[2]:


import re
joinedChat=' '.join(chat)
print(joinedChat[:500])

emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
joinedChat=(emoj.sub(r'', joinedChat)) # no emoji

joinedChat=re.sub(r"\[[^\]]*\]", '\n', joinedChat)
joinedChat=re.sub(r" Elif Yavrum:", 'Elif Yavrum:', joinedChat)
joinedChat=re.sub(r" Dogu Tamgac:", 'Dogu Tamgac:', joinedChat)
joinedChat=joinedChat.replace("\u200e", "")
print(joinedChat[:500])


# In[3]:


chars=sorted(list(set(joinedChat)))


# In[4]:


vocab_size=len(set(chars))
print(''.join(chars))
print(vocab_size)


# # Building character level language model. <br>
# stoi: Map strings to integers. <br>
# Then apply it and create encoding and decoding lambda functions <br>
# (Normally you can use sentencepiece from google or tiktoken)

# In[5]:


stoi={ch:i for i,ch in enumerate(chars)}
itoi={i:ch for i,ch in enumerate(chars)}
encode=lambda s: [stoi[i] for i in s]
decode=lambda l: ''.join([itoi[i] for i in l])
print(encode("l l"))
print(decode(encode("l l")))


# In[6]:


import torch
torch.cuda.is_available()
data=torch.tensor(encode(joinedChat), dtype=torch.long)
print(data.shape , data.dtype)
print(data[:100])


# In[7]:


n = int(0.95 * len(data))
train_data=data[:n]
val_data=data[n:]


# Now take the data asa blocks. We chose the block size 8 and initialized the length of the training data with 9 as there are 8 positions to guess

# In[8]:


block_size= 8
batch_size= 4

train_data[:block_size+1]


# In[9]:


torch.manual_seed(1337) # to be able to reproduce
def get_batch(split):
    data=train_data if split=='train' else val_data
    #shape batchsize row vector
    ix=torch.randint(len(data) - block_size, (batch_size,))
    #stacked multiple sentences as row vectors
    x=torch.stack([data[i:i+block_size] for i in ix])
    #to train
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
xb,yb= get_batch('train')
print("input:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)



# # Using Bigram model=> Simple fast easy to understand
# Predict the next word from the earlier word.(In our case character) Conditional probability

# In[47]:


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

vocab_size=len(set(chars))
print(vocab_size)
#get the earlier character then propagate it 

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab):
        super().__init__()
        #wrapper essentialy a table by number of characters*number of characters where you have the scores
        self.token_embedding_table=nn.Embedding(vocab, vocab)
        #print("tokentable", self.token_embedding_table)

    def forward(self, idx, targets=None):
        logits=self.token_embedding_table(idx)
        #logits B t C where B=4 batchsize, T=8 chunks, C=number of characters
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            #remove Chunk indexes like appending together
            logits= logits.view(B*T, C)
            targets= targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        return logits, loss # scores
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #Predictions
            logits, loss=self(idx)
            #Only take the last characters from chunks
            logits=logits[:,-1,:]
            probs=F.softmax(logits, dim=-1)
            #get one sample from the softmax
            idx_next=torch.multinomial(probs, num_samples=1) # (B,1)
            #append the word to the chunk
            idx=torch.cat((idx, idx_next),dim=1)  # (B,T+1)
        return idx
            
m=BigramLanguageModel(vocab_size)
logits, loss=m(xb,yb) # forward
print(logits.shape)
# estimated loss ln(1/131)
print(loss)

print(decode(m.generate(torch.zeros((1,1),dtype=torch.long), max_new_tokens=100)[0].tolist()))


# In[39]:


optimizer=torch.optim.AdamW(m.parameters(), lr=1e-3)


# In[43]:


batch_size=32

for steps in range(10000):
    xb,yb=get_batch('train')
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


# In[45]:


print(decode(m.generate(torch.zeros((1,1),dtype=torch.long), max_new_tokens=400)[0].tolist()))


# import ntlk
# AI_tokens=word_tokenize(joinedChat)
# print(AI_tokens[:500])

# from nltk.probability import FreqDist
# fdist=FreqDist()
# for word in AI_tokens:
#     fdist[word.lower()]+=1
# fdist

# In[ ]:





# In[ ]:




