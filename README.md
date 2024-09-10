# build_gpt_from_scratch
This Repository has all you need to build a gpt2 from scratch with explanation. some animation was created by me and other taked in books, article or papers.
# Data Loader & Dataset:
![image](https://github.com/user-attachments/assets/24f7cdbb-98e5-4bdb-bff9-21d55516512a)
* The Dataset contains all individual data samples, while the DataLoader organizes these samples into batches.
* The Dataset is static (just holds data), while the DataLoader is dynamic (processes and serves data).
* The DataLoader adds functionality like **shuffling**, **batching**, and **parallel data loading**, which aren't part of the Dataset itself.
### Dataset
```python
class DatasetV1(Dataset):
  def __init__(self,text,tokenizer,max_length,stride=1):
    self.tokenizer=tokenizer # in most case we will use tiktoken
    self.input_ids=[]
    self.target_ids=[]
    ids=tokenizer.encode(text)
    # Sliding Window
    for i in range(0,len(ids)-max_length,stride):
      inp=ids[i:i+max_length]
      out=ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(inp))
      self.target_ids.append(torch.tensor(out))
  def __len__(self):
    return len(self.input_ids)
  def __getitem__(self,idx):
    return self.input_ids[idx],self.target_ids[idx]
```
* sliding window : with this approach we will generate our data 
![image](https://github.com/user-attachments/assets/a6d9a9cc-2965-4bf1-96e7-7901a79cc9da)

### DataLoader
```python
def data_loader(text,max_length=256,stride=128,batch_size=4,shuffle=True,drop_last=True):
  # create tokenizer
  tokenizer=tiktoken.get_encoding("gpt2")
  #create dataset
  dataset=DatasetV1(text,tokenizer,max_length,stride)
  #create data loader
  data_loader_=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
  return data_loader_
```
# Positional Embedding & Token Embedding
![image](https://github.com/user-attachments/assets/4f06ec57-dc33-4c71-865f-1f6e569751b0)
1. Token Embeddings:

* Each word or subword token in the input sequence is mapped to a dense vector representation.
* These embeddings capture semantic meanings of the words.
* In the diagram, each word ("The", "quick", "brown", "fox", "jumps") is assigned a unique embedding vector.
* These embeddings are learned during the training process.

```python
# create token Embedding layer
word_embedding=nn.Embedding(
    model_params.vocab_size,
    model_params.embedding_size)
```
2. Positional Embeddings:

* These embeddings encode the position of each token in the sequence.
* They allow the model to understand the order of words, which is crucial for language understanding.
* Each position (1, 2, 3, 4, 5) has its own unique embedding vector.
* These can be learned or generated using mathematical functions (e.g., sine and cosine functions in the original Transformer paper).
```python
# create positional Embedding layer with rraining params
position_embedding=nn.Embedding(
    model_params.max_size,
    model_params.embedding_size
)
```

3. Final Embeddings:
* The final embedding for each token is the sum of its token embedding and its positional embedding.
* This combination allows the model to understand both the meaning of the word and its position in the sentence.

```python
# Embedding 
embedding=word_embedding+position_embedding
```
# Attentions
## Self Attention
![image](https://github.com/user-attachments/assets/93abb002-cba4-44c5-b52c-479809ff99c5)
* Self Attention Class
```python
class SelfAttention_v2(nn.Module):
  def __init__(self,in_d,out_d,b_qkv=False):
    super().__init__()
    self.in_dim=in_d
    self.out_dim=out_d
    self.w_key=nn.Linear(in_d,out_d,b_qkv)
    self.w_query=nn.Linear(in_d,out_d,b_qkv)
    self.w_value=nn.Linear(in_d,out_d,b_qkv)
  def forward(self,X):
    scores_weights=self.__attentions_scores(X)
    context_vects=self.__context_vects(scores_weights,X)
    return context_vects
  def __attentions_scores(self,X):
    queries=self.w_query(X)
    keys=self.w_key(X)
    scores=queries @ keys.T
    weights=torch.softmax(scores/keys.shape[-1]**0.5,dim=-1)
    return weights
  def __context_vects(self,weights,X):
    values=self.w_value(X)
    return weights@values
```
* code explanation
![image](https://github.com/user-attachments/assets/2cc05620-1a82-4e0a-b834-7aeeae0a8237)

## Causual Attention (Masked Attention)
The attention mask is essentially a way to stop the model from looking at the information we don't want it to look at. 
![image](https://github.com/user-attachments/assets/eae5dde7-f236-44cd-a4a2-5564a290110e)

* Implementation Steps
![image](https://github.com/user-attachments/assets/2d979eb1-a43a-44ff-9a13-18d547acfc1c)
```python
# calculate attentions weightd with Self
context_lenght=embed_mat.weight.shape[0]
in_dim,out_dim=embed_mat.weight.shape[1],embed_mat.weight.shape[0]
w_key=nn.Linear(in_dim,out_dim)
w_query=nn.Linear(in_dim,out_dim)
# calculate attentions weight
keys=w_key(embed_mat.weight)
queries=w_query(embed_mat.weight)
attention_scores=queries @ keys.T
# calculate attention weights
attention_weights=torch.softmax(attention_scores/context_lenght**0.5,dim=-1)
```

![image](https://github.com/user-attachments/assets/a927b4a4-a594-4062-a8fc-a13b1805b59a)
```python
# generate tril matrix
mask_mat=torch.tril(torch.ones((context_lenght,context_lenght)))
# calculate scores
masked_attention_scores=attention_weights * mask_mat
# make all zeros values equal -00 -> softmax(-00)=0
masked_attention_scores[masked_attention_scores==0]=-torch.inf
# calculate masked att weights
masked_attention_weights=torch.softmax(masked_attention_scores,dim=1)
```
## Dropout
Dropout is the process of randomly setting some nodes to output zero during the training process. 
```python
# create drop out layer with treshould of 0.2
dropout=nn.Dropout(0.2)
# apply dropout on attention_weights, the remain values will scaled of 1/0.2= 5
dropout(masked_attention_weights)
```
![image](https://github.com/user-attachments/assets/9615e3a6-0fff-46da-b32f-5d360aa86ef4)

## Causal Attention Class + Dropout
```python
class CausalAttention(nn.Module):
  def __init__(self,in_dim,out_dim,context_lenght,p_dropout=0.5,b_kqv=False):
    super().__init__()
    self.w_key=nn.Linear(in_dim,out_dim,b_kqv)
    self.w_query=nn.Linear(in_dim,out_dim,b_kqv)
    self.w_value=nn.Linear(in_dim,out_dim,b_kqv)
    self.dropout=nn.Dropout(p_dropout) # add drop out layer with threshold p_dropout
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_lenght,context_lenght),diagonal=1)
    )
  def forward(self,X):
    att_weights=self.__attention_weights(X)
    context_vects=self.__context_vects(X,att_weights)
    return context_vects
  def __attention_weights(self,X):
    # X shape (c,a,b)
    _,nbr_tokens_,_=X.shape
    keys=self.w_key(X)
    queries=self.w_query(X)
    scores=queries @ keys.transpose(1,2) # a=1,b=2
    scores.masked_fill_(
        self.mask.bool()[:nbr_tokens_,:nbr_tokens_],-torch.inf
    )
    masked_weight_att=torch.softmax(scores/keys.shape[-1]**0.5,dim=-1)
    drop_masked_weight_att=self.dropout(masked_weight_att)
    return drop_masked_weight_att
  def __context_vects(self,X,att_weights):
    values=self.w_value(X)
    return att_weights @ values
```

## Multi-Head Attentions
* Implementing multi-head attention involves creating multiple instances of the self-attention mechanism
![image](https://github.com/user-attachments/assets/e70dc264-957a-4b09-ab69-ed68f606d35f)
```python
class MultiheadAttentionWrapper(nn.Module):
  def __init__(self,in_dim,out_dim,context_lenght,num_head,p_drop=0.5,b_kqv=False):
    super().__init__()
    self.heads=nn.ModuleList([CausalAttention(in_dim,out_dim,context_lenght,p_drop,b_kqv) for _ in range(num_head)])
  def forward(self,X):
    context_vects=torch.cat([head(X) for head in self.heads],dim=-1) # concatenation of context vects
    return context_vects
```
![image](https://github.com/user-attachments/assets/e8ec1cea-c8b2-464b-bc86-170e4d32e738)
## MultiHead Attentions with Causual Att
The reason is that we only need one matrix multiplication to compute the keys, for instance,
  * keys = self.W_key(x) (the same is true for the queries and values).

In the MultiHeadAttentionWrapper, we needed to repeat this matrix multiplication, which is computationally one of the most expensive steps, for each attention head.

![image](https://github.com/user-attachments/assets/a0e50b40-708b-48ac-8760-1d9bda045d31)

```python
class MultiHeadAttention(nn.Module):
  def __init__(self,d_in,d_out,context_length,num_heads,p_drop=0.5,b_kqv=False):
    super().__init__()
    assert d_out%num_heads==0 , "d_out must be divisible by num_heads"
    self.d_out=d_out
    self.num_heads=num_heads
    self.head_out=d_out//num_heads
    self.context_length=context_length
    self.w_key=nn.Linear(d_in,d_out,bias=b_kqv)
    self.w_query=nn.Linear(d_in,d_out,bias=b_kqv)
    self.w_value=nn.Linear(d_in,d_out,bias=b_kqv)
    self.drop_out=nn.Dropout(p_drop)
    self.register_buffer("mask",
                         torch.triu(
                             torch.ones(context_length,context_length),diagonal=1
                         ))
    self.out_proj=nn.Linear(d_out,d_out)

  def forward(self,X):
    b,nbr_tokens,embed_out=X.shape
    keys=self.w_key(X)
    queries=self.w_query(X)
    values=self.w_value(X)
    # change the shape of k,q,v to be batch , context len, number of heads and head out
    keys=keys.view(b,nbr_tokens,self.num_heads,self.head_out)
    queries=queries.view(b,nbr_tokens,self.num_heads,self.head_out)
    values=values.view(b,nbr_tokens,self.num_heads,self.head_out)
    # transpose heads with nbr of tokens or context len
    keys=keys.transpose(1,2)
    queries=queries.transpose(1,2)
    values=values.transpose(1,2)
    # calculate attention scores
    att_scores=queries @ keys.transpose(2,3)
    # causual Attention
    mask_bool=self.mask.bool()[:nbr_tokens,:nbr_tokens]
    att_scores.masked_fill_(mask_bool,-torch.inf)
    # weigthts causual attentions
    att_weights=torch.softmax(att_scores/keys.shape[-1]**0.5,dim=-1) # b,head,token,head_out
    # drop out
    att_weights=self.drop_out(att_weights)
    # context vector
    context_vect=(att_weights @ values).transpose(1,2)
    context_vect=context_vect.contiguous().view(b,nbr_tokens,self.d_out) # b,token, context_vect  (num_heads * self.head_out)
    return self.out_proj(context_vect)
```


