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
