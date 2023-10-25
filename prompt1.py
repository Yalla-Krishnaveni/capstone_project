from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import util
import torch
import ast
import pandas as pd
from gpt4all import GPT4All
from langchain.chains.question_answering import load_qa_chain


para_df=pd.read_csv('data/paragraphs_embeddings.csv')
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
def order_documents(query):
    documents=para_df['page_content'].tolist()
    Embeddings=para_df['Embeddings'].tolist()
    output_list = [ast.literal_eval(item) for item in Embeddings]
    query_result = embeddings.embed_query(query)
    query_result = torch.Tensor(query_result)
    Embeddings1 = torch.Tensor(output_list)
    hits = util.semantic_search(query_result, Embeddings1, top_k=5)[0]
    # hits = hits[0]      #Get the hits for the first query
    docs = []
    for hit in hits:
        document = documents[hit['corpus_id']]
        docs.append(document)
    docs = ' '.join(docs)
    return docs

query = "what is python?"
res=order_documents(query)
print(res)


model = GPT4All(model_name='orca-mini-3b.ggmlv3.q4_0.bin')
# model = GPT]4All(model_name='orca-mini-13b.ggmlv3.q4_0.bin')
prompt1 = f"""text:{res}


        Answer the below Question only from the text above.If the answer cannot be found from text above respond like 'I don't know'.

        Question:{query}
        Answer:"""
output = model.generate(prompt=prompt1,temp=0, max_tokens=512)
print(output)