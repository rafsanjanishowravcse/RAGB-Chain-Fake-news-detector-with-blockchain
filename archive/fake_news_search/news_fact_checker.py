import os
import torch
import requests
import logging
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()

serper_api_key = "<key>"
if not serper_api_key:
    raise RuntimeError("mssing serper_api_key")
serper_search_url = "https://google.serper.dev/search"
model = "tiiuae/falcon-7b-instruct" 
device = 0 if torch.cuda.is_available() else -1 
max_docs = 5

# search
def search_serper(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "num": num_results}

    try:
        resp = requests.post(serper_search_url, headers=headers, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Serper api error: {e}")
        return []

    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title": item.get("title", "").strip(),
            "link": item.get("link", "").strip(),
            "snippet": item.get("snippet", "").strip()
        })
    return results

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=model,
    device=device,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    torch_dtype=torch.float16 if device >= 0 else torch.float32
)
llm = HuggingFacePipeline(pipeline=generator)

def fetch_search_docs(query: str, max_docs=max_docs):
    site_filters = "site:news18.com OR site:ptinews.com OR site:politifact.com OR site:timesofindia.com OR site:prothomalo.com OR site:bd-pratidin.com OR site:thedailystar.net" # relying on only trusted sites 
    query_with_sites = f"{query} {site_filters}"
    
    results = search_serper(query_with_sites, num_results=max_docs)

    docs = []
    for res in results:
        content = f"{res['title']}\n{res['snippet']}\nURL: {res['link']}"
        docs.append(Document(page_content=content, metadata={"source": res['link']}))
    return docs

def get_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

verdict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""

Claim:
{question}

Sources:
{context}
"""
)

def build_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": verdict_prompt},
        return_source_documents=True
    )

# news chek
def check_fake_news(raw_input):
    claim = raw_input.strip()
    if not claim:
        logging.warning("empty claim")
        return

    documents = fetch_search_docs(claim, max_docs=max_docs)
    if not documents:
        logging.warning("not found")
        return

    vectorstore = get_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    rag_chain = build_rag_chain(llm, retriever)
    result = rag_chain({"query": claim})

    print(f"\n Verdict:\n {result['result']}")
    print("============================================")
    print("\n Sources:")
    for doc in result.get("source_documents", []):
        print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
        input_text = input("Enter a headline or news claim\n> ")
        check_fake_news(input_text)
