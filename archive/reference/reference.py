import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import DuckDuckGoLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ─── Configuration ─────────────────────────────────────────
USE_MODEL = "Qwen/Qwen2.5-0.5B"  # or "mistralai/Mistral-7B-Instruct-v0.1" or "microsoft/phi-2"
DEVICE = 0  # CUDA:0 if GPU, or -1 for CPU
MAX_DOCS = 5

# ─── Load LLM ──────────────────────────────────────────────
print(f"Loading model: {USE_MODEL}")
hf_pipe = pipeline("text-generation", model=USE_MODEL, device=DEVICE, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# ─── Prompt for Claim Extraction ───────────────────────────
claim_prompt = PromptTemplate(
    input_variables=["raw_input"],
    template="Extract the core factual claim from this text:\n\n{raw_input}"
)

claim_extractor = LLMChain(llm=llm, prompt=claim_prompt)

# ─── DuckDuckGo Web Search Loader ──────────────────────────
def fetch_search_docs(query: str, max_docs=5):
    print(f"Searching: {query}")
    loader = DuckDuckGoLoader(query=query)
    docs = loader.load()
    return docs[:max_docs]

# ─── Vector Store Setup ────────────────────────────────────
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ─── Final Prompt for Verdict ──────────────────────────────
verdict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to determine if the claim is true or fake.
Respond only if the answer is supported by the documents.

Claim: {question}

Sources:
{context}

Answer in 2–3 lines, with a verdict (Real/Fake) and a short explanation.
"""
)

# ─── RAG Chain with Context ────────────────────────────────
def build_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": verdict_prompt}
    )

# ─── Main Interface ────────────────────────────────────────
def check_fake_news(raw_input):
    # Step 1: Extract Claim
    claim = claim_extractor.run(raw_input)
    print(f"\n🔎 Extracted claim: {claim}")

    # Step 2: Search Web for Relevant Docs
    raw_docs = fetch_search_docs(claim, max_docs=MAX_DOCS)
    documents = [Document(page_content=doc.page_content) for doc in raw_docs]

    # Step 3: Build Vectorstore and Retriever
    vectorstore = get_vectorstore(documents)
    retriever = vectorstore.as_retriever()

    # Step 4: RAG Verdict
    rag_chain = build_rag_chain(llm, retriever)
    result = rag_chain.run(claim)

    print("\n📢 Verdict:\n", result)

# ─── Run It ────────────────────────────────────────────────
if __name__ == "__main__":
    input_text = input("📰 Enter a headline or news claim:\n> ")
    check_fake_news(input_text)
