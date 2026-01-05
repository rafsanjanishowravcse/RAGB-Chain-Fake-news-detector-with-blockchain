import os
import pandas as pd
import requests
from dotenv import load_dotenv
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import Document


# 🔹 Serper Search
def get_relevant_documents(query: str):
    _SERPER_SEARCH_URL = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": "b338d0392a4e069306ba4c57b9ac54873665c77c",
        "Content-Type": "application/json"
    }
    payload = {
        "q": '(site:news18.com OR site:ptinews.com OR site:politifact.com) ' + query,
        "num": 5,
    }

    try:
        resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=5)
        if resp.status_code != 200:
            raise Exception(f"Serper API Error: {resp.text}")
        results = resp.json()
        
        documents = []
        for result in results.get("organic", []):
            content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
            documents.append(Document(page_content=content, metadata={"source": result.get("link", "")}))
        
        if not documents:
            raise Exception("No search results returned.")

        return documents
    except Exception as e:
        print(f"⚠️ Error while fetching Serper search results for '{query}': {e}")
        return [Document(page_content="No relevant evidence found.", metadata={})]


# 🔹 Prompt & LLM Chain
def train_with_evidence(claims_dataset, model, prompt_version=1, batch_size=16):
    if prompt_version == 1:
        template = '''
          Here’s a statement someone made:  
        {question}  
        Based on the evidence below, decide if the statement is Real or Fake.  
        {evidence}  
        Give only one-word answer: Real or Fake.
       '''
    elif prompt_version == 2:
        template = """
        You are governed by strict rules:
        - You must analyze the claim against the evidence.
        - You must return only "Real" or "Fake".
        - No additional commentary is allowed.

        Claim: {question}  
        Evidence: {evidence}
        """
    elif prompt_version == 3:
        template = """
        You are a fact-checking assistant.  
        Your task is to classify the claim as Real or Fake based on the evidence provided.  
        Do not provide explanations, just return the classification.

        Claim: {question}  
        Evidence: {evidence}

        Output must be exactly one of: Real or Fake.
        """
    elif prompt_version == 4:
        template = """
        You are an impartial fact-checking assistant.

    Your task is to classify the following claim as either REAL or FAKE, using only the provided evidence.

    Instructions:
    - Carefully read the claim and the evidence.
    - If the evidence clearly supports the claim and you are at least 80% confident, respond with REAL.
    - If the evidence contradicts the claim, is insufficient, or you are less than 80% confident, respond with FAKE.
    - Do not use any external knowledge or make assumptions beyond the evidence.

    Claim:
    {question}

    Evidence:
    {evidence}

    Respond with only one word: REAL or FAKE.  
        """
    else:
       template = """
        You are a fact-checking assistant.
          
          Claim: {question}
          
          Evidence:
          {evidence}
          
          Decide whether the claim is REAL or FAKE based only on the evidence.
          
          Respond in this format:
          Classification: REAL or FAKE

        """

    prompt = PromptTemplate.from_template(template)

    chain = (
        RunnableLambda(lambda x: {
            "question": x["question"],
            "evidence": "\n".join([doc.page_content for doc in x["evidence"]])
        })
        | prompt
        | ChatGroq(api_key="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", model_name=model)
        | StrOutputParser()
    )

    predictions = []
    for i in range(0, len(claims_dataset), batch_size):
        print(f"📦 Batch size {batch_size} at iteration {i}")
        batch = claims_dataset[i:i + batch_size]
        for claim in batch:
            docs = get_relevant_documents(claim)
            result = chain.invoke({"question": claim, "evidence": docs})
            print(f"🔎 Claim: {claim}\n🔁 Raw Prediction: {result}\n{'-'*50}")
            predictions.append(result)
    return predictions


# 🔹 Normalize labels from model
def normalize_label(label):
    if not isinstance(label, str):
        print(f"⚠️ Invalid prediction format: {label}")
        return "Unsure"
    label = label.strip().lower()
    if "real" in label:
        return "Real"
    elif "fake" in label:
        return "Fake"
    else:
        print(f"⚠️ Unexpected label format: '{label}'")
        return "Unsure"


# 🔹 Normalize labels from CSV
def normalize_ground_truth(label):
    if isinstance(label, bool):
        return "Real" if label else "Fake"
    label = str(label).strip().lower()
    if label in ["true", "real", "1"]:
        return "Real"
    elif label in ["false", "fake", "0"]:
        return "Fake"
    else:
        return "Unsure"


# 🔹 Main
if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), "fake_news_claims_factchecked.csv")
    df = pd.read_csv(csv_path,nrows=100)  # Load fewer rows for testing

    claims = df['claim'].tolist()
    true_labels_raw = df['label'].tolist()
    true_labels = [normalize_ground_truth(l) for l in true_labels_raw]

    model_name = "whisper-large-v3"
    raw_preds = train_with_evidence(claims, model=model_name)
    predicted_labels = [normalize_label(pred) for pred in raw_preds]

    print("\n🔎 Final Predictions:")
    for claim, pred in zip(claims, predicted_labels):
        print(f"Claim: {claim} → Prediction: {pred}")

    print("\n🔢 Prediction Summary:", Counter(predicted_labels))

    # Filter valid labels
    filtered_true = []
    filtered_pred = []
    for t, p in zip(true_labels, predicted_labels):
        if p in ["Real", "Fake"]:
            filtered_true.append(t)
            filtered_pred.append(p)

    if not filtered_true or not filtered_pred:
        print("⚠️ No valid predictions (Real or Fake) returned. Cannot compute metrics.")
    else:
        print("\n✅ Classification Report:")
        print(classification_report(filtered_true, filtered_pred))
        print(f"🎯 Accuracy: {accuracy_score(filtered_true, filtered_pred):.4f}")
        print(f"📊 F1 Score: {f1_score(filtered_true, filtered_pred, pos_label='Real', average='weighted'):.4f}")
