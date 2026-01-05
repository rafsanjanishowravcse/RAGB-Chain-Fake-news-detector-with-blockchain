from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

# Load environment variable
load_dotenv()

# Initialize the model
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

def detecting_fake_or_real(claim, evidence_list):
    # evidence_list = list of {"snippet": ..., "url": ...}
    evidence_json = json.dumps(evidence_list, indent=2)

    prompt_template = PromptTemplate(
    input_variables=["claim", "evidence_json"],
    template="""
    You are a fake news detection assistant. Do not include any preamble or commentary.

    You will be given:
    - A claim
    - A list of evidence items in JSON format, where each item contains:
    - "snippet": the text of the snippet
    - "url": the source of the snippet

    Your task is:
    1. Analyze the claim using the provided evidence.
    2. Pick exactly ONE snippet and url that best supports your final decision.
    3. Output your answer strictly in JSON format with the following fields:
        {{
        "Claim": "{{claim}}",
        "Detection Result": "fake" or "real",
        "Snippet": "<the most relevant snippet>",
        "URL": "<the URL associated with the selected snippet>",
        "Explanation": "<a brief explanation in one or two lines>"
        }}

    IMPORTANT: Do not output anything except this JSON object.

    Claim:
    {claim}

    Evidence List (JSON):
    {evidence_json}
    """
    )


    chain = prompt_template | llm
    response = chain.invoke({"claim": claim, "evidence_json": evidence_json})
    return response.content
