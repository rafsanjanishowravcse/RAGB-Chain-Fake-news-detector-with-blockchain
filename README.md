# Contributors
Rahul Kumar, Rahul Rai, Vishakha Kumari, Manoj Kumar, Mukunda, Pavan Kumar, Rajat Chaudhary

# Description
This project implements an intelligent chatbot capable of verifying whether a news claim — provided as text or audio — is REAL, FAKE, or UNSURE. It uses a hybrid architecture that combines:  
    -> Retrieval-Augmented Generation (RAG) for grounding the model's reasoning in factual web-based evidence.   
    -> SarvamAI APIs for real-time speech-to-text (STT) conversion and translation.  
    -> ChatGroq-powered LLMs (Qwen2.5, Mistral, LLaMA) for fact-checking, explanation, and natural language reasoning.  

# Key Features:
-> Input Flexibility: Accepts user input in multiple languages and formats (typed text, spoken audio).  
-> Multilingual Processing: Automatically translates non-English claims to English using SarvamAI and processes them through the LLM pipeline.  
-> Dual-Language Output: Explanations are returned in both English and the original input language for clarity and accessibility.  
-> Grounded Fact Checking: Verifies claims using real-time search results(trusted sources (e.g news18.com, ptinews.com)) and generates evidence-supported explanations via a LangChain RAG workflow.  
-> Powered by Open Source LLMs: Supports Qwen2.5, Mistral, and Phi-3-mini through the ChatGroq LLM API for fast and scalable responses.

## Enhanced Features (Latest Updates)

### Credibility Scoring
Both REAL and FAKE classifications now include a 0-100 credibility score indicating the confidence level in the classification. 
- **FAKE scores** indicate confidence that the claim is misinformation (100 = completely certain it's fake)
- **REAL scores** indicate confidence that the claim is accurate based on supporting evidence (100 = completely certain it's true)
- **UNSURE classifications** do not receive credibility scores due to insufficient evidence
- REAL classifications include summarized supporting evidence in the explanation

### Claim Storage System
All verified claims (REAL/FAKE/UNSURE) are automatically stored locally for future reference and analysis. Each stored record includes:
- Claim text (both normalized and original)
- Classification (REAL/FAKE/UNSURE)
- Credibility score (0-100 for REAL and FAKE claims, 0 for UNSURE)
- Evidence URLs and source information
- Timestamp of verification
- Language of the claim
- Semantic embedding vector (stored for offline analytics or future features)

Storage location: `code/claim_metadata/` directory (JSON format for easy inspection and portability)

# How It Works
1) Accepts a news claim as text or audio(in any Indian language).
2) Translates to English using SarvamAI (if needed).
3) Performs intelligent web search using Serper.dev.
4) Applies multi-query RAG to gather and summarize evidence.
5) Uses an LLM to classify the claim as REAL / FAKE / UNSURE with explanation and credibility score (for REAL and FAKE claims).
6) Optionally translates the verdict back to the original language.
7) Stores all claims with evidence sources for future reference.

# Setup
Step1: Install dependencies: pip install -r requirements.txt  
Step2: Create a .env file with:  
    SARVAM_API_KEY, GROK_API_KEY, SERP_DEV_API_KEY, model_multi_query, model_summarizer, model_judge.  
Step 3: execute app.py file  

# Hosted Demo
This project is hosted on Hugging Face Spaces. You can try it live by clicking the link below: 
Try it here: [Fake News Detection LLM on Hugging Face](https://huggingface.co/spaces/rahul8459875/Fake_News_Detection_LLM) 
No installation needed — just paste or speak a claim to get started! 

NOTE: This file(code/fake_news_detection_llm.py) contains the full pipeline for news verification using RAG, LangChain, SarvamAI, and Groq.
Refer code folder for more details.

## File Structure

Key files in the `code/` directory:
- `app.py` - Gradio UI for text and image verification
- `fact_check_llm.py` - Core verification logic with Bengali semantic search and credibility scoring
- `image_fact_checker.py` - Image processing, OCR, visual analysis
- `claim_storage.py` - Claim storage manager
- `claim_metadata/` - Directory for stored claim records (JSON files)
- `test_claim_storage.py` - Test suite for storage functionality
- `requirements.txt` - All dependencies including visual analysis models

## Usage

The application displays credibility scores in the UI for both REAL and FAKE claims. Every verification run persists a JSON record so operators can audit previous checks later.

**Example Output for REAL Claim:**
```
Classification: REAL
Credibility Score: 92/100
Explanation: Multiple reliable sources confirm...
```

**Example Output for FAKE Claim:**
```
Classification: FAKE
Credibility Score: 85/100
Explanation: This claim has been debunked by...
```

## Data Management

### Viewing Stored Claims
Stored claims can be inspected by viewing JSON files in the `code/claim_metadata/` directory. Each file contains a complete verification record including claim text, classification, credibility score, evidence sources, and timestamp.

### Clearing Stored Claims
To remove stored records, delete the JSON files inside `code/claim_metadata/`. New claims will generate fresh records automatically.

### Privacy & Storage
- Only claim text and public evidence URLs are stored
- No personal information is collected or stored
- Storage is local and can be migrated to a database in the future
- All data is stored in JSON format for easy inspection and portability

### Configuration
- Storage directory can be customized when initializing `ClaimStorageManager`
- Credibility scoring is integrated into LLM prompts and applies to both REAL and FAKE classifications
- Embedding storage can be disabled by omitting a SentenceTransformer model (the manager falls back to skipping embeddings if the model cannot load)

**Testing Recommendations:**
- Test REAL, FAKE, and UNSURE claims to see classification, evidence, and credibility scores
- Validate that Bengali and English inputs both produce bilingual outputs
- Inspect generated JSON files to ensure evidence links, explanations, and timestamps are recorded correctly

# Evaluation Metrics

The following tables summarize performance across languages and input types using different LLMs and strategies.  

Strategy 1: Multi-query generation from the input claim, followed by document retrieval and summarization. The resulting summary was then passed to the verdict-generation prompt (Judge Prompt).  
Strategy 2: Multi-query generation and document retrieval, with raw retrieved documents passed directly to the verdict-generation prompt, skipping summarization.  
Strategy 3: Direct retrieval using the original claim (no rephrasing), followed by summarization and verdict generation.
Each combination of model and strategy was evaluated based on:

## LLM Evaluation — English Claims (Strategy 3)

| Model	                                       |      Strategy 1   |    Strategy 2	   |    Strategy 3    |
|----------------------------------------------|----------------------------------------------------------|
                                               | TC	 | F1R  | F1F  | TC  | F1R  | F1F  | TC  | F1R  | F1F |
|---------------------------------------------------------------------------------------------------------|
| llama3-8b-8192	                           | 58% | 0.82 | 0.53 | 46% | 0.93 | 0.38 | 0.64| 0.88 | 0.71|
| qwen/qwen3-32b	                           | 63% | 0.89	| 0.6  | 33% | 0.92 | 0.33 | 0.72| 0.9	| 0.74|
| mistral-saba-24b	                           | 63% | 0.9	| 0.59 | 33% | 0.91	| 0.33 | 0.67| 0.9	| 0.68|
| deepseek-r1-distill-llama-70b	               | 65% | 0.9	| 0.63 | 31% | 0.93	| 0.32 | 0.69| 0.9	| 0.69|
| meta-llama/llama-4-scout-17b-16e-instruct	   | 67% | 0.9	| 0.65 | 30% | 0.94	| 0.34 | 0.7 | 0.91	| 0.71|
| meta-llama/llama-4-maverick-17b-128e-instruct| 68% | 0.9	| 0.66 | 32% | 0.92	| 0.38 | 0.52| 0.95	| 0.65|
| qwen-qwq-32b	                               | 84% | 0.9	| 0.88 | 32% | 0.93	| 0.4  | 0.67| 0.91	| 0.73|


TC - TotalCoverage, F1R - F1 Score(Real), F1F - F1 Score(Fake)

## LLM Evaluation — Regional Languages (Hindi/Kannada)

| Model                                           | Coverage | F1 Score (Real) | F1 Score (Fake) |
|------------------------------------------------|----------|------------------|------------------|
| llama3-8b-8192                                 | 0.68     | 0.92             | 0.69             |
| qwen/qwen3-32b                                  | 0.75     | 0.91             | 0.76             |
| mistral-saba-24b                                | 0.68     | 0.90             | 0.60             |
| deepseek-r1-distill-llama-70b                   | 0.70     | 0.88             | 0.58             |
| meta-llama/llama-4-scout-17b-16e-instruct       | 0.70     | 0.89             | 0.55             |
| meta-llama/llama-4-maverick-17b-128e-instruct   | 0.66     | 0.89             | 0.55             |
| qwen-qwq-32b                                     | 0.66     | 0.89             | 0.57            |


## LLM Evaluation — Audio Inputs (Multilingual)

| Model            | Coverage | F1 Score (Real) | F1 Score (Fake) |
|------------------|----------|------------------|------------------|
| llama3-8b-8192   | 0.22     | 0.57             | 0.28             |
| mistral-saba-24b | 0.22     | 0.95             | 0.52             |
| qwen-qwq-32b     | 0.51     | 0.66             | 0.64             |

## SarvamAI Speech & Translation Performance

| Metric        | Score   |
|---------------|---------|
| WER           | 0.2887  |
| CER           | 0.0887  |
| BLEU Score    | 0.2027  |
| METEOR Score  | 0.4949  |
| BERTScore     | 0.9149  |
