import pandas as pd
from sarvamai import SarvamAI

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

class TranslationEvaluator:
    def __init__(self, api_key):
        self.client = SarvamAI(api_subscription_key=api_key)
    
    def translate_text(self, text, source_lang="auto", target_lang="en-IN"):
        """
        Translate a single text using Sarvam AI
        """
        try:
            translation = self.client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang
            )
            return translation.translated_text
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
    
    def compute_bleu(self, df, original_col='trans_orig', predicted_col='trans_pred'):
        """
        Compute corpus BLEU score for translations in DataFrame
        
        Args:
            df: DataFrame with original and predicted translations
            original_col: Column name for reference translations
            predicted_col: Column name for predicted translations
        
        Returns:
            Single corpus BLEU score (float)
        """
        smoothing = SmoothingFunction()
        
        # Prepare all references and hypotheses
        references = []
        hypotheses = []
        
        for _, row in df.iterrows():
            reference = nltk.word_tokenize(row[original_col].lower())
            hypothesis = nltk.word_tokenize(row[predicted_col].lower())
            
            references.append([reference])  # Wrap in list for corpus_bleu
            hypotheses.append(hypothesis)
        
        # Compute corpus BLEU
        corpus_score = corpus_bleu(references, hypotheses, 
                                  smoothing_function=smoothing.method1)
        
        return corpus_score
    
    def compute_meteor(self, df, original_col='trans_orig', predicted_col='trans_pred'):
        """
        Compute average METEOR score for translations in DataFrame
        
        Args:
            df: DataFrame with original and predicted translations
            original_col: Column name for reference translations
            predicted_col: Column name for predicted translations
        
        Returns:
            Average METEOR score (float)
        """
        meteor_scores = []
        
        for _, row in df.iterrows():
            reference = nltk.word_tokenize(row[original_col].lower())
            hypothesis = nltk.word_tokenize(row[predicted_col].lower())
            
            meteor_val = meteor_score([reference], hypothesis)
            meteor_scores.append(meteor_val)
        
        return np.mean(meteor_scores)
    
    def compute_bertscore(self, df, original_col='trans_orig', predicted_col='trans_pred'):
        """
        Compute average BERTScore F1 for translations in DataFrame
        
        Args:
            df: DataFrame with original and predicted translations
            original_col: Column name for reference translations
            predicted_col: Column name for predicted translations
        
        Returns:
            Average BERTScore F1 (float)
        """
        references = df[original_col].tolist()
        predictions = df[predicted_col].tolist()
        
        P, R, F1 = bert_score(predictions, references, lang='en')
        
        return float(F1.mean())
    
    def translate_and_evaluate_from_csv(self, csv_file='translation_test_final.csv'):
        """
        Translate entire CSV file and evaluate results
        
        Args:
            csv_file: Path to CSV file with 'hindi' and 'english' columns
        
        Returns:
            DataFrame with translations and evaluation scores
        """
        # Read the test file
        try:
            df_test = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File '{csv_file}' not found")
            return None
        
        # Initialize results DataFrame
        df_translated = pd.DataFrame(columns=['hindi', 'trans_pred', 'trans_orig'])
        
        print(f"Translating {len(df_test)} texts...")
        
        # Translate each row
        for index, row in df_test.iterrows():
            print(f"Processing row {index + 1}/{len(df_test)}")
            
            try:
                translation = self.client.text.translate(
                   input=row['hindi'],
                   source_language_code="auto",
                   target_language_code="en-IN"
                )
                
                # Add to results DataFrame
                new_row = {
                    'hindi': row['hindi'], 
                    'trans_pred': translation.translated_text,
                    'trans_orig': row['english']
                }
                df_translated = pd.concat([df_translated, pd.DataFrame([new_row])], ignore_index=True)
                
            except Exception as e:
                print(f"Error translating row {index}: {e}")
                # Add failed translation as None
                new_row = {
                    'hindi': row['hindi'], 
                    'trans_pred': None,
                    'trans_orig': row['english']
                }
                df_translated = pd.concat([df_translated, pd.DataFrame([new_row])], ignore_index=True)
        
        return df_translated
    
    def evaluate_translations(self, df):
        """
        Evaluate translations in DataFrame and return scores
        
        Args:
            df: DataFrame with 'trans_orig' and 'trans_pred' columns
        
        Returns:
            Dictionary with BLEU, METEOR, and BERTScore F1 scores
        """
        # Remove rows with None translations
        df_clean = df.dropna(subset=['trans_pred', 'trans_orig'])
        
        if len(df_clean) == 0:
            print("Error: No valid translations to evaluate")
            return None
        
        print(f"Evaluating {len(df_clean)} translations...")
        
        bleu_score = self.compute_bleu(df_clean)
        meteor_score = self.compute_meteor(df_clean)
        bertscore_f1 = self.compute_bertscore(df_clean)
        
        results = {
            'bleu_score': bleu_score,
            'meteor_score': meteor_score,
            'bertscore_f1': bertscore_f1,
            'total_translations': len(df_clean),
            'failed_translations': len(df) - len(df_clean)
        }
        
        return results
    
    def full_evaluation_pipeline(self, csv_file='hindi_english_test_final.csv'):
        """
        Complete pipeline: translate CSV and evaluate results
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            Tuple of (DataFrame with translations, evaluation results)
        """
        # Step 1: Translate
        df_translated = self.translate_and_evaluate_from_csv(csv_file)
        
        if df_translated is None:
            return None, None
        
        # Step 2: Evaluate
        evaluation_results = self.evaluate_translations(df_translated)
        
        # Step 3: Display results
        if evaluation_results:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"BLEU Score: {evaluation_results['bleu_score']:.4f}")
            print(f"METEOR Score: {evaluation_results['meteor_score']:.4f}")
            print(f"BERTScore F1: {evaluation_results['bertscore_f1']:.4f}")
            print(f"Successfully translated: {evaluation_results['total_translations']}")
            print(f"Failed translations: {evaluation_results['failed_translations']}")
        
        return df_translated, evaluation_results

if __name__ == "__main__":
    """
        This code is to evaluate translations using Sarvam AI.
        To run this code, you need to have a valid Sarvam AI API key and a CSV file named translation_test_final.csv
    """

    # Initialize evaluator
    #evaluator = TranslationEvaluator(api_key='')
    #df_results, eval_scores = evaluator.full_evaluation_pipeline('translation_test_final.csv')
    