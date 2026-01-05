import pandas as pd
from sarvamai import SarvamAI
import jiwer
import numpy as np
import os
import mimetypes

class SpeechToTextEvaluator:
    def __init__(self, api_key):
        self.sarvam_api_key = api_key

    def transcribe_audio(self, audio):
        """
           Description: This function trascibes audio using SarvamAI STT model
        """
        try:
            client = SarvamAI(api_subscription_key = self.sarvam_api_key)
            mime_type, _ = mimetypes.guess_type(audio)

            with open(audio, "rb") as f:
                response = client.speech_to_text.transcribe(
                    file=("audio.mp3", f, mime_type or "audio/mpeg"),
                    model="saarika:v2.5",
                    language_code="unknown"
                )
            ret_var = response.transcript
            ret_lang = response.language_code
        except Exception as e:
            print(f"Error during translation: {e}")
            ret_var = ''

        return ret_var, ret_lang

    def transcribe_audio_from_csv(self, csv_file='train_data.csv'):
        """
        Transcribe entire CSV file and evaluate results

        Args:
            csv_file: Path to CSV file with 'audio_path' and 'transcript' columns

        Returns:
            DataFrame with transcriptions and evaluation scores
        """

        try:
            df_test = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File '{csv_file}' not found")
            return None

        df_transcribed = pd.DataFrame(columns=['file_name', 'stt_original', 'stt_predicted'])

        for index, row in df_test.iterrows():
            print(f"Processing row {index + 1}/{len(df_test)}")

            path_det = os.getcwd() + '\\audio\\' + row['file_name'] + '.wav'

            trans,lang1 = self.transcribe_audio(path_det)
            new_row = {
                    'file_name': row['file_name'],
                    'stt_original': row['stt_original'],
                    'stt_predicted': trans
                }

            df_transcribed = pd.concat([df_transcribed, pd.DataFrame([new_row])], ignore_index=True)

        return df_transcribed

    def compute_wer(self, df, original_col='stt_original', predicted_col='stt_predicted'):
        """
        Compute Word Error Rate for ASR in DataFrame

        Args:
            df: DataFrame with original and predicted transcriptions
            original_col: Column name for reference transcriptions
            predicted_col: Column name for predicted transcriptions

        Returns:
            Overall WER score (float)
        """
        # Get all reference and hypothesis texts
        references = df[original_col].tolist()
        hypotheses = df[predicted_col].tolist()

        # Compute WER using jiwer
        wer_score = jiwer.wer(references, hypotheses)

        return wer_score

    def compute_cer(self, df, original_col='stt_original', predicted_col='stt_predicted'):
        """
        Compute Character Error Rate for ASR in DataFrame

        Args:
            df: DataFrame with original and predicted transcriptions
            original_col: Column name for reference transcriptions
            predicted_col: Column name for predicted transcriptions

        Returns:
            Overall CER score (float)
        """
        # Get all reference and hypothesis texts
        references = df[original_col].tolist()
        hypotheses = df[predicted_col].tolist()

        # Compute CER using jiwer
        cer_score = jiwer.cer(references, hypotheses)

        return cer_score

    def evaluator_pipeline(self, csv_file='train_data.csv'):
        """
        Evaluate the entire pipeline: transcribe audio and compute WER and CER

        Returns:
            DataFrame with transcriptions and WER score
        """
        transcribed_df = self.transcribe_audio_from_csv(csv_file)
        if transcribed_df is None:
            return None

        wer_score = self.compute_wer(transcribed_df)
        cer_score = self.compute_cer(transcribed_df)

        print(f'\nWER Score: {wer_score:2f}')
        print(f'CER Score: {cer_score:2f}')

        return wer_score, cer_score

if __name__ == "__main__":
    """
        This code is to evaluate the performance of a Speech-to-Text (STT) model using SarvamAI.
        It transcribes audio files listed in a CSV, computes Word Error Rate (WER) and Character Error Rate (CER) for the transcriptions, and provides a summary of the evaluation results.

        To run this code, you need to have a valid SarvamAI API key and a CSV file named 'train_data.csv' with columns 'file_name' and 'stt_original'.
        The 'file_name' column should contain the names of audio files (without extensions) located in an 'audio' directory, and 'stt_original' should contain the expected transcriptions.
        Also, the audio files should be in WAV format and located in the 'audio' directory relative to the script's location.
    """

    #evaluator = SpeechToTextEvaluator(api_key='')
    #wer_score, cer_score = evaluator.evaluator_pipeline('train_data.csv')
    
