import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def process_chunking_metrics(input_dir: str, output_dir: str):
    # Assicurati che la cartella di output esista, altrimenti la crea
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Scorre tutti i file nella cartella di input
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Legge il CSV
            df = pd.read_csv(input_path)
            
            # 1. Retrieval Score (F1-Score: media armonica tra Precision e Recall)
            # Aggiungiamo un controllo per evitare divisioni per zero in caso di metriche vuote
            df['Retrieval_Score'] = (2 * df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall']).replace(0, np.nan)
            
            # Calculate (1 - NoiseSensitivty) * f1
            inv_noise = (1 - df['NoiseSensitivty']) * df['Retrieval_Score']
            
            # 2. Generation Score (Geometric mean of 3 metrics)
            # Formula: Cube root of (A * B * C)
            df['Generation_Score'] = np.cbrt(df['Faithfullness'] * df['AnswerCorrectness'] * inv_noise)
            
            # 3. Composite Score (Geometric mean of Retrieval and Generation)
            # Formula: Square root of (A * B)
            df['Composite_Score'] = np.sqrt(df['Retrieval_Score'] * df['Generation_Score'])
            
            # Salva il nuovo file CSV nella cartella di output
            df.to_csv(output_path, index=False)
            
            # Ordina e prende i top 3
            top_3 = df.sort_values(by='Composite_Score', ascending=False).head(3)
            
            # Stampa i risultati in console
            print(f"\n--- File: {filename} ---")
            print(top_3[['Method', 'Retrieval_Score', 'Generation_Score', 'Composite_Score']].to_string(index=False))

if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    process_chunking_metrics(input_folder, output_folder)
