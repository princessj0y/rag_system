import os
import json
import time
from pageindex import PageIndexClient

pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))
docs = [
    "./test/CELEX_32006L0054_IT_TXT.pdf",
    "./test/CELEX_32006L0054_EN_TXT.pdf",
    "./test/Strategia_italiana_per_l_Intelligenza_artificiale_2024-2026.pdf",
    "./test/Crime_and_Punishment_Critical_Analysis.pdf",
    "./test/0 -Avviso Pubblico Pro.vi 2025.2026 sito-signed.pdf",
]

# 1. Invio e recupero ID
for doc_name in docs:
    print(f'Submitting {doc_name}...')
    result = pi_client.submit_document(doc_name)
    doc_id = result["doc_id"]

    # 2. Loop di controllo
    while True:
        tree_result = pi_client.get_tree(doc_id) # Questo è già un dizionario Python
        
        if tree_result.get("status") == "completed":
            output_file = f"{doc_name}.pageindex.json"
            # 3. Scrittura del dizionario su file .json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tree_result, f, indent=4, ensure_ascii=False)
                
            print(f"File '{output_file}' salvato con successo.")
            break
        
        elif tree_result.get("status") == "failed":
            print("Errore nel processing.")
            break
            
        print("Ancora in elaborazione...")
        time.sleep(5)

