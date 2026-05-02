import os
import json
import time
from pageindex import PageIndexClient

pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))

# 1. Invio e recupero ID
result = pi_client.submit_document("./test/CELEX_32006L0054_EN_TXT.pdf")
doc_id = result["doc_id"]

# 2. Loop di controllo
while True:
    tree_result = pi_client.get_tree(doc_id) # Questo è già un dizionario Python
    
    if tree_result.get("status") == "completed":
        # 3. Scrittura del dizionario su file .json
        with open("pageindex-result-IT.json", "w", encoding="utf-8") as f:
            json.dump(tree_result, f, indent=4, ensure_ascii=False)
            
        print("File 'pageindex-result-EN.json' salvato con successo.")
        break
    
    elif tree_result.get("status") == "failed":
        print("Errore nel processing.")
        break
        
    print("Ancora in elaborazione...")
    time.sleep(5)

