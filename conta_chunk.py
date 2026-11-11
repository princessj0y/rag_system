#!/usr/bin/env python3
"""
Conta i chunk e ne misura la lunghezza, leggendo la cache di ChromaDB creata da chunking_evaluation.py.
Uso (dalla cartella legal_rag_system, con l'ambiente uv attivo):
    uv run python conta_chunk.py ./chroma_eval_cache  chunk_stats.csv
Ogni sottocartella di chroma_eval_cache e' un esperimento: <testo>_<llm>_<embedding>_<strategia>_<IT|EN>.
Per ogni esperimento scrive: numero di chunk, lunghezza media/mediana/massima (caratteri e parole).
Se la cache non c'e' piu', va rigenerata rilanciando la valutazione (le strategie agentiche e semantiche richiedono Ollama).
"""
import csv, pathlib, statistics as st, sys
import chromadb

radice = pathlib.Path(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else "chunk_stats.csv"
righe = []
for d in sorted(p for p in radice.iterdir() if p.is_dir()):
    try:
        col = chromadb.PersistentClient(path=str(d)).get_collection("eval_collection")
        docs = col.get(include=["documents"])["documents"]
    except Exception as e:
        print("saltata", d.name, "->", e); continue
    if not docs:
        continue
    c = [len(x) for x in docs]; w = [len(x.split()) for x in docs]
    righe.append(dict(esperimento=d.name, n_chunk=len(docs), car_media=round(st.mean(c)), car_mediana=round(st.median(c)), car_max=max(c),
                      parole_media=round(st.mean(w)), parole_max=max(w)))
with open(out, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(righe[0])); wr.writeheader(); wr.writerows(righe)
print(f"{len(righe)} esperimenti -> {out}")
