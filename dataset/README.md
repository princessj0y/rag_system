# Descrizione dei dataset

## Test set 1 (single‑section grounding):

- 20 short
- 20 medium
- 20 long

## Test set 2 (multi‑section reasoning):

- 20 short
- 20 medium
- 20 long

## Struttura

```json
{
  "id": "...",
  "test_set": 1 | 2,
  "answer_type": "short | medium | long",
  "difficulty": "easy | medium | hard",
  "question": "...",
  "answer": "...",
  "ground_truth": "...",
  "article_refs": ["Art. X", "Titolo Y"],
  "contexts": ["..."],
  "adversarial": true | false
}
```