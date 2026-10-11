import torch
from transformers import pipeline

class FinancialNERPredictor:
    def __init__(self, model_path: str = "./models/ner/sec_bert_final", device: str = None):
        self.model_path = model_path

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device.lower() in ["gpu", "cuda"] else -1

        self.pipeline = pipeline(
            "token-classification",
            model=self.model_path,
            tokenizer=self.model_path,
            aggregation_strategy="simple",
            device=self.device
        )

    def predict(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []

        output = self.pipeline(text)
        if not output:
            return []
        
        results = []
        previous = None
        current_scores = []

        for current in output:
            current_label = current.get('entity_group', current.get('entity'))

            if previous is not None and current_label == previous['label'] and current['start'] <= previous['end'] + 1:
                previous['end'] = current['end']
                previous['text'] = text[previous['start']:current['end']]

                current_scores.append(current['score'])
                previous['score'] = round(float(sum(current_scores) / len(current_scores)), 4)

            else:
                if previous is not None:
                    results.append(previous)

                current_scores = [current['score']]
                previous = {
                    "text": text[current['start']:current['end']],
                    "label": current_label,
                    "start": current['start'],
                    "end": current['end'],
                    "score": round(float(current["score"]), 4)
                }


        if previous is not None:
            results.append(previous)

        return results



if __name__ == "__main__":
    import sys

    default_sec_text = (
        "As of December 31, 2021, the company had a maximum borrowing capacity of $ 150.0 million "
        "under its revolving credit facility. The effective income tax rate was 21.5 %."
    )

    text = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else default_sec_text

    print("\n" + "=" * 75)
    print(" FINANCIAL DOCUMENT ANALYSIS — SEC-BERT ENTITY EXTRACTION")
    print("=" * 75)
    print(f"\n[Input Text]:\n\"{text}\"\n")

    predictor = FinancialNERPredictor()
    entities = predictor.predict(text)

    print("-" * 75)
    if not entities:
        print(" No financial entities detected.")
    else:
        print(f" Extracted Financial Entities ({len(entities)} found):")
        print("-" * 75)
        for i, ent in enumerate(entities, 1):
            print(f" {i}. Extracted Value : \"{ent['text']}\"")
            print(f"    XBRL Concept    : {ent['label']}")
            print(f"    Confidence      : {ent['score']:.2%}")
            print(f"    Character Span  : [{ent['start']}:{ent['end']}]")
            print()
    print("=" * 75 + "\n")

        