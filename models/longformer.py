import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LongformerOpponentModel:
    def __init__(self, model_name: str = "allenai/longformer-base-4096", num_labels: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def encode_batch(self, texts, max_len=512):
        return self.tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    def fit(self, texts, y, epochs=2, lr=2e-5, device="cpu"):
        self.model.to(device).train()
        enc = self.encode_batch(texts)
        labels = torch.tensor(y).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            out = self.model(**{k:v.to(device) for k,v in enc.items()}, labels=labels)
            out.loss.backward()
            opt.step()

    def predict_proba(self, texts, device="cpu"):
        self.model.eval()
        with torch.no_grad():
            enc = self.encode_batch(texts)
            logits = self.model(**{k:v.to(device) for k,v in enc.items()}).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
