from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
labels = ['positive', 'negative', 'neutral']

def predict_sentiment(news) -> tuple[float]:
    if news:
        tokens = tokenizer(news, return_tensors='pt', padding=True).to(device)
        logits = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])['logits']  # shape: (batch_size, num_classes)
        softmaxes = torch.nn.functional.softmax(logits, dim=-1)
        sum_softmaxes = torch.mean(softmaxes, dim=0)
        return sum_softmaxes[0].item(), sum_softmaxes[2].item()
    else:
        return 0, 0

