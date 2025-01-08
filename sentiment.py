from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
labels = ['positive', 'negative', 'neutral']

def predict_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors='pt', padding=True).to(device)
        logits = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])['logits']  # shape: (batch_size, num_classes)
        result = torch.nn.functional.softmax(torch.sum(logits, 0), dim=-1)  # softmax the sum for each class across the batch
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, 'neutral'

