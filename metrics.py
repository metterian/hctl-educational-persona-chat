from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load dataset and metric
dataset = load_dataset('glue', 'cola')
metric = load_metric("matthews_correlation")

# load tokenizer and model
cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")


