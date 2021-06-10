import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("fine_tune_output")
model = AutoModel.from_pretrained("fine_tune_output")
model.eval()