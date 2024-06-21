import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from normalizer import normalize

def get_bangla_summarizer(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    
    def bangla_summarize(input_sentence):
        input_ids = tokenizer(normalize(input_sentence), return_tensors="pt").input_ids.to(device)
        generated_tokens = model.generate(input_ids)
        decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]
        return decoded_tokens
    
    return bangla_summarize