from transformers import pipeline

def get_english_summarizer(model_path):
    summarizer = pipeline("summarization", model=model_path)
    return summarizer

def english_summarize(text, summarizer):
    summary = summarizer(text)
    return summary[0]['summary_text']