from fastapi import FastAPI, HTTPException, Request
from english_summary import get_english_summarizer, english_summarize
from bangla_summary import get_bangla_summarizer
from translate import get_translator

app = FastAPI()

model_paths = {
    "english": "H:\CSE400\Pretrained Models\FinSummT5-v2\checkpoint-2500",
    "bangla" : "H:\CSE400\Pretrained Models\BanglaFinSumm",
    "e2b": "H:\CSE400\Pretrained Models\e2b_mt",
    "b2e" : r"H:\CSE400\Pretrained Models\b2e_mt"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the summarization API!"}

@app.post("/summarize/{language}/")
async def create_summary(language: str, request: Request):
    try:
        body = await request.json()
        text = body.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided for summarization.")
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        tokens = text.split()
        if len(tokens) > 512:
            text = ' '.join(tokens[:512])
        
        if language == "english":
            summarizer_model_path = model_paths[language]
            summarizer = get_english_summarizer(summarizer_model_path)
            summary_text = english_summarize(text, summarizer)
            return {"summary": summary_text}
        
        elif language == "bangla":
            summarizer_model_path = model_paths[language]
            summarizer = get_bangla_summarizer(summarizer_model_path)
            summary_text = summarizer(text)
            for token in ["<pad>", "</eos>", "<sos/>", "<eos>", "<sos>", "</s>"]:
                summary_text = summary_text.replace(token, "")
            return {"summary": summary_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/{translate}/")
async def create_summary(translate: str, request: Request):
    try:
        body = await request.json()
        text = body.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided for summarization.")

        translator_model_path = model_paths[translate]
        translator = get_translator(translator_model_path)
        translated_text = translator(text)
        for token in ["<pad>", "</eos>", "<sos/>", "<eos>", "<sos>", "</s>"]:
            translated_text = translated_text.replace(token, "")
        return {"translation": translated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))