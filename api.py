from fastapi import FastAPI
import uvicorn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from typing import List,  Literal
from pydantic import BaseModel
from scipy.special import softmax

# Setup
model_path = f"GhylB/Sentiment_Analysis_DistilBERT"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Functions

# Preprocess text (username and link placeholders)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}

    return scores

# INPUT MODELING
class ModelInput(BaseModel):
    """Modeling of one input data in a type-restricted dictionary-like format

    column_name : variable type # strictly respect the name in the dataframe header.

    eg.:
    =========
    customer_age : int
    gender : Literal['male', 'female', 'other']
    """

    tweet : str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/tweet")
async def run(input: ModelInput):
    
    result = sentiment_analysis(text=input.tweet)

    return {
        "input_text": input.tweet,
        "confidence_scores":result
    }


if __name__ == "__main__":
    uvicorn.run("api:app", 
                reload=True
                )
