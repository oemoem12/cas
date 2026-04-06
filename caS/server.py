from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from .model_manager import ModelManager

app = FastAPI(title="caS API", version="0.1.0")
manager = ModelManager()


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


class PullRequest(BaseModel):
    model: str
    source: Literal["huggingface", "hf-mirror", "modelscope"] = "huggingface"
    gguf: bool = False


@app.post("/api/pull")
def pull(req: PullRequest):
    manager.pull(req.model, source=req.source, gguf=req.gguf)
    return {"status": "ok", "model": req.model, "source": req.source, "gguf": req.gguf}


@app.get("/api/models")
def list_models():
    return {"models": manager.list_models(), "details": manager.models}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        model, tokenizer = manager.load(req.model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if hasattr(input_ids, "to"):
        input_ids = input_ids.to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None and hasattr(attention_mask, "to"):
        attention_mask = attention_mask.to(model.device)
    generate_kwargs = {"input_ids": input_ids, "max_new_tokens": req.max_tokens}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    if req.temperature > 0:
        generate_kwargs["temperature"] = req.temperature
        generate_kwargs["do_sample"] = True
    outputs = model.generate(**generate_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
