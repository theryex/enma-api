import os
import uvicorn
import torch
import traceback
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from models import Completion



class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: torch.LongTensor):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        if input_ids.shape[1] >= self.stop_ids.shape[1]:
            last_tokens = input_ids[0, -self.stop_ids.shape[1]:]
            if torch.equal(last_tokens, self.stop_ids[0]):
                return True
        return False


app = FastAPI(
    title="Enma Inference API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


args = {
    "model_path": os.getenv("INFERENCE_MODEL", "./models/lotus-12B-fixed"),
    "port": int(os.getenv("INFERENCE_PORT", 8888))
}

print("--- Configuration ---")
print(f"Model Path: {args['model_path']}")
print(f"Port: {args['port']}")
print("---------------------")



print("Creating 4-bit quantization config...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", 
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model onto GPU 2 with 4-bit quantization...")

model_for_pipeline = AutoModelForCausalLM.from_pretrained(
    args['model_path'],
    quantization_config=quantization_config, 
    device_map="cuda:2",
    use_safetensors=True                      
)

print("Creating text-generation pipeline...")

pipe = pipeline(
    "text-generation",
    model=model_for_pipeline,
    tokenizer=tokenizer
)
print("Model loaded successfully.")



@app.post("/completion")
async def completion(completion_request: Completion):
    try:
        stopping_criteria_list = StoppingCriteriaList()
        if completion_request.stop_sequence:
            stop_sequence_ids = tokenizer(
                completion_request.stop_sequence,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(model_for_pipeline.device)
            
            stopping_criteria_list.append(StopOnTokens(stop_sequence_ids))

        all_bad_words_ids = [[544], [551], [654], [2634, 5190], [11202], [92], [60], [29], [27], [1163], [33], [1214]]
        if completion_request.bad_words:
            tokenized_bad_words = tokenizer(completion_request.bad_words, add_special_tokens=False).input_ids
            all_bad_words_ids.extend(tokenized_bad_words)
        
        result = pipe(
            completion_request.prompt,
            max_new_tokens=completion_request.max_new_tokens,
            temperature=completion_request.temperature,
            top_p=completion_request.top_p,
            top_k=completion_request.top_k,
            typical_p=completion_request.typical_p,
            repetition_penalty=completion_request.repetition_penalty,
            do_sample=completion_request.do_sample,
            penalty_alpha=completion_request.penalty_alpha,
            bad_words_ids=all_bad_words_ids,
            num_return_sequences=completion_request.num_return_sequences,
            stopping_criteria=stopping_criteria_list if stopping_criteria_list else None,
        )
        return result
    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        torch.cuda.empty_cache()
        return {"error": str(e), "details": error_details}



if __name__ == "__main__":
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=args["port"]
    )
