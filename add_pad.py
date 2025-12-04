from transformers import AutoModelForCausalLM, AutoTokenizer

ORIGINAL_MODEL_PATH = "./models/lotus-12B"
FIXED_MODEL_PATH = "./models/lotus-12B-fixed" # The new folder for the corrected model

print(f"Loading original model from: {ORIGINAL_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_PATH)

print("Applying the pad_token fix...")
# The fix that needs to be saved to disk
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Saving corrected model and tokenizer to: {FIXED_MODEL_PATH}")
# Save the corrected versions to a new directory
model.save_pretrained(FIXED_MODEL_PATH)
tokenizer.save_pretrained(FIXED_MODEL_PATH)

print("\nModel fixing complete.")
