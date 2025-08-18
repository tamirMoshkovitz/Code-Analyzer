import os
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Retrieve Hugging Face token from environment variable
hf_token = os.getenv('HF_TOKEN')

# If not set, prompt the user to enter the token
if not hf_token:
    hf_token = input("Please enter your Hugging Face READ token: ")
    os.environ['HF_TOKEN'] = hf_token  # Set the token for the current session

# Authenticate using the token
login(token=hf_token)

# Model configuration
model_id = "google/gemma-3-1b-it"
local_model_dir = "gemma-3-1b-it-model"

# Download and cache the model (one-time)
local_model_path = snapshot_download(
    model_id,
    local_dir=local_model_dir,
    repo_type="model",
)
print("Model downloaded to:", local_model_path)

# Load tokenizer and model with efficient settings
tokenizer = AutoTokenizer.from_pretrained(local_model_path, model_max_length=8192)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path
)

# Create text-generation pipeline with sampling enabled
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,           # Enable sampling
    temperature=0.7,         # Adjust temperature
    top_p=0.9,               # Set top-p
    top_k=40,                # Set top-k
    max_length=8192,
    padding=True,
    truncation=True
)

def get_model_response(prompt: str) -> str:
    return generator(
    prompt,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=True,
    truncation=True
)[0]["generated_text"]