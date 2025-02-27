from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

mixtral_llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.01,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text-generation"  # Explicitly specify the task
)