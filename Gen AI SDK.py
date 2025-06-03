# Install Google Gen AI SDK
# This line is typically for Colab environments.
# If running locally, you might prefer to install it via your terminal:
# pip install --upgrade --quiet google-genai pandas
%pip install --upgrade --quiet google-genai pandas==2.2.2

# Authenticate your notebook environment (Colab only)
import sys
if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()

# Use Google Gen AI SDK
import datetime
from google import genai
from google.genai.types import (
    CreateBatchJobConfig,
    CreateCachedContentConfig,
    EmbedContentConfig,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    SafetySetting,
    Tool,
)
from IPython.display import Markdown, display
from PIL import Image
import requests
import json
import fsspec # For reading from GCS
import pandas as pd # For DataFrame operations
import time # For batch job polling

# Set Google Cloud project information
# IMPORTANT: Replace "your-gcp-project-id" with your actual Google Cloud Project ID.
PROJECT_ID = "your-gcp-project-id"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Choose a model
MODEL_ID = "gemini-2.0-flash-001"  # @param {type: "string"}

print("--- Sending text prompts ---")
response = client.models.generate_content(
    model=MODEL_ID, contents="What's the largest planet in our solar system?"
)
print(response.text)
# display(Markdown(response.text)) # Optional: Use for rich display in notebooks

print("\n--- Sending multimodal prompts (image from URL) ---")
image = Image.open(
    requests.get(
        "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
        stream=True,
    ).raw
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture.",
    ],
)
print(response.text)

print("\n--- Sending multimodal prompts (Part.from_uri) ---")
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        Part.from_uri(
            file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
            mime_type="image/png",
        ),
        "Write a short and engaging blog post based on this picture.",
    ],
)
print(response.text)

print("\n--- Setting system instruction ---")
system_instruction = """
  You are a helpful language translator.
  Your mission is to translate text in English to French.
"""
prompt = """
  User input: I like bagels.
  Answer:
"""
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
    ),
)
print(response.text)

print("\n--- Configuring model parameters ---")
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)
print(response.text)

print("\n--- Configuring safety filters ---")
prompt = """
    Write a list of 2 things that I might say to the universe after stubbing my toe in the dark.
"""
safety_settings = [
    SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        safety_settings=safety_settings,
    ),
)
print(response.text)

print("\n--- Inspecting safety ratings ---")
if response.candidates:
    print(response.candidates[0].safety_ratings)
else:
    print("No candidates generated, possibly due to safety settings or other issues.")

print("\n--- Starting a multi-turn chat ---")
system_instruction_chat = """
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""
chat = client.chats.create(
    model=MODEL_ID,
    config=GenerateContentConfig(
        system_instruction=system_instruction_chat,
        temperature=0.5,
    ),
)
response = chat.send_message("Write a function that checks if a year is a leap year.")
print(response.text)
response = chat.send_message("Okay, write a unit test of the generated function.")
print(response.text)

print("\n--- Controlling generated output with Pydantic schema ---")
class Recipe(BaseModel):
    name: str
    description: str
    ingredients: list[str]

response = client.models.generate_content(
    model=MODEL_ID,
    contents="List a few popular cookie recipes and their ingredients.",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,
    ),
)
print(response.text)

print("\n--- Parsing response string to JSON (alternative to Pydantic) ---")
# Example of parsing a raw JSON string response
json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))

print("\n--- Controlling generated output with raw JSON schema ---")
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "rating": {"type": "INTEGER"},
            "flavor": {"type": "STRING"},
            "sentiment": {
                "type": "STRING",
                "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            },
            "explanation": {"type": "STRING"},
        },
        "required": ["rating", "flavor", "sentiment", "explanation"],
    },
}

prompt = """
  Analyze the following product reviews, output the sentiment classification and give an explanation.

  - "Absolutely loved it! Best ice cream I've ever had." Rating: 4, Flavor: Strawberry Cheesecake
  - "Quite good, but a bit too sweet for my taste." Rating: 1, Flavor: Mango Tango
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    ),
)
print(response.text)

print("\n--- Generating content stream ---")
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents="Tell me a story about a lonely robot who finds friendship in a most unexpected place.",
):
    print(chunk.text)
    print("*****************")

print("\n--- Sending asynchronous requests (converted to synchronous for linear script) ---")
# Original: response = await client.aio.models.generate_content(...)
# For a linear script, a direct synchronous call is often more straightforward.
# If true async operation is needed, wrap in an async function and use asyncio.run()
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Compose a song about the adventures of a time-traveling squirrel.",
)
print(response.text)

print("\n--- Counting tokens ---")
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)
print(f"Total tokens: {response.total_tokens}")

print("\n--- Computing tokens ---")
response = client.models.compute_tokens(
    model=MODEL_ID,
    contents="What's the longest word in the English language?",
)
print(f"Total tokens: {response.total_tokens}")

print("\n--- Function calling ---")
get_destination = FunctionDeclaration(
    name="get_destination",
    description="Get the destination that the user wants to go to",
    parameters={
        "type": "OBJECT",
        "properties": {
            "destination": {
                "type": "STRING",
                "description": "Destination that the user wants to go to",
            },
        },
        "required": ["destination"] # Added required field
    },
)

destination_tool = Tool(
    function_declarations=[get_destination],
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="I'd like to travel to Paris.",
    config=GenerateContentConfig(
        tools=[destination_tool],
        temperature=0,
    ),
)
if response.candidates and response.candidates[0].content.parts:
    for part in response.candidates[0].content.parts:
        if part.function_call:
            print(part.function_call)
else:
    print("No function call found in the response.")

print("\n--- Creating a cache ---")
system_instruction_cache = """
  You are an expert researcher who has years of experience in conducting systematic literature surveys and meta-analyses of different topics.
  You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.
  Now look at the research paper below, and answer the following questions in 1-2 sentences.
"""

pdf_parts = [
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
        mime_type="application/pdf",
    ),
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
        mime_type="application/pdf",
    ),
]

cached_content = client.caches.create(
    model="gemini-2.0-flash-001",
    config=CreateCachedContentConfig(
        system_instruction=system_instruction_cache,
        contents=pdf_parts,
        ttl="3600s",
    ),
)
print(f"Cache created: {cached_content.name}")

print("\n--- Using a cache ---")
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the research goal shared by these research papers?",
    config=GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)

print("\n--- Deleting a cache ---")
client.caches.delete(name=cached_content.name)
print(f"Cache {cached_content.name} deleted.")

print("\n--- Preparing batch input and output location ---")
INPUT_DATA = "gs://cloud-samples-data/generative-ai/batch/batch_requests_for_multimodal_input_2.jsonl"  # @param {type:"string"}

# IMPORTANT: Ensure you have permissions to create and write to this bucket.
# Replace "your-cloud-storage-bucket" with an existing bucket or let it create a new one.
BUCKET_URI = f"gs://{PROJECT_ID}-gen-ai-batch-output-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # Default to a unique bucket for the project
print(f"Using bucket for batch output: {BUCKET_URI}")

# Create the bucket if it doesn't exist
# This command requires gsutil to be installed and authenticated
try:
    ! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}
    print(f"Bucket {BUCKET_URI} created (or already exists).")
except Exception as e:
    print(f"Could not create bucket {BUCKET_URI}. Ensure gsutil is configured and you have permissions. Error: {e}")


print("\n--- Sending a batch prediction request ---")
batch_job = client.batches.create(
    model=MODEL_ID,
    src=INPUT_DATA,
    config=CreateBatchJobConfig(dest=BUCKET_URI),
)
print(f"Batch job name: {batch_job.name}")

# Poll for job status
print("Waiting for batch prediction job to complete...")
while batch_job.state in ["JOB_STATE_RUNNING", "JOB_STATE_PENDING"]:
    time.sleep(10) # Wait longer for batch jobs
    batch_job = client.batches.get(name=batch_job.name)
    print(f"Job state: {batch_job.state}")

# Check if the job succeeds
if batch_job.state == "JOB_STATE_SUCCEEDED":
    print("Job succeeded!")
    print("\n--- Retrieving batch prediction results ---")
    fs = fsspec.filesystem("gcs")
    # Batch job output is typically in a subdirectory within the destination bucket
    # Look for files like predictions_*.jsonl or predictions.jsonl
    file_paths = fs.glob(f"{batch_job.dest.gcs_uri}/**/*.jsonl") # Adjusted glob pattern for flexibility

    if file_paths:
        # Assuming the first found JSONL file is the one we want
        output_file_uri = f"gs://{file_paths[0]}"
        print(f"Loading results from: {output_file_uri}")
        df = pd.read_json(output_file_uri, lines=True)
        display(df)
    else:
        print("No prediction output files found in the destination bucket.")
else:
    print(f"Batch job failed. Final state: {batch_job.state}, Error: {batch_job.error}")

print("\n--- Getting text embeddings ---")
TEXT_EMBEDDING_MODEL_ID = "text-embedding-005"  # @param {type: "string"}
response = client.models.embed_content(
    model=TEXT_EMBEDDING_MODEL_ID,
    contents=[
        "How do I get a driver's license/learner's permit?",
        "How do I renew my driver's license?",
        "How do I change my address on my driver's license?",
    ],
    config=EmbedContentConfig(output_dimensionality=128),
)
print(response.embeddings)

print("\n--- Script execution complete ---")
