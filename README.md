# Google Generative AI SDK Lab Code

This repository contains Python code demonstrating various functionalities of the Google Generative AI SDK (formerly known as the Vertex AI SDK for Gemini API). This code was developed as part of a Google Cloud Skill Boost Generative AI learning exchange program, specifically covering concepts from the labs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup and Authentication](#setup-and-authentication)
  - [Prerequisites](#prerequisites)
  - [Google Cloud Project ID](#google-cloud-project-id)
  - [Installation](#installation)
  - [Authentication](#authentication)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project serves as a hands-on exploration of the Google Generative AI SDK. It showcases how to interact with Gemini models for a variety of tasks, ranging from basic text generation to more advanced features like multimodal prompts, chat interactions, function calling, batch predictions, and embeddings.

## Features

The `Gen AI SDK.py` script demonstrates the following capabilities of the Google Generative AI SDK:

* **Text Generation:** Sending simple text prompts to the model.
* **Multimodal Prompts:** Generating content from both text and image inputs (using local image files and GCS URIs).
* **System Instructions:** Guiding model behavior with specific system instructions (e.g., as a translator).
* **Model Parameter Configuration:** Adjusting generation parameters like `temperature`, `top_p`, `top_k`, `max_output_tokens`, and `stop_sequences`.
* **Safety Settings:** Configuring safety filters to control harmful content.
* **Multi-turn Chat:** Engaging in conversational interactions with the model, maintaining context.
* **Structured Output Control:** Controlling the model's output format using Pydantic models or raw JSON schemas.
* **Content Streaming:** Receiving generated content in a stream for real-time updates.
* **Asynchronous Requests:** Demonstrating (or adapting from) asynchronous API calls.
* **Token Counting and Computation:** Estimating token usage for prompts.
* **Function Calling:** Enabling the model to suggest and execute external functions based on user intent.
* **Content Caching:** Caching frequently used content (like PDFs) for faster and more efficient retrieval.
* **Batch Prediction:** Submitting multiple requests for content generation in a single batch job and retrieving results from Google Cloud Storage.
* **Text Embeddings:** Generating numerical representations (embeddings) of text for similarity searches and other applications.

## Setup and Authentication

### Prerequisites

* A Google Cloud Platform (GCP) project.
* Billing enabled for your GCP project (required for using Generative AI services).
* The Generative AI APIs enabled in your GCP project.
* Python 3.8 or higher.
* `gsutil` command-line tool (for batch prediction bucket creation).

### Google Cloud Project ID

**Important:** The `PROJECT_ID` variable in the `Gen AI SDK.py` file is a placeholder. You **must** replace `"your-gcp-project-id"` with your actual Google Cloud Project ID. You can find your Project ID in the [Google Cloud Console](https://console.cloud.google.com/) at the top of the page.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kaustuv26/Getting-Started-with-Google-Generative-AI-using-the-Gen-AI-SDK.git](https://github.com/kaustuv26/Getting-Started-with-Google-Generative-AI-using-the-Gen-AI-SDK.git)
    cd Getting-Started-with-Google-Generative-AI-using-the-Gen-AI-SDK # Navigate into the cloned directory
    ```
   
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Linux/macOS
    # venv\Scripts\activate # On Windows
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    (You might need to create a `requirements.txt` file first by running `pip freeze > requirements.txt` after installing dependencies or manually listing them as below).

    Alternatively, you can manually install the main dependencies:
    ```bash
    pip install --upgrade google-genai pandas==2.2.2 Pillow requests fsspec
    ```

### Authentication

This code is designed to run in environments authenticated to Google Cloud.

* **Google Colab:** The script includes `google.colab.auth.authenticate_user()` which handles authentication automatically when run in a Colab notebook.
* **Cloud Shell/Vertex AI Workbench:** If running in Google Cloud Shell or a Vertex AI Workbench notebook, your environment is usually pre-authenticated.
* **Local Machine/Other VMs:**
    * Ensure you have the Google Cloud SDK installed and authenticated:
        ```bash
        gcloud auth application-default login
        ```
    * This will open a browser window to complete the authentication process.

## Usage

1.  **Update `PROJECT_ID`:** Open `Gen AI SDK.py` and replace `"your-gcp-project-id"` with your actual Google Cloud Project ID.
2.  **Run the script:**
    ```bash
    python "Gen AI SDK.py"
    ```
    The script will execute each demonstration in sequence, printing outputs to your console.

    **Note on Batch Prediction:** The batch prediction section attempts to create a new Google Cloud Storage bucket. Ensure your service account or user account has the necessary permissions (`storage.buckets.create`) in your project. If you prefer to use an existing bucket, modify the `BUCKET_URI` variable accordingly.

## Contributing

Feel free to fork this repository, experiment with the code, and contribute improvements!

## License

This project is open-sourced under the [Apache License 2.0](LICENSE).
