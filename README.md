## AI Research Tools

There are currently 3 tools in the AI Research tools platform.

[YouTube Video](https://www.youtube.com/watch?v=p4ygW6npE3Y)

1. Web Researcher - For finding new research papers on the web
2. References Bot - For extracting references from PDF papers into APA format using a LLM
3. RAG Chatbot - A tool to convert your PDF, markdown or text files to a vector database and allow you to chat over them using Retrieval Augmented Generation (RAG).  Includes advanced LLM re-ranking techniques, etc.
4. Knowledge Distiller - A tool for extracting scientific literature meta-data from a corpus of PDF, markdown or text files and then through filtering and prompting creating a distilled data set for use with LLMs.  Essentially, human-in-the-loop distillation of scientific literature content.

### Preconfiguration

There are a couple of APIs that you can sign up for if you want commercial level LLMs (and also for Google Searching for Web Researcher (required)).

- [Google [Custom] Search Engine](https://programmablesearchengine.google.com/about/)
  - Create a custom search engine ID and record it.
- [Optional: Setup and Fund an Anthropic Account](https://console.anthropic.com/login?returnTo=%2F%3F)
  - Record the API key
- [Optional: Setup and Fund OpenAI API Key](https://platform.openai.com/api-keys)
  - Record the API key
- [Optional: Setup and Fund Google Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key)
  - Record the API key

**You will need access to a hosted Ollama instance if you want to use Web Researcher and References Bot.** A server with a NVidia RTX 3090+ with 8+ GB of VRAM is recommended.  The RAG Chatbot can work with OpenAI ChatGPT or Anthropic Claude.

You can install Ollama at [ollama.ai](http://ollama.ai). You will need to download the `phi4:14b` model (or `llama3.1:8b`, etc.).

`ollama pull phi4:14b`

We also highly recommend using Anaconda for setting up a virtual python environment to run the apps. [Anaconda download](https://www.anaconda.com/download).

### Installation

1. Download (or clone) the repository.
2. Create your anaconda environment:

```
    conda create --name ai_research
    conda activate ai_research
```

3. Install the required libraries:

`pip install streamlit requests python-dotenv PyPDF2 beautifulsoup4 pandas langchain langchain-community langchain-anthropic langchain-openai langchain-ollama langchain-chroma chromadb shutil jupyter`

4. In each folder, rename .env_sample to .env. Edit each .env file and put in your API key values, etc.
5. Run the appropriate .sh (Linux/Mac) or .bat (Windows) file. This will launch the respective web interface.
