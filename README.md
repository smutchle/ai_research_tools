## AI Research Tools

There are currently 4 tools in the AI Research tools platform.

[YouTube Video](https://www.youtube.com/watch?v=p4ygW6npE3Y)

1. Web Researcher - For finding new research papers on the web (i.e. web scraping)
2. References Bot - For extracting references from PDF papers into APA format using a LLM
3. RAG Chatbot - A tool to convert your PDF, markdown or text files to a vector database and allow you to chat over them using advanced Retrieval Augmented Generation (RAG). Includes advanced LLM re-ranking techniques, etc.
4. Knowledge Distiller - A tool for extracting scientific literature meta-data from a corpus of PDF, markdown or text files and then through filtering and prompting creating a distilled data set for use with LLMs. Essentially, human-in-the-loop distillation of scientific literature content. This is useful when you have a large context window LLM but need some level of distillation of many papers.

### Preconfiguration

There are a couple of APIs that you can sign up for if you want commercial level LLMs (and also for Google Searching for Web Researcher (required)).

- [Google [Custom] Search Engine](https://programmablesearchengine.google.com/about/)
  - Create a custom search engine ID and record it.
- [Recommended: Setup and Possibly Fund Google Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key)
  - Record the API key
- [Optional: Setup and Fund an Anthropic Account](https://console.anthropic.com/login?returnTo=%2F%3F)
  - Record the API key
- [Optional: Setup and Fund OpenAI API Key](https://platform.openai.com/api-keys)
  - Record the API key

**The RAG Chatbot can work with Google Gemini, OpenAI ChatGPT, Anthropic Claude or a local Ollama model.**

**You will need access to a hosted Ollama instance if you want to use Web Researcher and References Bot.** A server with a NVidia RTX 3090+ with 8+ GB of VRAM is recommended.

You can install Ollama at [ollama.ai](http://ollama.ai). You will need to download the `gemma3:12b` model (or `phi4:14b`, `llama3.1:8b`, etc.).

```bash
ollama pull gemma3:12b
```

### Installation

0. [Install git version control software](https://git-scm.com/downloads)
1. [Install anaconda for virtual python environments](https://www.anaconda.com/download)
2. Download (or clone) the repository.

```bash
md ai_tools
cd ai_tools
git clone https://github.com/smutchle/ai_research_tools
```

3. Create your anaconda environment:

```bash
conda create --name ai_research
conda activate ai_research
```

4. Install the required libraries:

```bash
pip install streamlit pandas python-dotenv PyPDF2 requests beautifulsoup4 urllib3 langchain langchain-community langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-chroma chromadb shutil jupyterlab
```

5. In each folder, rename `.env_sample` to `.env`. Edit each `.env` file and put in your API key values, etc.

OpenAI uses a different embedding size so you must using the OpenAI embedding model with the OpenAI LLM. To do this, set OpenAI as the default provider in `.env`:

```
EMBEDDING_PROVIDERS=OpenAI,Ollama,Google
LLM_PROVIDERS=OpenAI,Ollama,Anthropic,Google
```

This will cause OpenAI to be the default model and synch the two options.

**When rebuilding the vector database with a different embedding model**, stop the chatbot, delete the `vectorstore` subdirectory in your docs dir and then restart the chatbot. This is due to an issue with chromadb initialization.

6. Run the appropriate .sh (Linux/Mac) or .bat (Windows) file. This will launch the respective web interface.

```

```
