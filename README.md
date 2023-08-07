# AI Web Scraper Chat Bot

This repo performs 3 functions:
1. Scrapes a web site and follows links under the same path up to a maximum depth and outputs the scraped data to the `data` directory.
2. Runs an embedding model to embed the text into a Chroma vector database using disk storage (`chroma_db` directory)
3. Runs an OpenAI Chat Bot that uses the embeddings to answer questions about the website

`main.py` runs all 3 functions. Once the scraper and embbedings have completed once, they do not need to be run again. You can simply run the `chat.py` file.

## How to Run

Note: The chatbot requires an OpenAI key to be set in the environment variable `OPENAI_API_KEY`

Download the code:

```
git clone asdfad
cd asfasd
```

Optional - setup a virtual environment:
```
vitualenv venv
source venv/bin/activate
```

Download dependencies and run the functions
```
pip install -r requirements.txt
python main.py
```

After running `main.py` once, the embeddings are persisted, so you can simply use `python chat.py` to start a new chat.