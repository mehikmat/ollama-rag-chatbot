Building a Local RAG-Based Chatbot Using
-----------
- ChromaDB as a vector database
- LangChain4J for RAG orchestration in backend,
- Streamlit for Frontend UI
- Ollama for model servicing
- Llama3.2 as an inferencing model
- Apache iText for document parsing

Frameworks and Tools 
---------------------
- ChromaDB to store embeddings
- LangChain for document retrieval and more backend orchestration
- Ollama for running LLMs locally
- Streamlit for an interactive chatbot UI

OLaMA
-----------
 - Download and install ollama from https://ollama.com/download/Ollama-darwin.zip
 - ollama pull ollama3.2 # for inferencing
 - ollama pull mxbai-embed-large # for embedding generation

RAG
-----------
Retrieval Phase
    The AI first searches for relevant data from an external knowledge source such as ChromaDB,web,etc.
Augmentation Phase
    The retrieved information is injected into the AI’s context along with user question before generating a response.
Generation Phase
    The AI model uses both pre-trained knowledge and the retrieved data to generate a context-aware response.

Langchain
-----------
For orchestration of RAG.

Langchain Text Splitters
- CharacterTextSplitter:
  - It splits the text into chunks based on the chunk size given.
  - It does not consider natural breaks like punctuation or whitespace. It will split text exactly at the character count.
  - Use it if your text is random words and has no symantic linking across lines like web page text.
  
- RecursiveCharacterTextSplitter:
  - it splits text into chunks based on given chunk size but it also considers natural breaks not just chunk size.
  - So actual chunk size might be lesser or higher than given chunk size. 
  - Use it if the document has well-defined symantic linking across lines like long paragraph documents.

- ParagraphTextSplitter:
  - Use it if the document has well defined paragraphs and chunk size needed is equals to paragraph size.
  
- RegexTextSplitter:
  - This splitter allows you to split text based on a regular expression pattern.

- SentenceTextSplitter:
  - This splitter splits text by sentences, preserving the structure of the document. 
  - It’s useful when you want to ensure that chunks do not break sentences or disrupt the flow of the text.

- LineTextSplitter:
    - This splitter splits text based on lines, making it useful when you're processing text where each line should be preserved (e.g., CSV or log file).

ChromaDB
-----------
- for vector database