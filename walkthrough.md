# YouTube Assistant

## Objective

In this lab, we will build a "YouTube Assistant" application using LangChain for language model interactions and
Streamlit for the web interface. This project leverages LangChain's powerful capabilities to process and interact with
the content of YouTube videos. By combining various components and features of LangChain, such as document loaders, text
splitters, and vector stores, we will create an assistant capable of answering questions about a specific YouTube video.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Environment Setup

To start, we need to manage sensitive information such as API keys securely. Using a `.env` file is a standard practice
for this purpose.

1. **Create a `.env` file:**
    - This file will store your OpenAI API key. Ensure it is included in your `.gitignore` file to prevent it from being
      committed to your repository.

   Example `.env` file:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Install required packages:**
    - We need several packages for our
      project: `langchain`, `openai`, `streamlit`, `python-dotenv`, `langchain_community`, `youtube-transcript-api`, `tiktoken`,
      and `faiss-cpu`.

   Commands:
   ```bash
   pip install langchain openai streamlit python-dotenv
   ```

   ```bash
   pip install langchain_community
   ```

   ```bash
   pip install youtube-transcript-api
   ```

   ```bash
   pip install tiktoken
   ```

   ```bash
   pip install faiss-cpu
   ```

#### 2. Main Python Script and Helper Module

In the initial setup, we set up the main Python script and a helper module.

1. **Create `langchain_helper.py`:**
    - This script will contain the initial setup and import necessary modules.

   ```python
   # Module to use OpenAI language models.
   from langchain.llms import OpenAI
   # Importing the function to load environment variables from a .env file.
   from dotenv import load_dotenv
   # Module to create prompt templates for language models.
   from langchain import PromptTemplate
   # Module to create chains linking language models and prompt templates.
   from langchain.chains import LLMChain
   # Module to load and process YouTube videos.
   from langchain.document_loaders import YoutubeLoader
   # Module to split text into manageable chunks.
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   # Module to create and manage vector stores using FAISS.
   from langchain.vectorstores import FAISS
   # Module to create embeddings from OpenAI models.
   from langchain.embeddings import OpenAIEmbeddings

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()
   ```

2. **Create `main.py`:**
    - This script will serve as the entry point for our application. For now, it will be empty but ready for future
      additions.

   ```python
   # main.py
   ```

#### Key Concepts

##### 1. Environment Variables

- **Definition:** Environment variables are dynamic values that can affect the way running processes will behave on a
  computer.
- **Usage:** They are used to store configuration settings and sensitive data (like API keys) separately from the
  codebase.

##### 2. `dotenv` Library

- **Purpose:** The `dotenv` library reads key-value pairs from a `.env` file and can set them as environment variables.
- **Installation:** Use `pip install python-dotenv` to install the library.
- **Usage in Code:**
  ```python
  from dotenv import load_dotenv
  load_dotenv()  # Load the variables from .env into the environment
  ```

##### 3. OpenAI Embeddings

- **Definition:** OpenAI embeddings are numerical representations of text that capture its semantic meaning, which can
  be used for various NLP tasks like similarity search, clustering, and more.
- **Usage:** The `OpenAIEmbeddings` class from LangChain generates embeddings for text data.
- **Example:**
  ```python
  from langchain.embeddings import OpenAIEmbeddings

  embeddings = OpenAIEmbeddings()
  ```

##### 4. Other Components

- **OpenAI Language Models:** Used for processing and generating text.
- **Prompt Templates:** Predefined structures for formatting inputs to the language model.
- **LLM Chains:** Links between language models and prompt templates.
- **YouTube Loaders:** Modules for loading and processing YouTube video content.
- **Text Splitters:** Tools for dividing text into manageable chunks.
- **FAISS Vector Stores:** Libraries for creating and managing vector stores for efficient similarity search.

### Step 2: Add YouTube Video Processing and Vector Store Creation

#### Helper Module

In this step, we add functionality to process YouTube videos and create FAISS vector stores from their transcripts.

1. **Update `langchain_helper.py`:**
    - Add the `create_vector_db_from_youtube_url` function to process YouTube videos.
    - Create a FAISS vector store from the video transcripts.

   Updated `langchain_helper.py`:
   ```python
   from langchain.llms import OpenAI
   from dotenv import load_dotenv
   from langchain import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.document_loaders import YoutubeLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import FAISS
   from langchain_community.embeddings import OpenAIEmbeddings
   import os

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()

   def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
       # Create a loader object to fetch the YouTube video transcript.
       loader = YoutubeLoader.from_youtube_url(video_url)

       # Load the transcript of the YouTube video.
       transcript = loader.load()

       # Create a text splitter to divide the transcript into smaller chunks.
       # chunk_size is the maximum size of each chunk, and chunk_overlap is the overlap between chunks.
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

       # Split the transcript into smaller, manageable chunks.
       docs = text_splitter.split_documents(transcript)

       # Create a FAISS vector store from the document chunks, using the specified embeddings.
       db = FAISS.from_documents(docs, embeddings)

       # Return the FAISS vector store containing the embedded document chunks.
       return db

   video_url = "https://youtu.be/A9W6FAQPVuA?si=qshtmH3E_ah9QvbT"
   print(create_vector_db_from_youtube_url(video_url))
   ```
   Run the script for testing and u should found that u got the vector DB

   <img style="height: 200px" src="https://i.imghippo.com/files/LFash1717260729.png" alt="" border="0">

#### Key Concepts

##### 1. YouTube Video Processing

- **Definition:** This involves loading and processing transcripts of YouTube videos to extract and manage the content.
- **Usage:** The `YoutubeLoader` class from LangChain is used to fetch transcripts from YouTube videos.
- **Example:**
  ```python
  loader = YoutubeLoader.from_youtube_url(video_url)
  transcript = loader.load()
  ```

##### 2. Text Splitting

- **Definition:** Splitting large chunks of text into smaller, manageable pieces.
- **Usage:** The `RecursiveCharacterTextSplitter` class from LangChain splits the video transcript into smaller chunks
  for better processing.
- **Example:**
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)
  ```

##### 3. FAISS Vector Stores

- **Definition:** FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of
  dense vectors.
- **Usage:** The `FAISS` class from LangChain is used to create a vector store from the document chunks.
- **Example:**
  ```python
  db = FAISS.from_documents(docs, embeddings)
  ```

##### 4. OpenAI Embeddings

- **Definition:** OpenAI embeddings are numerical representations of text that capture its semantic meaning, used for
  various NLP tasks like similarity search, clustering, and more.
- **Usage:** The `OpenAIEmbeddings` class from LangChain generates embeddings for text data.
- **Example:**
  ```python
  embeddings = OpenAIEmbeddings()
  ```

### Step 3: Add Querying Functionality

#### Helper Module

In this step, we add a function to query the FAISS vector store and generate responses using OpenAI.

1. **Update `langchain_helper.py`:**
    - Add the `get_response_from_query` function to query the FAISS vector store and generate responses.

   Updated `langchain_helper.py`:
   ```python
   from langchain.llms import OpenAI
   from dotenv import load_dotenv
   from langchain import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.document_loaders import YoutubeLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import FAISS
   from langchain_community.embeddings import OpenAIEmbeddings
   import os

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()

   def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
       # Create a loader object to fetch the YouTube video transcript.
       loader = YoutubeLoader.from_youtube_url(video_url)

       # Load the transcript of the YouTube video.
       transcript = loader.load()

       # Create a text splitter to divide the transcript into smaller chunks.
       # chunk_size is the maximum size of each chunk, and chunk_overlap is the overlap between chunks.
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

       # Split the transcript into smaller, manageable chunks.
       docs = text_splitter.split_documents(transcript)

       # Create a FAISS vector store from the document chunks, using the specified embeddings.
       db = FAISS.from_documents(docs, embeddings)

       # Return the FAISS vector store containing the embedded document chunks.
       return db

   def get_response_from_query(db, query, k):
       # Search the query-relevant documents
       docs = db.similarity_search(query, k=k)

       # Combine the page content of the retrieved documents into a single string
       docs_page_content = " ".join([d.page_content for d in docs])

       # Initialize the OpenAI language model with specific parameters
       llm_OpenAI = OpenAI(model="text-davinci-003", temperature=0.8)

       # Define the prompt template for the language model
       prompt_template = PromptTemplate(
           input_variables=['question', 'docs'],
           template="""
           You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

           Answer the following question: {question}
           By searching the following video transcript: {docs}

           Only use the factual information from the transcript to answer the question.

           If you feel like you don't have enough information to answer the question, say "I don't know".

           Your answers should be detailed.
           """
       )

       # Create a language model chain with the defined prompt template
       name_chain = LLMChain(llm=llm_OpenAI, prompt=prompt_template)

       # Generate the response using the language model chain
       response = name_chain({'question': query, 'docs': docs_page_content})
       response = response.replace("\n", "")

       return response

   video_url = "https://youtu.be/A9W6FAQPVuA?si=qshtmH3E_ah9QvbT"
   print(create_vector_db_from_youtube_url(video_url))
   ```

   Run the script for testing
   <img src="https://i.imghippo.com/files/Ab4FO1717315223.png" alt="" border="0">

#### Key Concepts

##### 1. Querying Vector Stores

- **Definition:** Querying vector stores involves searching for documents that are most similar to a given query.
- **Usage:** The `similarity_search` method of the `FAISS` class is used to find documents that match the query.
- **Example:**
  ```python
  docs = db.similarity_search(query, k=k)
  ```

##### 2. Combining Document Content

- **Definition:** Combining document content involves merging the text content of multiple documents into a single
  string for further processing.
- **Usage:** The page content of the retrieved documents is concatenated into a single string.
- **Example:**
  ```python
  docs_page_content = " ".join([d.page_content for d in docs])
  ```

##### 3. Prompt Templates for Query Responses

- **Definition:** A prompt template defines the structure and content of the input provided to the language model to
  generate a response.
- **Usage:** The `PromptTemplate` class from LangChain creates templates for generating responses based on the query and
  document content.
- **Example:**
  ```python
  prompt_template = PromptTemplate(
      input_variables=['question', 'docs'],
      template="""
      You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

      Answer the following question: {question}
      By searching the following video transcript: {docs}

      Only use the factual information from the transcript to answer the question.

      If you feel like you don't have enough information to answer the question, say "I don't know".

      Your answers should be detailed.
      """
  )
  ```

##### 4. Generating Responses with LLM Chains

- **Definition:** An LLM chain links a language model with a prompt template to generate responses.
- **Usage:** The `LLMChain` class from LangChain creates a chain to generate a response based on the query and document
  content.
- **Example:**
  ```python
  name_chain = LLMChain(llm=llm_OpenAI, prompt=prompt_template)
  response = name_chain({'question': query, 'docs': docs_page_content})
  ```

### Step 4: Add YouTube Video Processing and Vector Store Creation (Refined)

#### Helper Module

In this step, we refine the functionality to process YouTube videos and create FAISS vector stores from their
transcripts.

1. **Update `langchain_helper.py`:**
    - Refine the `create_vector_db_from_youtube_url` function to process YouTube videos and create FAISS vector stores.

   Updated `langchain_helper.py`:
   ```python
   from langchain.llms import OpenAI
   from dotenv import load_dotenv
   from langchain import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.document_loaders import YoutubeLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import FAISS
   from langchain_community.embeddings import OpenAIEmbeddings
   import os

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()

   def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
       # Create a loader object to fetch the YouTube video transcript.
       loader = YoutubeLoader.from_youtube_url(video_url)

       # Load the transcript of the YouTube video.
       transcript = loader.load()

       # Create a text splitter to divide the transcript into smaller chunks.
       # chunk_size is the maximum size of each chunk, and chunk_overlap is the overlap between chunks.
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

       # Split the transcript into smaller, manageable chunks.
       docs = text_splitter.split_documents(transcript)

       # Create a FAISS vector store from the document chunks, using the specified embeddings.
       db = FAISS.from_documents(docs, embeddings)

       # Return the FAISS vector store containing the embedded document chunks.
       return db

   video_url = "https://youtu.be/A9W6FAQPVuA?si=qshtmH3E_ah9QvbT"
   print(create_vector_db_from_youtube_url(video_url))
   ```

#### Key Concepts

##### 1. YouTube Video Processing

- **Definition:** This involves loading and processing transcripts of YouTube videos to extract and manage the content.
- **Usage:** The `YoutubeLoader` class from LangChain is used to fetch transcripts from YouTube videos.
- **Example:**
  ```python
  loader = YoutubeLoader.from_youtube_url(video_url)
  transcript = loader.load()
  ```

##### 2. Text Splitting

- **Definition:** Splitting large chunks of text into smaller, manageable pieces.
- **Usage:** The `RecursiveCharacterTextSplitter` class from LangChain splits the video transcript into smaller chunks
  for better processing.
- **Example:**
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)
  ```

##### 3. FAISS Vector Stores

- **Definition:** FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of
  dense vectors.
- **Usage:** The `FAISS` class from LangChain is used to create a vector store from the document chunks.
- **Example:**
  ```python
  db = FAISS.from_documents(docs, embeddings)
  ```

##### 4. OpenAI Embeddings

- **Definition:** OpenAI embeddings are numerical representations of text that capture its semantic meaning, used for
  various NLP tasks like similarity search, clustering, and more.
- **Usage:** The `OpenAIEmbeddings` class from LangChain generates embeddings for text data.
- **Example:**
  ```python
  embeddings = OpenAIEmbeddings()
  ```

### Step 5: Add Querying Functionality for FAISS Vector Stores

#### Helper Module

In this step, we add the functionality to query the FAISS vector store and generate responses using OpenAI.

1. **Update `langchain_helper.py`:**
    - Add the `get_response_from_query` function to query the FAISS vector store and generate responses using OpenAI.
    - Include a test case for processing a YouTube video and querying its content.

   Updated `langchain_helper.py`:
   ```python
   from langchain.llms import OpenAI
   from dotenv import load_dotenv
   from langchain import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.document_loaders import YoutubeLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import FAISS
   from langchain_community.embeddings import OpenAIEmbeddings
   import os

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()

   def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
       # Create a loader object to fetch the YouTube video transcript.
       loader = YoutubeLoader.from_youtube_url(video_url)

       # Load the transcript of the YouTube video.
       transcript = loader.load()

       # Create a text splitter to divide the transcript into smaller chunks.
       # chunk_size is the maximum size of each chunk, and chunk_overlap is the overlap between chunks.
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

       # Split the transcript into smaller, manageable chunks.
       docs = text_splitter.split_documents(transcript)

       # Create a FAISS vector store from the document chunks, using the specified embeddings.
       db = FAISS.from_documents(docs, embeddings)

       # Return the FAISS vector store containing the embedded document chunks.
       return db

   def get_response_from_query(db, query, k=4):
       # Search the query-relevant documents
       docs = db.similarity_search(query, k=k)

       # Combine the page content of the retrieved documents into a single string
       docs_page_content = " ".join([d.page_content for d in docs])

       # Initialize the OpenAI language model with specific parameters
       llm_OpenAI = OpenAI(temperature=0.8)

       # Define the prompt template for the language model
       prompt_template = PromptTemplate(
           input_variables=['question', 'docs'],
           template="""
           You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

           Answer the following question: {question}
           By searching the following video transcript: {docs}

           Only use the factual information from the transcript to answer the question.

           If you feel like you don't have enough information to answer the question, say "I don't know".

           Your answers should be detailed.
           """
       )

       # Create a language model chain with the defined prompt template
       chain = LLMChain(llm=llm_OpenAI, prompt=prompt_template)

       # Generate the response using the language model chain
       response = chain.run(question=query, docs=docs_page_content)
       response = response.replace("\n", "")

       return response

   video_url = "https://www.youtube.com/watch?v=POkPq1XLr4I&t=15s"
   db = create_vector_db_from_youtube_url(video_url)
   query = "How will the verdict affect Trump in this election"
   print(get_response_from_query(db, query))
   ```

#### Key Concepts

##### 1. Querying Vector Stores

- **Definition:** Querying vector stores involves searching for documents that are most similar to a given query.
- **Usage:** The `similarity_search` method of the `FAISS` class is used to find documents that match the query.
- **Example:**
  ```python
  docs = db.similarity_search(query, k=k)
  ```

##### 2. Combining Document Content

- **Definition:** Combining document content involves merging the text content of multiple documents into a single
  string for further processing.
- **Usage:** The page content of the retrieved documents is concatenated into a single string.
- **Example:**
  ```python
  docs_page_content = " ".join([d.page_content for d in docs])
  ```

##### 3. Prompt Templates for Query Responses

- **Definition:** A prompt template defines the structure and content of the input provided to the language model to
  generate a response.
- **Usage:** The `PromptTemplate` class from LangChain creates templates for generating responses based on the query and
  document content.
- **Example:**
  ```python
  prompt_template = PromptTemplate(
      input_variables=['question', 'docs'],
      template="""
      You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

      Answer the following question: {question}
      By searching the following video transcript: {docs}

      Only use the factual information from the transcript to answer the question.

      If you feel like you don't have enough information to answer the question, say "I don't know".

      Your answers should be detailed.
      """
  )
  ```

##### 4. Generating Responses with LLM Chains

- **Definition:** An LLM chain links a language model with a prompt template to generate responses.
- **Usage:** The `LLMChain` class from LangChain creates a chain to generate a response based on the query and document
  content.
- **Example:**
  ```python
  chain = LLMChain(llm=llm_OpenAI, prompt=prompt_template)
  response = chain.run(question=query, docs=docs_page_content)
  ```

### Step 6: Add YouTube Assistant Interface and Query Processing

#### Helper Module

In this step, we integrate the `get_response_from_query` function and remove the test case for cleaner code.

1. **Update `langchain_helper.py`:**
    - Ensure `get_response_from_query` function is included and remove the test case.

   Updated `langchain_helper.py`:
   ```python
   from langchain.llms import OpenAI
   from dotenv import load_dotenv
   from langchain import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.document_loaders import YoutubeLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import FAISS
   from langchain_community.embeddings import OpenAIEmbeddings
   import os

   # Loading environment variables from a .env file.
   load_dotenv()

   # Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
   embeddings = OpenAIEmbeddings()

   def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
       # Create a loader object to fetch the YouTube video transcript.
       loader = YoutubeLoader.from_youtube_url(video_url)

       # Load the transcript of the YouTube video.
       transcript = loader.load()

       # Create a text splitter to divide the transcript into smaller chunks.
       # chunk_size is the maximum size of each chunk, and chunk_overlap is the overlap between chunks.
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

       # Split the transcript into smaller, manageable chunks.
       docs = text_splitter.split_documents(transcript)

       # Create a FAISS vector store from the document chunks, using the specified embeddings.
       db = FAISS.from_documents(docs, embeddings)

       # Return the FAISS vector store containing the embedded document chunks.
       return db

   def get_response_from_query(db, query, k=4):
       # Search the query-relevant documents
       docs = db.similarity_search(query, k=k)

       # Combine the page content of the retrieved documents into a single string
       docs_page_content = " ".join([d.page_content for d in docs])

       # Initialize the OpenAI language model with specific parameters
       llm_OpenAI = OpenAI(temperature=0.8)

       # Define the prompt template for the language model
       prompt_template = PromptTemplate(
           input_variables=['question', 'docs'],
           template="""
           You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

           Answer the following question: {question}
           By searching the following video transcript: {docs}

           Only use the factual information from the transcript to answer the question.

           If you feel like you don't have enough information to answer the question, say "I don't know".

           Your answers should be detailed.
           """
       )

       # Create a language model chain with the defined prompt template
       chain = LLMChain(llm=llm_OpenAI, prompt=prompt_template)

       # Generate the response using the language model chain
       response = chain.run(question=query, docs=docs_page_content)
       response = response.replace("\n", "")

       return response
   ```

#### Main Python Script

We create the main Python script with a Streamlit interface for inputting YouTube video URLs and queries, integrating
the functionality to process YouTube URLs and display responses based on video transcripts.

1. **Create `main.py`:**
    - Set up the Streamlit interface for the YouTube Assistant.

   New `main.py`:
   ```python
   import langchain_helper as lch
   import streamlit as st
   import textwrap

   st.title("YouTube Assistant")

   with st.sidebar:
       with st.form(key='my_form'):
           youtube_url = st.sidebar.text_area(
               label="Enter the YouTube video URL",
               max_chars=50
           )
           query = st.sidebar.text_area(
               label="Ask me about the video?",
               max_chars=50,
               key="query"
           )
           submit_button = st.form_submit_button(label='Submit')

   if query and youtube_url:
       db = lch.create_vector_db_from_youtube_url(youtube_url)
       response = lch.get_response_from_query(db, query)
       st.subheader("Response:")
       st.text(textwrap.fill(response, width=100))
   ```

   Boot the streamlit
    ```bash
    streamlit run main.py    
    ```
    <img src="https://i.imghippo.com/files/Vqw3t1717318691.png" alt="" border="0">

#### Key Concepts

##### 1. Streamlit Interface

- **Definition:** Streamlit is an open-source app framework for Machine Learning and Data Science projects. It allows
  you to create and share custom web apps for your machine learning projects.
- **Usage:** We use Streamlit to create an interactive interface for inputting YouTube video URLs and queries.
- **Example:**
  ```python
  import streamlit as st

  st.title("YouTube Assistant")
  ```

##### 2. Form Handling in Streamlit

- **Definition:** Forms in Streamlit allow for organized user input handling and submission.
- **Usage:** We use forms to take YouTube URLs and queries from the user.
- **Example:**
  ```python
  with st.sidebar:
      with st.form(key='my_form'):
          youtube_url = st.text_area(label="Enter the YouTube video URL", max_chars=50)
          query = st.text_area(label="Ask me about the video?", max_chars=50, key="query")
          submit_button = st.form_submit_button(label='Submit')
  ```

##### 3. Integrating LangChain Functions

- **Definition:** Integrating LangChain functions allows for processing YouTube URLs and querying their content.
- **Usage:** We call the `create_vector_db_from_youtube_url` and `get_response_from_query` functions to handle video
  processing and querying.
- **Example:**
  ```python
  db = lch.create_vector_db_from_youtube_url(youtube_url)
  response = lch.get_response_from_query(db, query)
  st.subheader("Response:")
  st.text(textwrap.fill(response, width=100))
  ```
