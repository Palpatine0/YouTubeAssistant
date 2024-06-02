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


# Loading environment variables from a .env file.
load_dotenv()

# Creating an instance of the OpenAIEmbeddings class to generate embeddings for text data.
embeddings = OpenAIEmbeddings()