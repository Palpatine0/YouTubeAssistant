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
# Module to generating text embeddings using OpenAI's models
from langchain_community.embeddings import OpenAIEmbeddings

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

    # Split the transcript into smaller, manageable chunks.
    docs = text_splitter.split_documents(transcript)

    # Create a FAISS vector store from the document chunks, using the specified embeddings.
    db = FAISS.from_documents(docs, embeddings)

    # Return the FAISS vector store containing the embedded document chunks.
    return db


video_url = "https://youtu.be/A9W6FAQPVuA?si=qshtmH3E_ah9QvbT"
print(create_vector_db_from_youtube_url(video_url))