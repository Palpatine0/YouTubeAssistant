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


def get_response_from_query(db, query, k = 4):
    # Search the query-relevant documents
    docs = db.similarity_search(query, k = k)

    # Combine the page content of the retrieved documents into a single string
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize the OpenAI language model with specific parameters
    llm_OpenAI = OpenAI(temperature = 0.8)

    # Define the prompt template for the language model
    prompt_template = PromptTemplate(
        input_variables = ['question', 'docs'],
        template = """
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be detailed.
        """
    )

    # Create a language model chain with the defined prompt template
    chain = LLMChain(llm = llm_OpenAI, prompt = prompt_template)

    # Generate the response using the language model chain
    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace("\n", "")

    return response
