import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from app.enum.llm import Llm


def get_pdf_text(pdf_docs: str):
    text = " "
    # Iterate through each PDF document path in the list
    for pdf in pdf_docs:
        # Create a PdfReader object for the current PDF document
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF document
        for page in pdf_reader.pages:
            # Extract text from the current page and append it to the 'text' string
            text += page.extract_text()
    # Return the concatenated text from all PDF documents
    return text


# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, llm: str):
    # Create embeddings using a Google Generative AI model
    if llm == Llm.GEMINI.value:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif llm == Llm.OPENAI.value:
        embeddings = OpenAIEmbeddings()
    else:
        pass
    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")


def get_gemini_model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    return model

def get_openai_model():
    model = ChatOpenAI(temperature=0, streaming=True)
    return model

def get_conversational_chain(llm: str):
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer
    Also give file name from which the content is extracted and the page number also. This is important if user didn't ask about it also\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if llm == Llm.GEMINI.value:
        # Initialize a ChatGoogleGenerativeAI model for conversational AI
        model = get_gemini_model()
    elif llm == Llm.OPENAI.value:
        model = get_openai_model()
    else:
        pass

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, llm: str):
    if llm == Llm.GEMINI.value:
        # Create embeddings for the user question using a Google Generative AI model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        chain = get_conversational_chain(llm=Llm.GEMINI.value)
    elif llm == Llm.OPENAI.value:
        # Load a FAISS vector database from a local file
        embeddings = OpenAIEmbeddings()
        chain = get_conversational_chain(llm=Llm.OPENAI.value)
    else:
        pass

    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings)

    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(user_question)

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    st.write("Reply: ", response["output_text"])
    return response
