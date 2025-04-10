import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os


PAGE_TITLE = "📚 NK's RAG System"
MAIN_TITLE = "Explore PDF's From Srihari's RAG 🛜...."
UPLOAD_PROMPT = "Upload your PDF file"
QUESTION_PROMPT = "Hai buddyy..Ask me a Question..!"
PROCESSING_MESSAGE = "Analysing PDF..."
SUCCESS_MESSAGE = "PDF analysed successfully!"
GENERATING_MESSAGE = "Hold on Buddddyyy...!!..Your answer is generating...!"
ANSWER_HEADER = "Answer:"
SOURCES_HEADER = "Sources:"
NO_FILE_MESSAGE = "Please upload a PDF file to begin."
MODEL_NAME = "mistral"
PERSIST_DIR = "vectorstore_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEARCH_KWARGS = {"k": 3}
SOURCE_DOC_LENGTH = 200
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Rubik', sans-serif;
        color: #f5f5f5;
    }

    h1 {
    overflow: hidden;
    white-space: nowrap;
    margin: 0 auto;
    letter-spacing: .1em;
    animation: typing 3.5s steps(40, end);
    color: #00ffe7;
    font-size: 2.5rem;
    text-align: center;
    width: max-content;
    padding-bottom: 1rem;
}

@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: orange }
    }

  .stFileUploader {
    background-color: #1e1e1e;
    padding: 1.2rem;
    border-radius: 14px;
    border: 2px dashed #3e4e57;
    width: 60%;
    margin: auto;
    animation: zoomInOut 2s infinite alternate;
    transition: all 0.4s ease-in-out;
}

/* Hover zoom + soft glow */
.stFileUploader:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(0, 255, 231, 0.15);
    border-color: #5c6bc0;
    background: #222;
}

.stFileUploader label {
    font-weight: 500;
    color: #cfd8dc;
    font-size: 1rem;
    transition: color 0.3s ease;
}

.stFileUploader:hover label {
    color: #ffffff;
}


.stFileUploader label {
    font-weight: bold;
    color: #00ffe7;
    font-size: 1rem;
    transition: color 0.4s ease-in-out;
}

.stFileUploader:hover label {
    color: #7c4dff;
}


 @keyframes zoomInOut {
        0% { transform: scale(0.98); }
        100% { transform: scale(1.02); }
    }


    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #00ffe7;
        padding: 10px;
        background-color: #1a1a1a;
        color: white;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:hover {
        box-shadow: 0 0 10px #00ffe7;
        border-color: #1de9b6;
    }
    @keyframes pulseGlow {
    0% {
        box-shadow: 0 0 10px #1de9b6;
    }
    100% {
        box-shadow: 0 0 20px #64ffda;
    }
}

.stFileUploader:hover {
    animation: pulseGlow 1.5s ease-in-out infinite alternate;
}


    .stFileUploader button {
background: linear-gradient(to right, #116466, #2c7873);

    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.4s ease-in-out;
}

/* Hover effect for Browse Files button */
.stFileUploader button:hover {
    transform: scale(1.08);
    background: linear-gradient(to right, #3aafa9, #2b7a78);
box-shadow: 0 0 10px rgba(58, 175, 169, 0.5);

    color: #ffffff;
}

    .stSpinner {
        color: #00e676;
        font-weight: bold;
    }

    .stSubheader {
        color: #81d4fa;
        margin-top: 2rem;
    }

    .stMarkdown p {
        font-size: 1.05rem;
        line-height: 1.6;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
    }

    .stInfo {
        background-color: #263238;
        border-left: 5px solid #00ffe7;
        padding: 12px;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    .stTextInput label {
    font-size: 1.3rem !important;
    font-weight: 600;
    color: #00ffe7;
}
    </style>
    """,
    unsafe_allow_html=True,
)




st.title(MAIN_TITLE) #initialize


ollama_model = OllamaLLM(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model=MODEL_NAME)


def process_pdf(uploaded_file):
    """Processes the uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="NK_rag",
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()

    os.unlink(tmp_path)
    return vectorstore


uploaded_file = st.file_uploader(UPLOAD_PROMPT, type=["pdf"])

if uploaded_file is not None:
    with st.spinner(PROCESSING_MESSAGE):
        vectorstore = process_pdf(uploaded_file)
        st.success(SUCCESS_MESSAGE)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=SEARCH_KWARGS),
        return_source_documents=True,
    )

    user_question = st.text_input(QUESTION_PROMPT)

    if user_question:
        with st.spinner(GENERATING_MESSAGE):
            response = qa_chain.invoke({"query": user_question})

            st.subheader(ANSWER_HEADER)
            st.write(response["result"])

            st.subheader(SOURCES_HEADER)
            for doc in response["source_documents"]:
                st.write("- " + doc.page_content[:SOURCE_DOC_LENGTH] + "...")
else:
    st.info(NO_FILE_MESSAGE)