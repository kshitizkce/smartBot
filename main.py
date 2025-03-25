import streamlit as st
from app.enum.llm import Llm
from app.core.config import load_config
from app.llm import get_pdf_text, get_chunks, get_vector_store, user_input


def main():
    load_config()
    st.set_page_config(page_title="Smart Bot", layout="wide")

    # Initialize session state for storing messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    llm_option = st.radio(
        "Choose Model", 
        [Llm.GEMINI.value, Llm.OPENAI.value],
        help="Select the language model to use for answering your question.",
        horizontal=True
    )

    # Main header
    st.markdown("""
        <div style="background-color:#2C3E50;padding:20px;border-radius:10px;text-align:center;">
        <h1 style="color:white;">📄 Smart Bot</h1>
        <h3 style="color:white;">Interact with your PDF files in a smart way!</h3>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar menu for file upload and processing
    st.sidebar.markdown("""
        <div style="background-color:#f1f1f1;padding:20px;border-radius:10px;text-align:center;">
        <h2>📂 Upload and Process Files</h2>
        <p>Select multiple PDF files to upload and process.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "", 
            accept_multiple_files=True, 
            help="Select multiple PDF files to upload."
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(
                    text_chunks,
                    llm=Llm.GEMINI.value if llm_option == Llm.GEMINI.value else Llm.OPENAI.value
                )
                st.success("Files have been processed successfully!")

    # Main area for user interaction
    st.markdown("""
        <div style="background-color:#F8F9F9;padding:20px;border-radius:10px;margin-top:20px;">
        <h2>💬 Ask a Question</h2>
        </div>
        """, unsafe_allow_html=True)
    user_question = st.text_input(
        "Type your question here and press Enter", 
        help="Ask any question related to the uploaded PDF files."
    )

    # Handle user question input
    if user_question:
        response = user_input(
            user_question,
            llm=Llm.GEMINI.value if llm_option == Llm.GEMINI.value else Llm.OPENAI.value
        )
        print(response)
        
        if response:
            # Save the user question and response to session state
            st.session_state.messages.append((user_question, response["output_text"]))
            if len(st.session_state.messages) > 5:
                st.session_state.messages.pop(0)
        else:
            st.error("Failed to get a response from the model.")

    # Display the last five messages in a chatbox style
    st.markdown("""
        <div style="background-color:#F8F9F9;padding:20px;border-radius:10px;margin-top:20px;">
        <h2>📜 Conversation History (last 5 messages)</h2>
        </div>
        """, unsafe_allow_html=True)
    chatbox_style = """
        <style>
        .chatbox {
            background: #e6e6e6;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            max-height: 300px;
            overflow-y: auto;
        }
        .user-message {
            background: #0084ff;
            color: white;
            padding: 8px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
            float: right;
            clear: both;
            max-width: 70%;
        }
        .bot-message {
            background: #f1f0f0;
            padding: 8px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: left;
            float: left;
            clear: both;
            max-width: 70%;
        }
        </style>
    """
    st.markdown(chatbox_style, unsafe_allow_html=True)

    if st.session_state.messages:
        chat_html = '<div class="chatbox">'
        for question, answer in reversed(st.session_state.messages):
            chat_html += f'<div class="user-message">{question}</div>'
            chat_html += f'<div class="bot-message">{answer}</div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # Footer with additional information
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2C3E50;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        <div class="footer">
            © 2024 Smart Doc Bot. All rights reserved.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
