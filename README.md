RAG-Based Document Q&A Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and Google Gemini API. It allows users to upload documents and ask context-aware questions based on the uploaded content.

Run Locally
-----------

1. Clone the repository and install dependencies:

   pip install -r requirements.txt

2. Create a `.env` file and add your Gemini API key:

   GEMINI_API_KEY=your_api_key_here

3. Start the app:

   streamlit run app.py

4. Open in your browser:

   http://localhost:8501/

Features
--------

- Upload `.txt`, `.pdf`, `.docx`, or `.xlsx` files
- Ask questions based on uploaded documents
- Uses FAISS for fast vector search
- Uses Google Gemini for natural language response generation

Created By
----------

Himanshu vijay
