import streamlit as st
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document  
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found! Set up your API key in the .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Function to select the best available AI model
def select_model():
    models = [
        "models/gemini-2.0-pro-exp",
        "models/gemini-1.5-pro",
        "models/chat-bison-001",
        "models/gemini-1.5-pro-latest",
    ]
    try:
        available_models = genai.list_models()
        model_names = {model.name for model in available_models}
        return next((m for m in models if m in model_names), None)
    except Exception as e:
        st.error(f"Error selecting model: {e}")
        return None

# ✅ Extract text from PDF
def get_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ✅ Split text into chunks
def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)

# ✅ Create FAISS vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        store = FAISS.from_texts(text_chunks, embedding=embeddings)
        store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# ✅ Save text as PDF
def save_text_as_pdf(text, filename="ai_response.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    y_position = 750

    for line in text.split("\n"):
        c.drawString(100, y_position, line)
        y_position -= 20
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750

    c.save()
    return filename

# ✅ Add text to PDF
def add_text_to_pdf(uploaded_pdf, user_text, filename="updated_document.pdf"):
    try:
        reader, writer = PdfReader(uploaded_pdf), PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        
        temp_pdf = save_text_as_pdf(user_text, "temp_text_page.pdf")
        temp_reader = PdfReader(temp_pdf)
        writer.add_page(temp_reader.pages[0])
        
        with open(filename, "wb") as output_pdf:
            writer.write(output_pdf)
        
        os.remove(temp_pdf)
        return filename
    except Exception as e:
        st.error(f"Error adding text to PDF: {e}")
        return None

# ✅ Process question with AI
def process_question(user_text):
    try:
        model = select_model()
        if not model:
            return "No valid model selected."

        llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_text)

        if not docs:
            return "No relevant documents found."

        prompt = PromptTemplate(
            template="You are an AI assistant. Answer using the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"],
        )

        response = load_qa_chain(llm, chain_type="stuff", prompt=prompt)({
            "input_documents": [Document(page_content=doc.page_content) for doc in docs],
            "context": docs[0].page_content,
            "question": user_text
        })

        return response.get("output_text", "No answer found.")
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return "Failed to generate an answer."

# ✅ Merge multiple PDFs
def merge_pdfs(uploaded_pdfs, filename="merged_document.pdf"):
    try:
        merger = PdfMerger()
        for pdf in uploaded_pdfs:
            merger.append(pdf)
        merger.write(filename)
        merger.close()
        return filename
    except Exception as e:
        st.error(f"Error merging PDFs: {e}")
        return None

# ✅ Split PDF into pages
def split_pdf(uploaded_pdf):
    try:
        reader = PdfReader(uploaded_pdf)
        file_list = []
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            output_filename = f"split_page_{i+1}.pdf"
            with open(output_filename, "wb") as output_pdf:
                writer.write(output_pdf)
            file_list.append(output_filename)
        return file_list
    except Exception as e:
        st.error(f"Error splitting PDF: {e}")
        return []

# ✅ Streamlit App Interface
def main():
    st.set_page_config(page_title="AI PDF Chat & Editor")
    st.header("📄 AI PDF Chat & Editor 💁")

    mode = st.radio("Choose an option:", ("Ask a Question", "Improve Document", "Text to PDF", "Add Text to PDF", "Merge PDFs", "Split PDF"))
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    user_text = st.text_area("Enter text")

    if st.button("Submit"):
        if mode == "Ask a Question" and uploaded_pdfs:
            st.write(process_question(user_text))
        elif mode == "Improve Document" and uploaded_pdfs:
            st.write(get_pdf_text(uploaded_pdfs[0]))
        elif mode == "Text to PDF":
            st.download_button("📥 Download PDF", open(save_text_as_pdf(user_text), "rb"), "custom_text.pdf")
        elif mode == "Add Text to PDF" and uploaded_pdfs:
            st.download_button("📥 Download Updated PDF", open(add_text_to_pdf(uploaded_pdfs[0], user_text), "rb"), "updated_document.pdf")
        elif mode == "Merge PDFs" and uploaded_pdfs:
            st.download_button("📥 Download Merged PDF", open(merge_pdfs(uploaded_pdfs), "rb"), "merged_document.pdf")
        elif mode == "Split PDF" and uploaded_pdfs:
            for file in split_pdf(uploaded_pdfs[0]):
                st.download_button(f"📥 Download {file}", open(file, "rb"), file)

if __name__ == "__main__":
    main()
