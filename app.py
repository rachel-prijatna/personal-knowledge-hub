from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai


app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # needed for flashing messages


# Configuration for uploads and vector store
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
ALLOWED_EXTENSIONS = {"txt", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings and persistent Chroma store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')


def is_allowed_filename(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part in the request")
        return redirect(url_for("admin"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("admin"))

    if not is_allowed_filename(file.filename):
        flash("Invalid file type. Please upload a .txt or .pdf file.")
        return redirect(url_for("admin"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Ingestion: read file contents (.txt or .pdf)
    text_content = ""
    ext = filename.rsplit(".", 1)[1].lower()
    try:
        if ext == "txt":
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        elif ext == "pdf":
            reader = PdfReader(save_path)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            text_content = "\n".join(pages_text)
        else:
            flash("Unsupported file type.")
            return redirect(url_for("admin"))
    except Exception as exc:
        flash(f"Failed to read uploaded file: {exc}")
        return redirect(url_for("admin"))

    # Split text into chunks using LangChain's CharacterTextSplitter
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text_content)

    # Embed and store chunks into persistent Chroma vector DB
    try:
        vectorstore.add_texts(chunks)
        # Note: Chroma 0.4+ automatically persists, no need for manual persist()
    except Exception as exc:
        flash(f"Failed to store embeddings: {exc}")
        return redirect(url_for("admin"))

    flash(f"File '{filename}' processed and stored in the knowledge base ({len(chunks)} chunks).")
    return redirect(url_for("admin"))


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Reuse the same embeddings and persistent Chroma vector store
        # Perform similarity search for the top 3 chunks
        results = vectorstore.similarity_search(question, k=3)
        context_chunks = [doc.page_content for doc in results]
        context = "\n\n".join(context_chunks) if context_chunks else ""

        # Create prompt for Gemini API
        prompt = f"""You are an AI assistant for a personal knowledge hub belonging to a person named Rachel. Your persona is professional, friendly, and helpful. You are speaking to a general user, not to Rachel.

Your primary goal is to answer the user's question based on the provided "Context" from Rachel's documents.

1.  **If the context is relevant to the user's question:** Synthesize the information into a natural, conversational answer. Do NOT start your answer with phrases like "Based on the provided context...". Answer the question directly as if you already know the information about Rachel.
2.  **If the context is empty or irrelevant:** Do not mention the context. Instead, handle the user's query gracefully. For simple greetings (like "hi" or "hello"), respond with a friendly greeting to the user. For questions you cannot answer, politely state that you can't find information on that specific topic in Rachel's documents and gently guide the conversation back to her main areas (e.g., "I can't find specific information about that in Rachel's documents. Is there anything I can help you with regarding her portfolio, teaching, or hobbies?").
Context:
{context}

Question:
{question}"""

        # Generate response using Gemini API
        response = model.generate_content(prompt)
        answer = response.text

        return jsonify({
            "question": question,
            "answer": answer,
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to process question: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

