**Ask Rachel - A Personal AI Knowledge Hub**

"Ask Rachel" is an AI assistant you can chat with. Rachel's knowledge comes from a curated set of documents, allowing you to ask detailed questions and get conversational, accurate answers. It's like having a personal expert on a specific topic, ready to help you find the information you need.

This project is built with privacy and accessibility in mind, using free, open-source, and locally-runnable tools.

**How to Use**
Navigate to the chat interface.

Type your question in the text box at the bottom of the screen.

Press Send and get your answer!

**Features**
Public Query Interface: A user-friendly chat interface for anyone to ask questions.

AI-Powered Search: Uses modern vector embeddings to understand the semantic meaning of your questions, finding the most relevant information even if the keywords don't match exactly.

Conversational Answers: Leverages a Generative LLM to synthesize the retrieved information into a natural, easy-to-understand answer.

Personal Knowledge Curation: The knowledge base is built from a specific set of documents, making it an expert in its domain.

For Developers: Running This Project Locally
The following sections provide details on the project's architecture and how to set it up on your own machine.

**Architecture Overview**
The application is split into two primary workflows:

1. Data Ingestion Flow
This is the process for adding new information to the knowledge base:

Upload: Documents are uploaded through a private interface.

Processing: The backend server parses the document, splits it into manageable text chunks, and generates vector embeddings for each chunk using an AI model.

Storage: The vector embeddings are stored in a database, ready for retrieval.

2. Query & Answering Flow
This is the process for asking and answering questions:

User Question: A user asks a question in the public chat interface.

Query Embedding: The question is converted into a vector embedding.

Similarity Search: The system searches the Vector Database to find the text chunks most similar to the question. This is the "context."

Answer Generation: The context and the original question are sent to a Generative LLM, which synthesizes the information into a final, conversational answer.

Display: The answer is sent back to the user and displayed in the chat interface.

Tech Stack
Backend: Python with Flask

AI & Orchestration: LangChain

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 via HuggingFace

Vector Database: ChromaDB (local, persistent storage)

PDF/Text Processing: PyPDF2

Frontend: HTML, CSS, and vanilla JavaScript

Setup and Installation
Follow these steps to get the project running on your local machine.

Prerequisites:

Python 3.8+

pip (Python package installer)

1. Clone the repository:

git clone [(https://github.com/rachel-prijatna/personal-knowledge-hub.git)]

2. Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate

3. Install dependencies:
A requirements.txt file should be created containing the following:

Flask
langchain
chromadb
pypdf2
sentence-transformers
.
.
.

Install them with:

pip install -r requirements.txt

How to Run
1. Run the Application:
Execute the following command in your terminal from the project's root directory:

python app.py

The application will be available at http://127.0.0.1:5000.

2. Add Knowledge:
The application includes a simple interface for uploading your knowledge documents (.txt, .pdf). Once the server is running, you can add your files to build the knowledge base.

3. Ask Questions:
Navigate to the main interface at http://127.0.0.1:5000/ and use the chat box to ask questions related to the documents you've uploaded.
