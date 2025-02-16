# Agentic RAG Chat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A fully custom chatbot built with Agentic RAG (Retrieval-Augmented Generation), combining Gemini models (free tier) with a local knowledge base for accurate, context-aware, and explainable responses. Features a lightweight, dependency-free frontend and a streamlined FastAPI backend for complete control and simplicity.

![Demo](demo.gif)


## Features

- Pure HTML/CSS/JavaScript frontend with no external dependencies
- FastAPI backend with Gemini integration
- Agentic RAG implementation with:
  - Context retrieval using embeddings and cosine similarity
  - Step-by-step reasoning with Chain of Thought
  - Function calling for dynamic context retrieval
- Comprehensive error handling and logging
- Type-safe implementation with Python type hints
- Configurable through environment variables

## Project Structure

```
agentic_rag/
├── backend/
│   ├── embeddings.py    # Embedding and similarity functions
│   ├── rag_engine.py    # Core RAG implementation
│   └── server.py        # FastAPI server
├── frontend/
│   └── index.html       # Web interface
├── requirements.txt     # Python dependencies
├── .env.sample         # Sample environment variables
└── README.md           # Documentation
```

## Prerequisites

- Python 3.11 or higher
- Gemini API key (you can get one for free at google ai studio)
- Git (for version control)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AndrewNgo-ini/agentic_rag.git
cd agentic_rag
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.sample .env
```
Then edit `.env` with your configuration:
```
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.0-flash  # or another compatible model
GEMINI_EMBEDDING_MODEL=text-embedding-3-small # or another compatible model
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python server.py
```

2. Access the frontend:
- Option 1: Open `frontend/index.html` directly in your web browser
- Option 2: Serve using Python's built-in server:
```bash
cd frontend
python -m http.server 3000
```

Then visit http://localhost:3000 in your browser.

## Usage

1. Type your question in the input field
2. The system will:
   - Retrieve relevant context using embeddings
   - Generate a step-by-step reasoning process
   - Provide a final answer based on the context and reasoning
3. View the intermediate steps and reasoning process in the response

## Configuration

The application can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| GEMINI_API_KEY | Your Gemini API key | Required |
| GEMINI_MODEL | Gemini model to use | gemini-2.0-flash |
| HOST | Backend server host | 0.0.0.0 |
| PORT | Backend server port | 8000 |

## Error Handling

The application includes comprehensive error handling:

- API errors are logged and returned with appropriate status codes
- Frontend displays user-friendly error messages
- Detailed logging for debugging and monitoring
- Graceful fallbacks for common failure scenarios


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Troubleshooting

Common issues and solutions:

1. **Gemini API Error**
   - Verify your API key is correct
   - Check your API usage limits
   - Ensure the model name is valid

2. **Backend Connection Failed**
   - Confirm the backend server is running
   - Check the port is not in use
   - Verify firewall settings

3. **Embedding Errors**
   - Ensure input text is not empty
   - Check for proper text encoding
   - Verify numpy installation

## Acknowledgments

- Gemini for their API and models
- FastAPI framework
- Contributors and maintainers
