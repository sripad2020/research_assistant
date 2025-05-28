# 🔍 Research Agent  

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Flask](https://img.shields.io/badge/flask-2.0%2B-red)](https://flask.palletsprojects.com/)

**An AI-powered research assistant that streamlines literature reviews and academic research**

![Research Agent Demo](https://via.placeholder.com/800x400?text=Research+Agent+Interface+Demo)

## 🌟 Features

### 🚀 Core Capabilities
- **Multi-source search** (arXiv, OpenAlex, Google Scholar)
- **Smart paper summarization** with key insights extraction
- **Citation network visualization** for understanding paper relationships
- **Research gap identification** to find unexplored areas
- **Automated literature review** generation

### 🔍 Advanced Features
- Trend analysis with publication statistics
- Author impact and collaboration networks
- Comparative paper analysis
- Research timeline visualization
- Methodology extraction from papers
- Interactive knowledge graphs

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- (Optional) SerpAPI key for Google Scholar integration

### Setup Instructions

1. Clone the repository:
```bash
https://github.com/sripad2020/research_assistant.git
cd research_assistant

Create and activate virtual environment:

bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate

bash
pip install -r requirements.txt
Download NLTK data:

bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
Create .env file:

ini
GEMINI_API_KEY=your_google_api_key
SERPAPI_API_KEY=your_serpapi_key  # optional
FLASK_SECRET_KEY=your_secret_key
🖥️ Usage
Start the application:

bash
python app.py
Access the web interface at: http://localhost:5000

Basic Commands
/search [query] - Search academic papers

/summarize [paper_id] - Generate paper summary

/visualize citations [paper_id] - Show citation network

/review [topic] - Generate literature review

/gap [field] - Identify research gaps

📚 Tech Stack
Core Dependencies
Category	Libraries
Web Framework	Flask, gunicorn
NLP Processing	NLTK, spaCy, scikit-learn
PDF Analysis	PyPDF2, pdfminer
Visualization	Plotly, NetworkX, Matplotlib
AI Integration	google-generativeai
API Integrations
Google Gemini AI

arXiv API

OpenAlex

SerpAPI (for Google Scholar)

📂 Project Structure
research-agent/
├── app.py                # Main application
├── config.py             # Configuration settings
├── requirements.txt      # Dependencies
├── static/               # CSS/JS assets
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── dashboard.html    # Main interface
│   └── results/         # Result templates
├── modules/              # Functional modules
│   ├── search.py         # Paper search
│   ├── analysis.py       # NLP processing
│   └── visualization.py  # Graph generation
└── uploads/              # User document storage