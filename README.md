# 🧠 ResearchGPT Assistant

ResearchGPT Assistant is a Python-based AI tool designed to help researchers process, analyze, and summarize PDF research papers efficiently. It leverages advanced reasoning techniques such as Chain-of-Thought (CoT), Self-Consistency, ReAct workflow, and QA verification to provide accurate, step-by-step research insights.


## 📂 Project Structure
research_gpt_assistant/
│
├── config.py                 # Configuration for API keys, file paths, and parameters
├── main.py                   # Main script to run demos and workflows
├── research_assistant.py     # Core AI reasoning and research functions
├── data/
│   ├── sample_papers/        # PDF research papers for testing
│   └── processed/            # Extracted text from PDFs
├── results/                  # Output results: JSON, text summaries
└── prompts/                  # Optional prompt templates for AI tasks

## ⚙️ Features

- Document Processing: Read and extract text from PDF research papers.

- Search Index: Build a searchable index of document chunks for similarity queries.

- Chain-of-Thought (CoT) Reasoning: Step-by-step reasoning for complex research questions.

- Self-Consistency: Generate multiple answers and select the most consistent one.

- ReAct Workflow: Execute a research workflow with actions like SEARCH, SUMMARIZE, and STOP.

- Verification: Verify and improve answers against provided context for accuracy.

## 🚀 Installation Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/JamilaAr/ResearchGPT_Assistant
   cd research_gpt_assistant

## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

## Install dependencies
pip install -r requirements.txt

## Add your .env file in research_gpt_assistant/ with your Mistral API key:
MISTRAL_API_KEY=your_api_key_here
Make sure .env is listed in .gitignore to keep it private.

## 🚀 How to Run:
python main.py
