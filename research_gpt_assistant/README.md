# 🧠 ResearchGPT Assistant

ResearchGPT Assistant is an **AI-powered research assistant** that helps you analyze PDF documents, extract knowledge, and answer questions using the **Mistral API** and advanced prompt engineering.  
It includes document processing (PDF extraction, cleaning, chunking, TF-IDF similarity search) and AI agents (Summarizer, QA, Research Workflow).

---

## ⚙️ Features

- 📄 Extract and clean text from PDFs
- ✂️ Preprocess and chunk text for analysis
- 🔍 Perform similarity search with TF-IDF
- 🤖 Use Mistral API for AI-powered Q&A
- 🪜 Chain-of-Thought & Self-Consistency reasoning
- 🔄 ReAct workflow with thought-action-observation cycles
- 👥 Multi-agent system (Summarizer, QA, Workflow agents)
- 📊 Example usage scenarios and performance metrics

---

## 🚀 Installation Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/research_gpt_assistant.git
   cd research_gpt_assistant

## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

## Install dependencies
pip install -r requirements.txt

## System Architecture

research_gpt_assistant/
│
├── config.py               # Configuration and API key handling
├── document_processor.py   # PDF text extraction, preprocessing, chunking
├── research_assistant.py   # AI integration + advanced prompting
├── research_agents.py      # Multi-agent system (Summarizer, QA, Workflow)
├── main.py                 # Entry point to run the assistant
│
├── data/                   # Sample PDFs
├── results/                # Outputs, summaries, analyses
└── prompts/                # Prompt templates
