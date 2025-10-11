# research_agents.py
import json
from typing import List, Dict, Any
# Import the assistant class from research_assistant.py
from research_assistant import ResearchGPTAssistant


# Base Agent

class BaseAgent:
    def __init__(self, research_assistant: ResearchGPTAssistant):
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"

    def execute_task(self, task_input: dict) -> dict:
        """Each agent overrides this."""
        raise NotImplementedError("Subclasses must implement execute_task()")


# Summarizer Agent

class SummarizerAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"

    def summarize_document(self, doc_id: str) -> dict:
        """Summarize one document by ID."""
        context_chunks = self.assistant.doc_processor.get_document_chunks(doc_id)
        if not context_chunks:
            return {"doc_id": doc_id, "error": "Document not found or empty"}

        context = "\n".join(context_chunks[:5])
        prompt = f"""
Summarize the following document in JSON format with:
- key_points
- methodology
- findings
- limitations
- potential_future_work

Document content:
{context}
"""
        response = self.assistant._call_mistral(prompt, temperature=0.3)
        try:
            return json.loads(response)
        except Exception:
            return {"doc_id": doc_id, "summary_text": response}

    def summarize_literature(self, doc_ids: List[str]) -> dict:
        """Summarize multiple documents and provide synthesis."""
        summaries = [self.summarize_document(doc_id) for doc_id in doc_ids]

        combined_context = "\n".join(
            str(s) for s in summaries if isinstance(s, dict)
        )
        synthesis_prompt = f"""
Based on the following document summaries, synthesize an overview:

{combined_context}

Return JSON with:
- major_themes
- points_of_agreement
- points_of_conflict
- research_gaps
"""
        synthesis = self.assistant._call_mistral(synthesis_prompt, temperature=0.5)
        try:
            synthesis_json = json.loads(synthesis)
        except Exception:
            synthesis_json = {"synthesis_text": synthesis}

        return {"individual_summaries": summaries, "synthesis": synthesis_json}

    def execute_task(self, task_input: dict) -> dict:
        print(f"[{self.agent_name}] Executing summarization task...")
        if "doc_ids" in task_input:
            return self.summarize_literature(task_input["doc_ids"])
        elif "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        else:
            return {"error": "No document IDs provided"}


# QA Agent
class QAAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"

    def answer_factual_question(self, question: str) -> dict:
        """Answer factual questions by retrieving chunks directly."""
        relevant_chunks = self.assistant.doc_processor.find_similar_chunks(
            question, top_k=5
        )
        context = "\n".join(relevant_chunks)
        prompt = f"""
Question: {question}
Context: {context}

Answer directly with only facts from context.
If answer is not in context, say "Not found in provided documents".
"""
        response = self.assistant._call_mistral(prompt, temperature=0)
        return {"question": question, "answer": response, "reasoning_type": "factual"}

    def answer_analytical_question(self, question: str) -> dict:
        """Answer analytical questions using chain-of-thought + self-consistency."""
        relevant = self.assistant.doc_processor.find_similar_chunks(question, top_k=6)

        responses = []
        for _ in range(3):  # 3 attempts
            resp = self.assistant.chain_of_thought_reasoning(
                question, relevant
            )
            responses.append(resp)

        # Select majority or fallback
        try:
            best_answer = max(set(responses), key=responses.count)
        except Exception:
            best_answer = responses[0]

        return {
            "question": question,
            "analysis": best_answer,
            "reasoning_type": "chain_of_thought + self_consistency",
            "attempts": responses
        }

    def execute_task(self, task_input: dict) -> dict:
        print(f"[{self.agent_name}] Executing QA task...")
        question = task_input.get("question", "")
        q_type = task_input.get("type", "factual")
        if q_type == "factual":
            return self.answer_factual_question(question)
        else:
            return self.answer_analytical_question(question)


# Verification Agent
class VerificationAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "VerificationAgent"

    def verify_answer(self, answer: str, question: str, context: str) -> dict:
        """Check the quality of an answer against context."""
        prompt = f"""
Check this answer against the provided context.

Question: {question}
Answer: {answer}
Context: {context}

Return JSON with:
- validity (true/false)
- issues (list of problems)
- improved_answer (corrected version)
- confidence (0-1)
"""
        raw = self.assistant._call_mistral(prompt, temperature=0)
        try:
            return json.loads(raw)
        except Exception:
            return {"verification_text": raw}

    def execute_task(self, task_input: dict) -> dict:
        print(f"[{self.agent_name}] Verifying answer...")
        return self.verify_answer(
            task_input.get("answer", ""),
            task_input.get("question", ""),
            task_input.get("context", "")
        )


# Research Workflow Agent
class ResearchWorkflowAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"

    def conduct_research_session(self, research_goal: str, doc_ids: List[str]) -> dict:
        """Full research workflow across multiple steps."""
        # Step 1: Summarize literature
        summarizer = SummarizerAgent(self.assistant)
        summaries = summarizer.summarize_literature(doc_ids)

        # Step 2: Identify key questions
        prompt = f"""
Research goal: {research_goal}
Literature synthesis: {summaries.get('synthesis', '')}

Suggest 3-5 critical research questions to investigate.
Return JSON with "questions".
"""
        question_resp = self.assistant._call_mistral(prompt, temperature=0.7)
        try:
            questions = json.loads(question_resp).get("questions", [])
        except Exception:
            questions = [question_resp]

        # Step 3: Answer questions
        qa_agent = QAAgent(self.assistant)
        answers = [qa_agent.answer_analytical_question(q) for q in questions]

        # Step 4: Identify gaps
        gap_prompt = f"""
Research goal: {research_goal}
Answers: {answers}

Identify remaining gaps or open research directions.
Return JSON with "gaps".
"""
        gap_resp = self.assistant._call_mistral(gap_prompt, temperature=0.6)
        try:
            gaps = json.loads(gap_resp).get("gaps", [])
        except Exception:
            gaps = [gap_resp]

        return {
            "goal": research_goal,
            "literature_summaries": summaries,
            "questions": questions,
            "answers": answers,
            "gaps": gaps,
        }

    def execute_task(self, task_input: dict) -> dict:
        print(f"[{self.agent_name}] Executing workflow task...")
        return self.conduct_research_session(
            task_input.get("goal", ""),
            task_input.get("doc_ids", []),
        )


# Agent Orchestrator
class AgentOrchestrator:
    def __init__(self, research_assistant: ResearchGPTAssistant):
        self.agents = {
            "summarizer": SummarizerAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "verify": VerificationAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant),
        }

    def route_task(self, agent_type: str, task_input: dict) -> dict:
        print(f"[Orchestrator] Routing to {agent_type}...")
        if agent_type not in self.agents:
            return {"error": f"Unknown agent type: {agent_type}"}
        return self.agents[agent_type].execute_task(task_input)

    def execute_complex_workflow(self, workflow_description: str) -> dict:
        """Very simple workflow parser for demo purposes."""
        if "summarize" in workflow_description.lower():
            return self.route_task("summarizer", {"doc_ids": ["doc1", "doc2"]})
        if "qa" in workflow_description.lower():
            return self.route_task("qa", {"question": "What are the key findings?"})
        return {
            "workflow_description": workflow_description,
            "note": "Basic parser only."
        }
