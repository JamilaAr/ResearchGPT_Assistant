# main.py
"""
Complete demo script for ResearchGPT Assistant
Implements:
- System initialization
- Document processing & indexing
- Demonstrations: similarity search, CoT, Self-Consistency, ReAct
- Agent coordination demos (summarizer, QA, workflow, verification)
- Results saving and final demo report
"""

import os
import json
import traceback
from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator



def main():
    print("=== ResearchGPT Assistant - Full Demo ===\n")

    # Step 1: Initialize system
    print("1) Initializing system...")
    cfg = Config()
    dp = DocumentProcessor(cfg)
    assistant = ResearchGPTAssistant(cfg, dp)
    orchestrator = AgentOrchestrator(assistant)
    print("   System initialized.\n")

    # Step 2: Process sample PDFs
    print("2) Processing sample documents...")
    cfg.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_papers")
    if not os.path.exists(sample_dir):
        print(f"   Sample papers dir not found: {sample_dir}")
        print("   Create the folder and add PDFs, then rerun.")
        return

    pdf_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"   No PDFs found in {sample_dir}. Add PDFs and run again.")
        return

    processed_doc_ids = []
    for pdf in pdf_files:
        try:
            path = os.path.join(sample_dir, pdf)
            print(f"   Processing: {pdf}")
            doc_id = dp.process_document(path)
            processed_doc_ids.append(doc_id)
            print(f"     -> doc_id: {doc_id}")
        except Exception as e:
            print(f"     ! Failed processing {pdf}: {e}")
            traceback.print_exc()

    # Step 3: Build search index
    print("\n3) Building search index...")
    try:
        dp.build_search_index()
        print("   Index built.")
    except Exception as e:
        print("   ! Failed to build index:", e)
        traceback.print_exc()

    # Step 4: Document stats
    print("\n4) Document statistics:")
    stats = dp.get_document_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    _save_result("document_stats.json", stats, cfg)

    # Step 5: Basic similarity search demo
    print("\n5) Similarity search demo")
    q_sim = "machine learning algorithms"
    print(f"   Query: {q_sim}")
    try:
        sim = dp.find_similar_chunks(q_sim, top_k=3)
        print(f"   Found {len(sim)} chunks")
        _save_result("similarity_search.json", [{"doc_id": c[2], "score": c[1], "text": c[0][:400]} for c in sim], cfg)
    except Exception as e:
        print("   ! Similarity search failed:", e)
        traceback.print_exc()

    # Step 6: Chain-of-Thought reasoning demo
    print("\n6) Chain-of-Thought (CoT) demo")
    cot_q = "What are the main advantages and limitations of deep learning?"
    try:
        cot_resp = assistant.answer_research_question(cot_q, use_cot=True, use_verification=False)
        print("   CoT answer length:", len(cot_resp))
        _save_result("cot_response.json", cot_resp, cfg)
    except Exception as e:
        print("   ! CoT demo failed:", e)
        traceback.print_exc()

    # Step 7: Self-Consistency demo
    print("\n7) Self-Consistency demo")
    sc_q = "How do neural networks learn?"
    try:
        relevant = dp.find_similar_chunks(sc_q, top_k=5)
        sc_resp = assistant.self_consistency_generate(sc_q, relevant, num_attempts=3)
        # Save as text if string, else JSON
        if isinstance(sc_resp, str):
            _save_result("self_consistency_response.txt", sc_resp, cfg, is_text=True)
        else:
            _save_result("self_consistency_response.json", sc_resp, cfg)
        print("   Self-consistency done.")
    except Exception as e:
        print("   ! Self-consistency failed:", e)
        traceback.print_exc()

    # Step 8: ReAct workflow demo
    print("\n8) ReAct research workflow demo")
    react_q = "What are the current trends in natural language processing?"
    try:
        react_resp = assistant.react_research_workflow(react_q)
        _save_result("react_workflow.json", react_resp, cfg)
        steps = react_resp.get("workflow_steps", []) if isinstance(react_resp, dict) else []
        print(f"   ReAct completed with {len(steps)} steps.")
    except Exception as e:
        print("   ! ReAct workflow failed:", e)
        traceback.print_exc()

    # Step 9: Agent coordination demos
    print("\n9) Agent demonstrations via Orchestrator")

    # Summarizer Agent (use first processed doc)
    first_doc = processed_doc_ids[0] if processed_doc_ids else None
    if first_doc:
        print(f"   Summarizer Agent -> doc_id: {first_doc}")
        try:
            summ = orchestrator.route_task("summarizer", {"doc_id": first_doc})
            _save_result("document_summary.json", summ, cfg)
            print("   Summarizer saved.")
        except Exception as e:
            print("   ! Summarizer failed:", e)
            traceback.print_exc()
    else:
        print("   No documents processed for summarizer demo.")

    # QA Agent
    print("   QA Agent (analytical) demo")
    try:
        qa_task = {"question": "What methodology was used in the research?", "type": "analytical"}
        qa_res = orchestrator.route_task("qa", qa_task)
        _save_result("qa_response.json", qa_res, cfg)
        print("   QA saved.")
    except Exception as e:
        print("   ! QA agent failed:", e)
        traceback.print_exc()

    # Research Workflow Agent
    print("   Research Workflow Agent demo")
    try:
        # supply topic and doc_ids for a better run
        wf_task = {"goal": "artificial intelligence applications", "doc_ids": processed_doc_ids}
        wf_res = orchestrator.route_task("workflow", wf_task)
        _save_result("research_workflow.json", wf_res, cfg)
        print("   Research workflow saved.")
    except Exception as e:
        print("   ! Workflow agent failed:", e)
        traceback.print_exc()

    # Verification Agent (via orchestrator if available)
    print("\n10) Verification demo")
    try:
        # Use assistant's verify function directly (or route to orchestrator if it has 'verify')
        test_answer = "Neural networks are computational models inspired by biological neural networks."
        test_query = "What are neural networks?"
        # try orchestrator first
        try:
            ver = orchestrator.route_task("verify", {"answer": test_answer, "question": test_query, "context": "Sample context"})
        except Exception:
            ver = assistant.verify_and_edit_answer(test_answer, test_query, "Sample context")
        _save_result("verification_result.json", ver, cfg)
        print("   Verification result saved.")
    except Exception as e:
        print("   ! Verification failed:", e)
        traceback.print_exc()

    # Final demo report
    print("\n11) Generating demo report")
    try:
        report = _generate_demo_report(cfg, dp)
        _save_result("demo_report.md", report, cfg, is_text=True)
        print("\nDemo complete. Results saved to:", cfg.RESULTS_DIR)
    except Exception as e:
        print("   ! Failed generating demo report:", e)
        traceback.print_exc()


def _save_result(filename, data, config, is_text=False):
    """Save result safely (JSON or plain text)."""
    try:
        results_dir = getattr(config, "RESULTS_DIR", "results")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)

        if is_text or isinstance(data, str):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"   Saved: {filename}")
    except Exception as e:
        print(f"   Error saving {filename}: {e}")
        traceback.print_exc()


def _generate_demo_report(config, doc_processor):
    """Create a markdown demo report summarizing outcomes."""
    stats = doc_processor.get_document_stats()
    report = f"""# ResearchGPT Assistant - Demo Report

## System
- Model: {getattr(config, 'MODEL_NAME', 'unknown')}
- Temperature: {getattr(config, 'TEMPERATURE', 'unknown')}
- Max tokens: {getattr(config, 'MAX_TOKENS', 'unknown')}

## Document stats
{json.dumps(stats, indent=2, ensure_ascii=False)}

## Notes
- Results saved in: {getattr(config, 'RESULTS_DIR', 'results')}
- If an agent failed, there will be an error message in the console.

"""
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print("\nDemo failed:", str(e))
        traceback.print_exc()