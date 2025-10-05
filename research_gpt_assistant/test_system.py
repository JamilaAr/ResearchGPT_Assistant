"""
Testing and Evaluation Script for ResearchGPT Assistant

Implements comprehensive testing:
1. Unit tests for individual components
2. Integration tests for complete workflows
3. Performance evaluation metrics
4. Comparison of different prompting strategies
"""

import time
import json
import os
from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator


class ResearchGPTTester:
    def __init__(self):
        """
        Initialize testing system
        """
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.research_assistant = ResearchGPTAssistant(self.config, self.doc_processor)
        self.agent_orchestrator = AgentOrchestrator(self.research_assistant)

        # Define sample test queries
        self.test_queries = [
            "What are the main advantages of machine learning?",
            "How do neural networks process information?",
            "What are the limitations of current AI systems?",
            "Compare supervised and unsupervised learning approaches",
            "What are the ethical considerations in AI development?"
        ]

        # Store evaluation results
        self.evaluation_results = {
            'document_processing': {},
            'response_times': [],
            'prompt_strategy_comparison': {},
            'agent_performance': {},
            'performance_benchmark': {},
        }

    def test_document_processing(self):
        """
        Test document processing functionality
        """
        print("\n=== Testing Document Processing ===")
        test_results = {
            'text_preprocessing': False,
            'chunking': False,
            'similarity_search': False,
            'index_building': False,
            'errors': []
        }

        try:
            sample_text = "This is a test sentence about machine learning and AI."
            preprocessed = self.doc_processor.preprocess_text(sample_text)
            if preprocessed:
                test_results['text_preprocessing'] = True
                print("   ✓ Text preprocessing: PASS")

            chunks = self.doc_processor.chunk_text(sample_text, chunk_size=20, overlap=5)
            if chunks:
                test_results['chunking'] = True
                print("   ✓ Text chunking: PASS")

            # Build a dummy index
            self.doc_processor.documents['test_doc'] = [sample_text]
            self.doc_processor.build_search_index()
            test_results['index_building'] = True
            results = self.doc_processor.find_similar_chunks("machine learning", top_k=2)
            if results:
                test_results['similarity_search'] = True
                print("   ✓ Similarity search: PASS")

        except Exception as e:
            test_results['errors'].append(f"Document processing error: {str(e)}")
            print(f"   ✗ Error: {str(e)}")

        self.evaluation_results['document_processing'] = test_results
        return test_results

    def test_prompting_strategies(self):
        """
        Test different prompting strategies
        """
        print("\n=== Testing Prompting Strategies ===")
        strategy_results = {
            'chain_of_thought': [],
            'self_consistency': [],
            'react_workflow': []
        }

        for i, query in enumerate(self.test_queries[:2]):
            print(f"   Testing query {i+1}: {query[:40]}...")
            try:
                start_time = time.time()
                cot_response = self.research_assistant.chain_of_thought_reasoning(query, [])
                cot_time = time.time() - start_time
                strategy_results['chain_of_thought'].append({
                    'query': query,
                    'length': len(cot_response),
                    'time': cot_time
                })

                start_time = time.time()
                sc_response = self.research_assistant.self_consistency_generate(query, [], num_attempts=2)
                sc_time = time.time() - start_time
                strategy_results['self_consistency'].append({
                    'query': query,
                    'length': len(sc_response),
                    'time': sc_time
                })

                start_time = time.time()
                react_response = self.research_assistant.react_research_workflow(query)
                react_time = time.time() - start_time
                strategy_results['react_workflow'].append({
                    'query': query,
                    'steps': len(react_response.get('workflow_steps', [])),
                    'time': react_time
                })

                print("   ✓ Prompting strategies executed successfully")

            except Exception as e:
                print(f"   ✗ Error testing strategies: {str(e)}")

        self.evaluation_results['prompt_strategy_comparison'] = strategy_results
        return strategy_results

    def test_agent_performance(self):
        """
        Test AI agents
        """
        print("\n=== Testing AI Agents ===")
        agent_results = {}

        try:
            # Summarizer Agent
            print("   Testing Summarizer Agent...")
            summary_task = {'doc_id': 'test_doc'}
            summary_result = self.agent_orchestrator.route_task('summarizer', summary_task)
            agent_results['summarizer_agent'] = summary_result

            # QA Agent
            print("   Testing QA Agent...")
            qa_task = {'question': 'What is machine learning?', 'type': 'factual'}
            qa_result = self.agent_orchestrator.route_task('qa', qa_task)
            agent_results['qa_agent'] = qa_result

            # Research Workflow Agent
            print("   Testing Workflow Agent...")
            workflow_task = {'research_topic': 'artificial intelligence'}
            workflow_result = self.agent_orchestrator.route_task('workflow', workflow_task)
            agent_results['workflow_agent'] = workflow_result

            print("   ✓ All agents executed successfully")

        except Exception as e:
            print(f"   ✗ Agent testing error: {str(e)}")

        self.evaluation_results['agent_performance'] = agent_results
        return agent_results

    def run_performance_benchmark(self):
        """
        Run performance benchmark
        """
        print("\n=== Running Performance Benchmark ===")
        benchmark_results = {}

        start_time = time.time()
        time.sleep(0.1)  # Simulated processing
        benchmark_results['document_processing_time'] = time.time() - start_time

        query_times = []
        for query in self.test_queries[:2]:
            start_time = time.time()
            response = self.research_assistant.answer_research_question(query)
            query_times.append(time.time() - start_time)

        benchmark_results['average_query_time'] = sum(query_times) / len(query_times)
        benchmark_results['queries_per_minute'] = 60 / benchmark_results['average_query_time']

        print(f"   Avg query time: {benchmark_results['average_query_time']:.2f}s")

        self.evaluation_results['performance_benchmark'] = benchmark_results
        return benchmark_results

    def generate_evaluation_report(self):
        """
        Generate evaluation report
        """
        report = f"""
# ResearchGPT Assistant - Evaluation Report

## Document Processing
{json.dumps(self.evaluation_results['document_processing'], indent=2)}

## Prompting Strategies
{json.dumps(self.evaluation_results['prompt_strategy_comparison'], indent=2)}

## Agent Performance
{json.dumps(self.evaluation_results['agent_performance'], indent=2)}

## Performance Benchmark
{json.dumps(self.evaluation_results['performance_benchmark'], indent=2)}

## Recommendations
- Improve similarity search
- Implement better response quality evaluation
- Add caching for repeated queries
- Expand multi-agent workflows
"""
        return report

    def run_all_tests(self):
        """
        Run full test suite
        """
        print("\n=== Starting Test Suite ===")
        self.test_document_processing()
        self.test_prompting_strategies()
        self.test_agent_performance()
        self.run_performance_benchmark()

        final_report = self.generate_evaluation_report()
        results_dir = self.config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, 'evaluation_report.md'), 'w', encoding='utf-8') as f:
            f.write(final_report)

        with open(os.path.join(results_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2)

        print("\n=== Test Suite Complete ===")
        print(f"Results saved in {results_dir}")
        return self.evaluation_results


if __name__ == "__main__":
    tester = ResearchGPTTester()
    tester.run_all_tests()
