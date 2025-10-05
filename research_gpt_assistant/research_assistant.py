# research_assistant.py

import json
import time
import difflib
import logging
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

class ResearchGPTAssistant:
    def __init__(self, config, doc_processor):
        """
        Initialize the ResearchGPT Assistant.
        Args:
            config (Config): Configuration object
            doc_processor (DocumentProcessor): Document processor instance
        """
        self.config = config
        self.doc_processor = doc_processor
        # Initialize Mistral client
        self.client = MistralClient(api_key=self.config.MISTRAL_API_KEY)

    def _call_mistral(self, prompt, temperature=0.2):
        """
        Call Mistral API with a prompt and return response text.
        """
        try:
            response = self.client.chat_completion.create(
                model="mistral-7b-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Mistral API call failed: {str(e)}")
            return "Error: Mistral API call failed."

    def _format_context(self, context_chunks, max_chunks=5, max_chars=3000):
        if not context_chunks:
            return ""
        parts = []
        for c in context_chunks[:max_chunks]:
            if isinstance(c, (list, tuple)) and len(c) >= 1:
                parts.append(c[0])
            elif isinstance(c, str):
                parts.append(c)
        joined = "\n\n---\n\n".join(parts)
        if len(joined) <= max_chars:
            return joined
        truncated = joined[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > int(max_chars * 0.3):
            truncated = truncated[: last_period + 1]
        return truncated

    def chain_of_thought_reasoning(self, query, context_chunks):
        context = self._format_context(context_chunks)
        cot_template = (
            "You are an expert research assistant. Use step-by-step reasoning to answer the question.\n\n"
            "Context (relevant excerpts):\n{context}\n\n"
            "Question: {query}\n\n"
            "Instructions:\n"
            "- First write your step-by-step chain-of-thought reasoning, labeling sections 'Step 1', 'Step 2', etc.\n"
            "- After the reasoning, write a short final answer under the header 'Final Answer:'.\n"
            "- Be concise in the final answer (1-3 sentences).\n"
        )
        prompt = cot_template.format(context=context or "No context available.", query=query)
        response = self._call_mistral(prompt, temperature=0.2)
        return response

    def self_consistency_generate(self, query, context_chunks, num_attempts=3):
        context = self._format_context(context_chunks)
        attempts = []
        for i in range(max(1, num_attempts)):
            temp = min(0.1 + i * 0.25, 0.9)
            prompt = (
                "Provide an answer to the question below. Keep the answer focused and cite relevant context if possible.\n\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".format(
                    context=context or "No context available.", query=query)
            )
            resp = self._call_mistral(prompt, temperature=temp)
            attempts.append(resp.strip())

        if not attempts:
            return ""
        if len(attempts) == 1:
            return attempts[0]

        scores = []
        for i, a in enumerate(attempts):
            total = 0.0
            for j, b in enumerate(attempts):
                if i == j:
                    continue
                total += difflib.SequenceMatcher(None, a, b).ratio()
            avg = total / max(1, (len(attempts) - 1))
            scores.append((avg, i, a))

        scores.sort(reverse=True, key=lambda t: t[0])
        return scores[0][2]

    def react_research_workflow(self, query, max_steps=5):
        workflow_steps = []
        observations = []
        for step in range(max_steps):
            prompt = (
                "You are running a short research workflow for the query:\n\n"
                f"Question: {query}\n\n"
                f"Previous observations (short):\n{(' | '.join(observations[-5:]) if observations else 'None')}\n\n"
                "Produce a short 'Thought:' describing what you will do next, and an 'Action:' chosen from [SEARCH, SUMMARIZE, STOP].\n"
                "If you choose SEARCH, provide a one-line 'Action Query:' to use for document search.\n\n"
                "Format (exactly):\nThought: <text>\nAction: SEARCH|SUMMARIZE|STOP\nAction Query: <one-line search query if SEARCH else leave blank>\n"
            )
            raw = self._call_mistral(prompt, temperature=0.3)
            thought = ""
            action = ""
            action_query = ""
            try:
                for line in raw.splitlines():
                    if line.strip().lower().startswith("thought:"):
                        thought = line.split(":", 1)[1].strip()
                    elif line.strip().lower().startswith("action:"):
                        action = line.split(":", 1)[1].strip().upper()
                    elif line.strip().lower().startswith("action query:"):
                        action_query = line.split(":", 1)[1].strip()
            except Exception:
                thought = raw.strip()

            observation = ""
            if action == "SEARCH":
                search_q = action_query or query
                try:
                    results = self.doc_processor.find_similar_chunks(search_q, top_k=3)
                    observation = results[0][0] if results else "No relevant results"
                except Exception as e:
                    observation = f"Search failed: {str(e)}"
            elif action == "SUMMARIZE":
                combined = "\n\n".join(observations[-5:]) or "No observations to summarize."
                sum_prompt = f"Summarize the following observations into 3-5 key points:\n\n{combined}"
                observation = self._call_mistral(sum_prompt, temperature=0.2)
            else:
                observation = "Stopping workflow."
                workflow_steps.append({
                    "step": step + 1,
                    "thought": thought,
                    "action": action,
                    "action_query": action_query,
                    "observation": observation
                })
                break

            workflow_steps.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_query": action_query,
                "observation": observation
            })
            observations.append(observation)

            if self._should_conclude_workflow(observation):
                break

        synth_context = "\n\n".join(observations[-6:]) or "No observations gathered."
        final_prompt = (
            "Using the gathered observations below, provide a concise final answer to the original question.\n\n"
            f"Question: {query}\n\nObservations:\n{synth_context}\n\nFinal Answer:"
        )
        final_answer = self._call_mistral(final_prompt, temperature=0.2)

        return {"workflow_steps": workflow_steps, "final_answer": final_answer}

    def _should_conclude_workflow(self, observation):
        if not observation:
            return False
        low = observation.lower()
        if "no relevant" in low or "no results" in low:
            return True
        if len(observation.split()) > 80 and ("conclusion" in low or "final" in low or "in summary" in low):
            return True
        return False

    def _try_parse_json_from_text(self, text):
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def verify_and_edit_answer(self, answer, original_query, context):
        verify_prompt = (
            "You are a careful research assistant. Verify the factual claims in 'answer' against the provided context.\n\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Original question:\n"
            f"{original_query}\n\n"
            "Context (evidence):\n"
            f"{context}\n\n"
            "Tasks:\n"
            "1) Check each factual claim and indicate whether it's supported by the context.\n"
            "2) If some claims are unsupported or uncertain, rewrite the answer to be accurate and precise.\n"
            "3) Return a JSON object ONLY with these keys: improved_answer (string), confidence (number 0.0-1.0), notes (short string).\n\n"
            "Example output:\n"
            '{"improved_answer": "Corrected answer...", "confidence": 0.75, "notes": "Some claims lacked direct support."}\n'
        )

        raw = self._call_mistral(verify_prompt, temperature=0.15)
        parsed = self._try_parse_json_from_text(raw)
        if parsed:
            improved = parsed.get("improved_answer", answer)
            confidence = parsed.get("confidence", 0.5)
            notes = parsed.get("notes", "")
            verification_result_text = raw
        else:
            backup_prompt = (
                "The previous verification output could not be parsed as JSON. "
                "Please provide a short improved answer (1-3 sentences) and then a single-line confidence estimate (0.0-1.0).\n\n"
                f"Answer:\n{answer}\n\nContext:\n{context}\n\nImproved Answer:"
            )
            raw2 = self._call_mistral(backup_prompt, temperature=0.2)
            improved = raw2.strip()
            confidence = 0.5
            notes = "Could not parse structured verification output."
            verification_result_text = raw2

        return {
            "original_answer": answer,
            "verification_result": verification_result_text,
            "improved_answer": improved,
            "confidence_score": float(confidence),
            "notes": notes
        }
