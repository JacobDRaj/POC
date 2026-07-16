from config import call_llm_with_cache, count_token_usage
from utils.logger import logger
import json
import re

def intent_classifier(user_question: str, conversation_memory: dict = None, token_usage: dict = None) -> str:
    """Stage 1: Intent Classifier - Classify user question intent."""
    try:
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-3:]
            lines = []
            for entry in recent:
                lines.append(f"User: {entry.get('user_question', '')}")
                lines.append(f"Assistant: {entry.get('assistant_answer', '') or entry.get('sql_query', '')}")
            conv_context = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are an Intent Classification assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Your job is to analyze the user's question and conversation history to classify the user's intent.\n\n"
            f"{conv_context}"
            f"USER QUESTION: \"{user_question}\"\n\n"
            "Choose exactly one of the following classification labels:\n"
            "- 'data_retrieval': requests a list or raw rows of data (e.g., 'show latest 10 filings', 'list attorneys in NY', 'show top 3 risk cases', 'highest score cases'). Note: queries asking for 'top N', 'latest N', 'highest/lowest risk cases', 'highest match scores', or listing cases/records are 'data_retrieval' intent, NOT aggregation.\n"
            "- 'aggregation': requests counts, sums, averages, grouping, or statistics (e.g., 'total filings', 'cases by state', 'average match score').\n"
            "- 'filter': requests filtering records based on values without aggregation (e.g., 'show only active cases').\n"
            "- 'visualization': explicitly requests a chart, plot, pie, bar, graph, or distribution representation (e.g., 'draw a pie chart of chapter distribution', 'plot filings over time').\n"
            "- 'follow_up': references the previous question/results, or uses pronouns/reference words like 'these', 'those', 'that', 'convert to chart' (e.g., 'filter those by active status', 'plot that').\n"
            "- 'unclear': off-topic queries, greetings, or ambiguous text (e.g., 'hello', 'what is bankruptcy').\n\n"
            "Respond ONLY with a JSON object containing the classified intent, formatted as:\n"
            "{\n"
            '  "intent": "data_retrieval|aggregation|filter|visualization|follow_up|unclear"\n'
            "}\n"
            "Do not include markdown, code blocks, or explanation text outside the JSON."
        )
        
        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v
        
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            
        data = json.loads(cleaned)
        return data.get("intent", "data_retrieval")
    except Exception as e:
        logger.warning("Intent classification failed, falling back to 'data_retrieval': %s", e)
        return "data_retrieval"

