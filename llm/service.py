import re
import json
from config import call_llm_haiku
from utils.logger import logger

# =============================================================================
# SMART QUERY UNDERSTANDING (HAIKU PRE-PROCESSOR)
# =============================================================================

def smart_query_understanding(user_query: str, schema: dict, conversation_memory: dict = None) -> dict:
    """
    Use a dedicated Claude Haiku session to understand any user question by
    referencing the full schema.json as a knowledge base.

    Returns a dict with:
      - normalized_query: A rewritten, schema-aware version of the user query
      - intent: One of 'data_retrieval', 'aggregation', 'filter', 'visualization', 'follow_up', 'unclear'
      - relevant_columns: List of column names from the schema that are relevant
      - time_filter: Any detected time/date constraint (or None)
      - is_answerable: True if the query can be answered from the schema
      - clarification_needed: A short message if the query is unclear (or None)
    """
    try:
        # Build a compact schema summary for the prompt
        schema_cols = schema.get("columns", []) if schema else []
        table_name = schema.get("table_name", "bankruptcy_data") if schema else "bankruptcy_data"
        schema_summary_lines = [
            f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}"
            for col in schema_cols
        ]
        schema_text = f"Table: {table_name}\nColumns:\n" + "\n".join(schema_summary_lines)
        print("schema_text",schema_text)

        # Build recent conversation context
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-3:]
            lines = []
            for entry in recent:
                lines.append(f"  User: {entry.get('user_question', '')}")
                if entry.get("sql_query"):
                    lines.append(f"  SQL result: {entry.get('record_count', 0)} rows")
            conv_context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are a Senior Business Intelligence Query Understanding Assistant for a Bankruptcy Analytics platform.\n"
            "Your responsibility is to understand the user's business intent, analytical objective, and conversational context before mapping the request to the database schema.\n"
            "Think like an experienced BI Analyst rather than a keyword matching engine.\n"
            "Preserve the user's business meaning while producing an unambiguous normalized query.\n\n"
            f"DATABASE SCHEMA:\n{schema_text}\n\n"
            "BUSINESS UNDERSTANDING:\n"
            "- Understand what business question the user is trying to answer.\n"
            "- Continue the current analytical journey instead of treating every request as independent.\n"
            "- Use the conversation history whenever the question refers to previous results.\n\n"
            f"{conv_context}"
            "CONVERSATION RULES:\n"
            "- Resolve references like 'these', 'those', 'them', 'previous', 'same', 'top', and 'above' using the latest conversation.\n"
            "- Preserve previous filters unless the user explicitly changes them.\n"
            "- Continue the current analysis naturally.\n\n"


            "USER QUESTION: \"" + user_query + "\"\n\n"
            "KEY COLUMN MAPPING RULES (use these to resolve ambiguity):\n"
            "- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' → use the 'status' column (values: Active, Closed, Dismissed, Converted, Pending, Discharged). Note: if the query specifically requests a comparison/breakdown between specific statuses (like 'active vs closed'), only return those specific statuses.\n"
            "- 'active/inactive flag' → use the 'active_status' column\n"
            "- 'new/closed/reopened/update record stage' → use the 'record_type' column (values: New, Closed, Reopened, Update)\n"
            "- CRITICAL DISAMBIGUATION: 'New' is EXCLUSIVELY a record_type value (NEVER a status value). When the user says 'new vs closed', 'closed vs new', 'new cases', always use 'record_type' column. Use 'status' only for legal status values like Active, Pending, Dismissed, Discharged, Converted.\n"
            "- 'bankruptcy chapter', 'chapter 7/11/13' → use the 'chapter' column\n"
            "- 'filing date', 'when filed' → use the 'date_filed' column\n"
            "- 'debtor name' → use 'first_name' and 'last_name'\n"
            "- 'state' (debtor location) → use the 'state' column\n"
            "- 'client', 'client name', 'client code', 'VP', 'SYF' or any client identifier → use the 'client_name' column. NEVER map client terms to 'match_code'.\n"
            "- 'attorney', 'lawyer' → use the 'attorney_first_name', 'attorney_last_name' or 'attorney_dba' columns. When analyzing or grouping by attorney, use 'attorney_dba' (or include first and last names along with 'attorney_dba') to represent different practitioners/firms.\n"
            "- 'year', 'yearly', 'by year', 'annual' → use the 'date_filed' or 'open_date' column (or date columns) and extract the year using SQLite's strftime('%Y', ...) function.\n"
            "- 'matchcode', 'match code' (values like P1, P2, P3, M1, M2, M3) → use the 'match_code' column.\n"
            "- 'P2 matchcode' → matches the 'match_code' column with value 'P2'.\n\n"
            "QUERY UNDERSTANDING PRINCIPLES:\n"
            "- Determine what insight the user is seeking.\n"
            "- Distinguish between retrieval, aggregation, comparison, trend, ranking, distribution, and visualization requests.\n"
            "- Preserve every explicit filter mentioned by the user.\n"
            "- Never change the user's business intent.\n\n"
            "INSTRUCTIONS:\n"
            "1. Understand the user's intent (data_retrieval, aggregation, filter, visualization, follow_up, or unclear).\n"
            "2. Rewrite the question as a clear, unambiguous PLAIN ENGLISH query using EXACT column names from the schema.\n"
                "   - Think from a business perspective before rewriting.\n"
                "   - normalized_query should explain WHAT the user wants instead of HOW SQL should retrieve it.\n"
                "   - Make the rewritten query understandable by both business and technical users.\n"
            "   IMPORTANT: normalized_query must be a natural language sentence, NOT SQL code.\n"
            "   - Map informal terms to schema columns using the KEY COLUMN MAPPING RULES above.\n"
            "   - Use ONLY column names that exist in the schema. Never invent column names.\n"
            "   - Expand abbreviations and correct spelling based on schema knowledge.\n"
            "   - Preserve all specific filter values (e.g. state names like 'NY', status names like 'Active', chapter numbers like '7', and match codes like 'P2' or 'M1'). Never generalize specific filter values or drop them.\n"
            "   - Never remove existing business filters.\n"
            "   - Never broaden the scope of the user's request.\n"
            "   - Never replace explicit entities with generic terms.\n"
            "   - If the user asked for a 'bar chart', 'pie chart', 'plot', etc., keep those words in the normalized_query.\n"
            "   - Example: 'bar chart of filings by state' → 'Show a bar chart of the count grouped by state column'\n"
            "   - Example: 'show debtors in NY' → 'Retrieve records where state = NY showing first_name, last_name, city, state'\n"
            "   - Example: 'active vs closed case breakdown' → 'Show count of cases grouped by status column filtered to show only status is Active or Closed'\n"
            "   - Example: 'Analyze filings by year and status' → 'Retrieve counts of records grouped by the year (extracted using strftime from date_filed or open_date column) and status column'\n"
            "   - Example: 'Show chapter distribution for each state' → 'Retrieve counts of records grouped by state and chapter columns showing both columns'\n"
            "   - Example: 'P2 matchcode yearwise distribution' → 'Show count of cases grouped by year (extracted from date_filed) where match_code is P2'\n"
            "   - Example: 'VP yearwise distribution' → 'Show count of cases grouped by year (extracted from date_filed) where client_name is VP'\n"
            "   - Example: 'VP closed vs new cases' → 'Show count of cases grouped by record_type column where client_name is VP and record_type is New or Closed'\n"
            "   - Example: 'active vs closed breakdown' → 'Show count of cases grouped by status column filtered to status is Active or Closed'\n"
            "3. List only the minimum required schema columns needed to answer the user's request.\n"
                "Exclude unrelated columns.\n"
            "4. Extract any time/date filter mentioned (e.g., '2024', 'last year', 'Q1 2023') or null.\n"
            "5. Determine if the question is answerable from the schema (true/false).\n"
            "6. If clarification is required, ask exactly one short business-focused clarification question.\n"
"Only request clarification when absolutely necessary.\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "QUALITY CHECK:\n"
        "- Verify every column exists in the schema.\n"
        "- Preserve the user's business intent.\n"
        "- Ensure normalized_query contains no SQL.\n"
        "- Resolve conversational references correctly.\n"
        "- Keep the response concise and business-friendly.\n\n"
            "{\n"
            '  "normalized_query": "plain English rewrite - never SQL",\n'
            '  "intent": "data_retrieval|aggregation|filter|visualization|follow_up|unclear",\n'
            '  "relevant_columns": ["col1", "col2"],\n'
            '  "time_filter": "2024" or null,\n'
            '  "is_answerable": true or false,\n'
            '  "clarification_needed": "..." or null\n'
            "}\n"
            "Do NOT include markdown, SQL code, or any text outside the JSON."
            "- Always prefer business-friendly columns over technical identifier columns.\n"
            "- If multiple columns match, choose the column with the strongest business meaning.\n"
            "- Never invent columns that are not present in the schema.\n"
            "- Resolve common business synonyms using schema descriptions whenever possible.\n"
        )

        response = call_llm_haiku(prompt)
        logger.info("Smart query understanding response: %s", response[:300] if response else "EMPTY")

        # Parse the JSON response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        result = json.loads(cleaned)

        # Validate: reject normalized_query if it looks like SQL
        normalized = result.get("normalized_query", user_query).strip()
        if normalized.strip().upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
            logger.warning("Haiku returned SQL as normalized_query — falling back to user_query")
            normalized = user_query

        return {
            "normalized_query": normalized,
            "intent": result.get("intent", "data_retrieval"),
            "relevant_columns": result.get("relevant_columns", []),
            "time_filter": result.get("time_filter"),
            "is_answerable": bool(result.get("is_answerable", True)),
            "clarification_needed": result.get("clarification_needed"),
        }

    except Exception as e:
        logger.warning("Smart query understanding failed (falling back to original query): %s", e)
        return {
            "normalized_query": user_query,
            "intent": "data_retrieval",
            "relevant_columns": [],
            "time_filter": None,
            "is_answerable": True,
            "clarification_needed": None,
        }

