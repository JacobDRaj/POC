import re
import json
from utils.logger import logger
from config import call_llm_with_cache, count_token_usage
from query.sql_executor import clean_sql_for_whitespace, validate_sql_with_judge
from database.schema_loader import load_sample_data_knowledge
from utils.sql_utils import extract_sql_from_response


# =============================================================================
# SQL CONVERSION & VALIDATION
# =============================================================================



def column_mapper(extracted_entities: dict, schema: dict, main_schema: dict = None, token_usage: dict = None) -> dict:
    """Stage 3: Column Mapper - Map extracted entities to exact database column names."""
    try:
        def _format_cols(s):
            if not s:
                return ""
            return "\n".join([f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}" for col in s.get("columns", [])])

        schema_text = f"Target Table: {schema['table_name']}\nColumns:\n" + _format_cols(schema)
        if main_schema and schema['table_name'] != main_schema['table_name']:
            schema_text += f"\n\nMain Table: {main_schema['table_name']}\nColumns:\n" + _format_cols(main_schema)

        prompt = (
            "You are a Database Column Mapping assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Your job is to map extracted user question entities/concepts to the exact, case-sensitive database column names from the schema.\n\n"
            f"DATABASE SCHEMA:\n{schema_text}\n\n"
            f"EXTRACTED ENTITIES:\n{json.dumps(extracted_entities, indent=2)}\n\n"
            "MAPPING RULES:\n"
            "- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' -> use 'status' column.\n"
            "- 'active/inactive flag' -> use 'active_status' column.\n"
            "- 'new/closed/reopened/update record stage' -> 'record_type' column.\n"
            "- 'bankruptcy chapter', 'chapter 7/11/13' -> 'chapter' column.\n"
            "- 'filing date', 'when filed' -> 'date_filed' column.\n"
            "- 'debtor name' -> use 'first_name' and 'last_name'.\n"
            "- 'state' -> 'state' column.\n"
            "- 'client', 'client name', 'client code', 'VP', 'SYF' or any client identifier -> 'client_name' column. NEVER map to 'match_code'.\n"
            "- 'attorney', 'lawyer' -> 'attorney_first_name', 'attorney_last_name', or 'attorney_dba' columns.\n"
            "- 'year', 'yearly', 'by year', 'annual' -> use 'date_filed' (prefer if available) or 'open_date'.\n"
            "- 'month', 'monthly', 'by month' -> use 'date_filed' (prefer if available) or 'open_date'.\n"
            "- 'matchcode', 'match_code' -> 'match_code' column.\n"
            "- 'record_type', 'record lifecycle', 'new/closed/reopened/update' -> 'record_type' column. When record_type contains filter values like ['New', 'Closed'], map it to 'record_type' column.\n"
            "- 'city', 'debtor city' -> 'city' column.\n"
            "- 'trustee city', 'trusty city' -> 'trustee_city' column.\n"
            "- 'trustee name', 'trusty name' -> 'trustee_name' column.\n"
            "- 'risk', 'risk score', 'match score', 'match_score' -> use 'match_score' column.\n"
            "- 'sort_by_field' -> Map to 'match_score' if value is 'risk' or 'score', 'date_filed' if value is 'date', or other appropriate column name. If null, map to null.\n"
            "- 'limit' and 'sort_order' -> Keep their values as-is (e.g. integer or string). Do not map them to columns.\n"
            "- 'group_by_fields' -> Only map the elements that are inside the input 'group_by_fields' list. If 'group_by_fields' is empty in the input, the output 'group_by_fields' MUST be empty [].\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example 1:\n"
            "Entities: {\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": "P2",\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year"],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n"
            "Mapping:\n"
            "{\n"
            '  "status": "status",\n'
            '  "chapter": "chapter",\n'
            '  "state": "state",\n'
            '  "date_or_year": "date_filed",\n'
            '  "attorney_name": ["attorney_first_name", "attorney_last_name", "attorney_dba"],\n'
            '  "debtor_name": ["first_name", "last_name"],\n'
            '  "match_code": "match_code",\n'
            '  "client_name": "client_name",\n'
            '  "city": "city",\n'
            '  "trustee_name": "trustee_name",\n'
            '  "trustee_city": "trustee_city",\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["date_filed"],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example 2:\n"
            "Entities: {\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "risk",\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": 3,\n'
            '  "sort_order": "desc"\n'
            "}\n"
            "Mapping:\n"
            "{\n"
            '  "status": "status",\n'
            '  "chapter": "chapter",\n'
            '  "state": "state",\n'
            '  "date_or_year": "date_filed",\n'
            '  "attorney_name": ["attorney_first_name", "attorney_last_name", "attorney_dba"],\n'
            '  "debtor_name": ["first_name", "last_name"],\n'
            '  "match_code": "match_code",\n'
            '  "client_name": "client_name",\n'
            '  "city": "city",\n'
            '  "trustee_name": "trustee_name",\n'
            '  "trustee_city": "trustee_city",\n'
            '  "sort_by_field": "match_score",\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": 3,\n'
            '  "sort_order": "desc"\n'
            "}\n\n"
            "Output a JSON object mapping each key in EXTRACTED ENTITIES to the exact database column name(s) (as a list or string, or null if no mapping is needed) or keeping its value as-is. Follow the few-shot example structure exactly.\n"
            "Respond ONLY with this JSON. Do not include markdown or explanations."
        )

        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Column mapping failed: %s", e)
        return {}



def sql_builder(user_question: str, intent: str, extracted_entities: dict, column_mapping: dict, schema: dict, main_schema: dict = None, conversation_memory: dict = None, token_usage: dict = None) -> str:
    """Stage 4: SQL Builder - Construct the SQLite query based on intent, entities, and mapping."""
    try:
        def _format_cols(s):
            if not s:
                return ""
            return "\n".join([f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}" for col in s.get("columns", [])])

        schema_desc = f"Table: {schema['table_name']}\nColumns:\n" + _format_cols(schema)
        if main_schema and schema['table_name'] != main_schema['table_name']:
            schema_desc += f"\n\nTable: {main_schema['table_name']}\nColumns:\n" + _format_cols(main_schema)

        # Load sample data as knowledge base for grounding the SQL generation
        sample_data_str = load_sample_data_knowledge()
        sample_data_section = ""
        if sample_data_str:
            # Truncate to avoid blowing up the prompt (keep first 3000 chars)
            truncated = sample_data_str[:3000]
            sample_data_section = (
                "SAMPLE DATA (first 20 rows from the live database — use this to understand actual values, "
                "column formats, match_code distribution, date formats, and data patterns before writing SQL):\n"
                "```csv\n"
                f"{truncated}\n"
                "```\n\n"
            )
        print("extracted_entities",json.dumps(extracted_entities, indent=2))
        print("column_mapping",json.dumps(column_mapping, indent=2))
        print("schema_desc",schema_desc)
        print("INTENT",intent)
        print("user_question",user_question)
        prompt = (
            "You are a Senior SQL Analyst specializing in SQLite query construction for bankruptcy analytics.\n"
            "Your task is to build a single valid SQLite query that answers the user's question, using the provided entity extraction and column mappings.\n\n"
            f"{sample_data_section}"
            f"USER QUESTION: \"{user_question}\"\n"
            f"INTENT: {intent}\n"
            f"EXTRACTED ENTITIES:\n{json.dumps(extracted_entities, indent=2)}\n"
            f"COLUMN MAPPINGS:\n{json.dumps(column_mapping, indent=2)}\n"
            f"DATABASE SCHEMA:\n{schema_desc}\n\n"
            "CRITICAL CONSTRUCT RULES:\n"
            "1. You MUST use only columns and tables that are listed in the schema.\n"
            "2. Table and column names are case-sensitive. Use them exactly as shown.\n"
            "3. Date columns are filtered/grouped by year using SQLite strftime, e.g. `strftime('%Y', date_column) = '2024'` or `substr(date_column, 1, 4) = '2024'`.\n"
            "4. If text filtering is done on status or other columns, use `TRIM(column_name)` to handle trailing/leading whitespace in text fields.\n"
            "5. If a distribution/breakdown of X by Y is requested (e.g., 'chapter distribution for each state'), select both columns, group by both columns, and count the total records.\n"
            "6. Keep the query clean: do not end with a semicolon ';', and only use a SELECT statement.\n"
            "7. Use the SAMPLE DATA above to infer exact column value formats (e.g. match_code uses values like 'P1', 'P2', 'M1'; status uses 'Active', 'Closed'; date_filed is in YYYY-MM-DD format).\n"
            "8. When filtering by match_code (e.g. 'P2 matchcode'), ALWAYS add a WHERE clause: WHERE TRIM(match_code) = 'P2'. Do NOT omit this filter — the goal is a filtered distribution, not all match codes.\n"
            "9. When filtering by client (e.g. 'VP yearwise'), ALWAYS add a WHERE clause using 'client_name' column: WHERE TRIM(client_name) = 'VP'. NEVER use match_code for client filtering.\n"
            "9b. When filtering by record_type (e.g. 'new vs closed cases'), add WHERE TRIM(record_type) IN ('New', 'Closed') and GROUP BY record_type. If record_type is a list in EXTRACTED ENTITIES, use IN with all values. If a single string, use = with that value.\n"
            "10. If a specific year (e.g., '2024') is requested in the user query and present in `date_or_year` in EXTRACTED ENTITIES, you MUST include a WHERE filter for that year (e.g., `strftime('%Y', date_filed) = '2024'` or `substr(date_filed, 1, 4) = '2024'`). DO NOT retrieve or aggregate over all years when a specific year is explicitly requested in the query. If the user asks for a distribution/breakdown within a single year (e.g., '2024 distribution') but does not specify another grouping attribute (like chapter or state), group by month (e.g. `strftime('%m', date_filed) AS month` or `substr(date_filed, 6, 2) AS month`) to show a meaningful distribution over the months of 2024.\n"
            "11. If the user requests sorting or limiting (e.g., 'top 3 risk cases', 'highest score cases', or `limit` and `sort_by_field` are set in EXTRACTED ENTITIES), you MUST construct a SELECT query that retrieves case records (selecting relevant columns like match_score, first_name, last_name, client_name, match_code, status, date_filed, case_number), order by the mapped `sort_by_field` column (e.g. match_score for risk/score) according to the `sort_order` (e.g. `ORDER BY match_score DESC`), and apply the `limit` (e.g. `LIMIT 3`). DO NOT group or count unless specifically asked.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Please Refer to the below examples for Query Generations:\n\n"
            "Example 1:\n"
            "Question: \"P2 matchcode yearwise distribution\"\n"
            "Entities: {\"match_code\": \"P2\", \"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"match_code\": \"match_code\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data WHERE TRIM(match_code) = 'P2' GROUP BY year ORDER BY year\n\n"
            "Example VP:\n"
            "Question: \"VP yearwise distribution\"\n"
            "Entities: {\"client_name\": \"VP\", \"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' GROUP BY year ORDER BY year\n\n"
            "Example VP 2024:\n"
            "Question: \"VP 2024 distributions\"\n"
            "Entities: {\"client_name\": \"VP\", \"date_or_year\": \"2024\", \"group_by_fields\": [\"month\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"date_or_year\": \"date_filed\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%m', date_filed) AS month, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' AND strftime('%Y', date_filed) = '2024' GROUP BY month ORDER BY month\n\n"
            "Example 2:\n"
            "Question: \"chapterwise status count\"\n"
            "Entities: {\"group_by_fields\": [\"chapter\", \"status\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"group_by_fields\": [\"chapter\", \"status\"]}\n"
            "Sample SQL: SELECT chapter, status, COUNT(*) AS count FROM uploaded_data GROUP BY chapter, status\n\n"
            "Example 3:\n"
            "Question: \"M1 matchcode statewise breakdown\"\n"
            "Entities: {\"match_code\": \"M1\", \"group_by_fields\": [\"state\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"match_code\": \"match_code\", \"group_by_fields\": [\"state\"]}\n"
            "Sample SQL: SELECT state, COUNT(*) AS count FROM uploaded_data WHERE TRIM(match_code) = 'M1' GROUP BY state ORDER BY count DESC\n\n"
            "Example 4:\n"
            "Question: \"show me top 3 risk cases\"\n"
            "Entities: {\"limit\": 3, \"sort_by_field\": \"risk\", \"sort_order\": \"desc\"}\n"
            "Mappings: {\"limit\": 3, \"sort_by_field\": \"match_score\", \"sort_order\": \"desc\"}\n"
            "Sample SQL: SELECT match_score, first_name, last_name, client_name, match_code, status, date_filed FROM uploaded_data ORDER BY match_score DESC LIMIT 3\n\n"
            "Example 5 (record_type — new vs closed):\n"
            "Question: \"VP closed vs new cases\"\n"
            "Entities: {\"client_name\": \"VP\", \"record_type\": [\"New\", \"Closed\"], \"group_by_fields\": [\"record_type\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"record_type\": \"record_type\", \"group_by_fields\": [\"record_type\"]}\n"
            "Sample SQL: SELECT record_type, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' AND TRIM(record_type) IN ('New', 'Closed') GROUP BY record_type ORDER BY count DESC\n\n"
            "Example 6 (status — active vs closed, NOT record_type):\n"
            "Question: \"active vs closed breakdown\"\n"
            "Entities: {\"status\": [\"Active\", \"Closed\"], \"group_by_fields\": [\"status\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"status\": \"status\", \"group_by_fields\": [\"status\"]}\n"
            "Sample SQL: SELECT status, COUNT(*) AS count FROM uploaded_data WHERE TRIM(status) IN ('Active', 'Closed') GROUP BY status ORDER BY count DESC\n\n"
            "Output ONLY a JSON object with a single key `sql` whose value is the built SQLite query string:\n"
            "{\n"
            '  "sql": "SELECT ..."\n'
            "}\n"
            "Do not include markdown or explanations."
        )

        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v

        sql_query = extract_sql_from_response(response)
        return sql_query
    except Exception as e:
        logger.exception("SQL building failed: %s", e)
        return ""



def sql_validator(sql_query: str, user_question: str, schema: dict, main_schema: dict = None, token_usage: dict = None) -> dict:
    """Stage 5: SQL Validator - Validates the query for safety, schema alignment, and repairs errors."""
    return validate_sql_with_judge(sql_query, user_question, schema, token_usage=token_usage, main_schema=main_schema)