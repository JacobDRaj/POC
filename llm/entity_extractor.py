from utils.logger import logger
from config import call_llm_with_cache, count_token_usage
import json
import re




def entity_extractor(user_question: str, intent: str, conversation_memory: dict = None, token_usage: dict = None) -> dict:
    """Stage 2: Entity Extractor - Extract filter values, fields, and constraints."""
    try:
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-3:]
            lines = []
            for entry in recent:
                lines.append(f"User: {entry.get('user_question', '')}")
            conv_context = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are an Entity Extraction assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Extract entities, filter criteria, parameters, and constraints from the user's question.\n\n"
            f"INTENT: {intent}\n"
            f"{conv_context}"
            f"USER QUESTION: \"{user_question}\"\n\n"
            "Identify and extract the following entity types if present (use null if not found):\n"
            "1. status: e.g. Active, Closed, Dismissed, Converted, Pending, Discharged.\n"
            "2. chapter: e.g. 7, 11, 13.\n"
            "3. state: e.g. NY, CA, TX, Florida.\n"
            "4. date_or_year: e.g. 2024, last year, since 2023, open date range.\n"
            "5. attorney_name: e.g. John Doe, Smith.\n"
            "6. debtor_name: debtor first or last name.\n"
            "7. match_code: e.g. P1, P2, P3, M1, M2, M3 (ONLY set this for match code values — never for client names like VP or SYF).\n"
            "8. client_name: client or lender code, e.g. VP, SYF, SYNC. Set this when the user mentions a client name/code like 'VP yearwise', 'SYF distribution'. NEVER confuse with match_code.\n"
            "9. aggregation_type: count, sum, average, none.\n"
            "10. group_by_fields: fields to group results by (e.g. state, chapter, year, match_code, client_name, trustee_city).\n"
            "11. limit: number of records to return.\n"
            "12. sort_order: asc, desc, or null.\n"
            "13. city: general city name or debtor city (e.g. Houston, Chicago).\n"
            "14. trustee_name: trustee name.\n"
            "15. trustee_city: trustee city (e.g. Albany, Boston).\n"
            "16. sort_by_field: field to sort or order the results by (e.g. risk, score, date, open date).\n"
            "17. record_type: lifecycle stage of the bankruptcy record. Values: New, Update, Reopened, Closed. Use when the user mentions 'new cases', 'new filings', 'closed records', 'reopened cases', 'new vs closed'. Can be a single string (e.g. 'New') or a list of strings for comparisons (e.g. ['New', 'Closed']).\n\n"
            "KEY PARSING RULE FOR COLLOQUIAL / INCORRECT ENGLISH:\n"
            "- Suffixes like 'wise' (e.g. yearwise, chapterwise, statewise, matchcodewise) specify fields that must be placed inside the 'group_by_fields' list (e.g. ['year'], ['chapter'], ['state'], ['match_code']).\n"
            "- Abbreviated filters like 'P2 matchcode' should set 'match_code' to 'P2'.\n"
            "- Client codes like 'VP yearwise', 'SYF distribution' should set 'client_name' to the code (e.g. 'VP') and group_by_fields to ['year']. Do NOT set match_code for client names.\n"
            "- If the user asks for a distribution or breakdown for a single year/date (e.g., 'VP 2024 distributions', '2024 cases distribution'), set 'date_or_year' to that year (e.g. '2024') and set 'group_by_fields' to ['month'] (instead of 'year') so that the distribution is shown across the months of that year.\n"
            "- Phrasings like 'NY attorney list' should set 'state' to 'NY' and keep track of other attributes.\n"
            "- Phrasings like 'trusty city -Albany' or 'trustee city Albany' should set 'trustee_city' to 'Albany'. Do NOT map city names (e.g., 'Albany') to the 'state' field as 'NY' or any other state abbreviation.\n"
            "- Phrasings like 'trustee name Smith' should set 'trustee_name' to 'Smith'.\n"
            "- Phrasings like 'top 3 risk cases' or 'highest risk cases' should set 'limit' to 3 (or the specified number), 'sort_by_field' to 'risk', and 'sort_order' to 'desc'.\n"
            "- CRITICAL DISAMBIGUATION between 'record_type' and 'status':\n"
            "  * 'New' is ONLY a record_type value (NEVER a status). If user mentions 'new cases', 'new filings', or 'new vs closed', set 'record_type' NOT 'status'.\n"
            "  * 'Closed' appears in BOTH record_type and status. Disambiguate by context:\n"
            "    - 'new vs closed', 'closed vs new' → set record_type to ['New', 'Closed'] and group_by_fields to ['record_type']\n"
            "    - 'active vs closed', 'dismissed vs closed' → set status (Active/Dismissed are status values)\n"
            "    - 'reopened vs closed' → set record_type to ['Reopened', 'Closed'] (Reopened is a record_type value)\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example 1:\n"
            "Question: \"P2 matchcode yearwise distribution\"\n"
            "JSON:\n"
            "{\n"
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
            "}\n\n"
            "Example 2:\n"
            "Question: \"chapterwise status count\"\n"
            "JSON:\n"
            "{\n"
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
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["chapter", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example 3:\n"
            "Question: \"active status cases in NY\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Active",\n'
            '  "chapter": null,\n'
            '  "state": "NY",\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example VP client yearwise:\n"
            "Question: \"VP yearwise distribution\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": "VP",\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year"],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example VP client single year distribution:\n"
            "Question: \"VP 2024 distributions\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": "2024",\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": "VP",\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["month"],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example 4:\n"
            "Question: \"filing counts for trusty city -Albany\"\n"
            "JSON:\n"
            "{\n"
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
            '  "trustee_city": "Albany",\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null\n'
            "}\n\n"
            "Example 5 (record_type disambiguation - new vs closed):\n"
            "Question: \"VP closed vs new cases\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": "VP",\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["record_type"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": ["New", "Closed"]\n'
            "}\n\n"
            "Example 6 (status, NOT record_type - active vs closed):\n"
            "Question: \"active vs closed breakdown\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
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
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["status"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null\n'
            "}\n\n"
            "Respond ONLY with a JSON object containing the extracted entities matching this exact schema:\n"
            "{\n"
            '  "status": null or string,\n'
            '  "chapter": null or integer/string,\n'
            '  "state": null or string,\n'
            '  "date_or_year": null or string,\n'
            '  "attorney_name": null or string,\n'
            '  "debtor_name": null or string,\n'
            '  "match_code": null or string,\n'
            '  "client_name": null or string,\n'
            '  "city": null or string,\n'
            '  "trustee_name": null or string,\n'
            '  "trustee_city": null or string,\n'
            '  "sort_by_field": null or string,\n'
            '  "aggregation_type": null or string,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null or integer,\n'
            '  "sort_order": null or string,\n'
            '  "record_type": null or string or list of strings\n'
            "}\n"
            "Do not include markdown, code blocks, or explanation text outside the JSON."
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
        logger.warning("Entity extraction failed: %s", e)
        return {}

