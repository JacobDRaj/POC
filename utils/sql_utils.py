
import re
import json
from utils.logger import logger


def extract_sql_from_response(text: str) -> str:
    """Robust parser to extract SQL query string from JSON, markdown or raw text output."""
    try:
        if not text:
            return ""
        
        # 1. Try to extract value from "sql" or "CORRECTED_QUERY" key via regex (handles malformed/escaped JSON)
        for key in ["sql", "CORRECTED_QUERY"]:
            m_field = re.search(r'"' + key + r'"\s*:\s*"([\s\S]*?)"', text, re.I)
            if m_field:
                val = m_field.group(1).strip()
                # Unescape common escaped characters in JSON strings
                val = val.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
                if val.lower().strip().startswith("select"):
                    return val

        # 2. Try to find and parse a JSON block anywhere in the text
        m_json = re.search(r"(\{[\s\S]*\})", text)
        if m_json:
            json_str = m_json.group(1)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "sql" in data:
                    return data["sql"].strip()
                elif isinstance(data, dict) and "CORRECTED_QUERY" in data:
                    return data["CORRECTED_QUERY"].strip()
            except Exception:
                try:
                    cleaned_str = json_str.replace("\\'", "'")
                    data = json.loads(cleaned_str)
                    if isinstance(data, dict) and "sql" in data:
                        return data["sql"].strip()
                    elif isinstance(data, dict) and "CORRECTED_QUERY" in data:
                        return data["CORRECTED_QUERY"].strip()
                except Exception:
                    pass

        # 3. Try to extract from ```sql ... ``` block
        m_sql_block = re.search(r"```sql\s*([\s\S]*?)```", text, re.I)
        if m_sql_block:
            return m_sql_block.group(1).strip()

        # 4. Try to extract from standard ``` ... ``` block
        m_block = re.search(r"```\s*([\s\S]*?)```", text, re.I)
        if m_block:
            content = m_block.group(1).strip()
            # If the content contains "json" at the beginning, strip it and try parsing
            if content.lower().startswith("json"):
                try:
                    cleaned_json = content[4:].strip()
                    try:
                        data = json.loads(cleaned_json)
                        if isinstance(data, dict) and "sql" in data:
                            return data["sql"].strip()
                    except Exception:
                        cleaned_json_fixed = cleaned_json.replace("\\'", "'")
                        data = json.loads(cleaned_json_fixed)
                        if isinstance(data, dict) and "sql" in data:
                            return data["sql"].strip()
                except Exception:
                    pass
            # If it's just raw SQL inside the block (and not a JSON string), return it
            if "select" in content.lower() and not (content.strip().startswith("{") or '"sql"' in content.lower()):
                return content
            
        # 5. Regex fallback for raw SELECT statement
        m_select = re.search(r"\b(SELECT[\s\S]*?);?\s*$", text, re.I)
        if m_select:
            sql = m_select.group(1).strip()
            # Clean up trailing markdown/JSON characters if the LLM output was garbled
            sql = re.sub(r'["\'\}]+$', '', sql).strip()
            return sql

        return ""
    except Exception as e:
        logger.exception("Error extracting SQL: %s", e)
        return ""
