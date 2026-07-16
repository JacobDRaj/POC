import re
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from config import call_llm_with_cache, count_token_usage
from utils.logger import logger
from utils.sql_utils import extract_sql_from_response



def execute_sql_query(sql_query):
    """Run SQLite read query against database"""

    conn = None

    try:
        sql_query = clean_sql_for_whitespace(sql_query)
        logger.info("Executing SQLite SQL: %s", sql_query)

        conn = sqlite3.connect("data.db")
        df = pd.read_sql_query(sql_query, conn)

        logger.info("Executed query successfully | returned %d rows", len(df))

        return {
            "success": True,
            "data": df,
            "error": None,
            "sql": sql_query,
        }

    except Exception as e:
        logger.exception("Database query execution failed: %s", e)

        return {
            "success": False,
            "data": None,
            "error": str(e),
            "sql": sql_query,
        }

    finally:
        if conn:
            conn.close()


def clean_sql_for_whitespace(sql_query):
    """Add TRIM() to WHERE clauses to handle whitespace in data"""
    try:
        pattern = r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'"])([^\2]*?)\2'

        def replace_with_trim(match):
            col_name = match.group(1)
            quote = match.group(2)
            value = match.group(3)
            return f"WHERE TRIM({col_name}) = {quote}{value}{quote}"

        modified_query = re.sub(pattern, replace_with_trim, sql_query, flags=re.IGNORECASE)
        if modified_query != sql_query:
            logger.info("Added TRIM() to WHERE clause for whitespace handling")
        return modified_query
    except Exception as e:
        logger.exception("Error cleaning SQL for whitespace: %s", e)
        return sql_query


def validate_sql_with_judge(sql_query, user_question, schema, token_usage=None, main_schema=None):
    """Use an LLM as a judge to validate if the generated SQL is correct"""
    sql_upper = sql_query.strip().upper()

    ddl_dml = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'RENAME', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'REPLACE']
    for keyword in ddl_dml:
        if keyword in sql_upper:
            return {
                "is_valid": False,
                "status": "BLOCKED",
                "explanation": f"Write operation keyword '{keyword}' is forbidden. Only read SELECT operations are allowed.",
                "repaired_query": None,
            }

    if not sql_upper.startswith('SELECT'):
        return {
            "is_valid": False,
            "status": "INVALID",
            "explanation": "Query must begin with a SELECT keyword.",
            "repaired_query": None,
        }

    try:
        if main_schema and schema and schema.get('table_name') != main_schema.get('table_name'):
            table_desc = f"Main table: {main_schema['table_name']} or Temporary table: {schema['table_name']}"
            columns_desc = (
                f"Columns in Main table `{main_schema['table_name']}`:\n" +
                "\n".join([f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}" for col in main_schema['columns']]) +
                f"\n\nColumns in Temporary table `{schema['table_name']}`:\n" +
                "\n".join([f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}" for col in schema['columns']])
            )
            rule_5 = f"5. Table name must be either '{main_schema['table_name']}' or '{schema['table_name']}'"
        else:
            table_desc = f"Table: {schema['table_name']}"
            columns_desc = "\n".join([
                f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}"
                for col in schema['columns']
            ])
            rule_5 = f"5. Table name must be '{schema['table_name']}'"

        prompt = f"""You are an expert SQLite3 database validator. Your task is to judge whether the following SQL query is correct and valid.

DATABASE SCHEMA:
{table_desc}
Columns:
{columns_desc}

USER QUESTION: {user_question}

SQL QUERY TO VALIDATE:
{sql_query}

VALIDATION RULES:
1. Check if the SQL syntax is valid SQLite3
2. Check if column names are EXACT matches from the schema — same spelling AND same casing (e.g., 'status' not 'Status', 'active_status' not 'Active_Status'). If a column name does not exist in the schema, the query is INVALID.
3. Check if the query answers the user's question using the CORRECT column:
   - 'active vs closed' or 'case status' should use the 'status' column, NOT 'record_type' or 'active_status'
   - 'new vs closed', 'closed vs new', 'new cases', 'reopened', 'record stage' (New/Closed/Reopened/Update) should use 'record_type'. CRITICAL: 'New' is exclusively a record_type value, NEVER a status value. If the question mentions 'new' in the context of cases/records, it MUST use record_type, NOT status.
   - 'active/inactive flag' should use 'active_status'
4. Check if the query limits the results to only the specified comparison categories if the user asked for a breakdown/comparison between specific categories (e.g., if the user asked for 'active vs closed breakdown', the SQL must filter status to 'Active' and 'Closed' using a WHERE/HAVING clause like TRIM(status) IN ('Active', 'Closed')). If the user did NOT request filtering by specific categories, the SQL should NOT include such filters.
5. Check if the user's question asks for analysis, counts, or grouping 'by year' or 'yearly'. If so, the query must extract the year using strftime('%Y', date_column) or substr(date_column, 1, 4), name it `year`, include it in the SELECT list, and group by it. If the query fails to group by year when requested, it is INVALID.
6. Check if the user's question asks for a distribution, breakdown, ratio, or comparison of one column (e.g. chapter) by/for/each another column (e.g. state) (e.g., 'chapter distribution for each state'). If so, the query MUST select and group by BOTH columns (e.g. selecting both state and chapter, grouping by both state and chapter), not just one. If the query only selects or groups by one column, it is INVALID and you MUST repair it to select both columns in the SELECT clause and group by both in the GROUP BY clause.
7. Check if the date formats are correct (YYYY-MM-DD) if dates are involved.
{rule_5}
9. Do not include ';' at the end of the query
10. Reject any queries that attempt to modify data
11. Check if the user's question specifies a specific year (e.g., '2024' or '2023'). If so, the SQL query MUST contain a WHERE clause filtering by that year (e.g., strftime('%Y', date_filed) = '2024' or substr(date_filed, 1, 4) = '2024'). If the query fails to filter by the specified year and instead retrieves all years, the query is INVALID and you MUST repair it. Furthermore, if the user asks for a distribution/breakdown within a single year (e.g., '2024 distribution'), it should be grouped by month (e.g. strftime('%m', date_filed) as month) to show the distribution across the year.
12. If the user's question requests sorting or limiting (e.g. 'show me top 3 risk cases', 'highest score cases'), the query MUST order by `match_score` descending (or ascending if lowest is asked) and apply `LIMIT N`. If the query fails to sort by `match_score` or fails to apply the limit, it is INVALID and you MUST repair it.

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT ..."}}

Do not include any explanations, text, or additional content outside the JSON object."""

        response = call_llm_with_cache(prompt)
        if token_usage is not None:
            token_usage.update(count_token_usage(prompt, response))
        logger.info("SQL Validation Response:\n%s", response)

        is_valid = False
        explanation = "Validation response received"
        query = None

        response_text = response.strip()

        if 'YES' in response_text.upper():
            explanation = "Query is valid"
            is_valid = True
        else:
            explanation = "Query is invalid — auto-repaired"
            query = extract_sql_from_response(response_text)

        return {
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID",
            "explanation": explanation,
            "repaired_query": query if not is_valid else None,
        }
    except Exception as e:
        logger.exception("Error validating SQL with judge: %s", e)
        return {
            "is_valid": False,
            "status": "ERROR",
            "explanation": f"Validation error: {str(e)}",
            "repaired_query": None
        }



def execute_sqlite(sql_query: str) -> pd.DataFrame:
    """Stage 6: Execute SQLite - Execute query against database and return pandas DataFrame."""
    return execute_sql_query(sql_query)
