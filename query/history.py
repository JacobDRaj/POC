import logging
import sqlite3
import streamlit as st
from query.chart import is_chart_request

logger = logging.getLogger("bankruptcy_genbi")

# =============================================================================
# CONVERSATION MEMORY FUNCTIONS
# =============================================================================




def _initialize_conversation_memory():
    """Initialize conversation memory structure"""
    return {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
    }


def _append_conversation_memory(user_query, sql_query, records, validation_result=None, answer=None):
    """Append structured conversation memory"""
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory

        if not answer:
            memory_entry = {
                "user_question": user_query,
                "sql_query": sql_query,
                "records": records,
                "record_count": len(records) if records else 0,
            }
        else:
            memory_entry = {
                "user_question": user_query,
                "assistant_answer": answer,
            }

        if validation_result:
            memory_entry["validation_result"] = validation_result.get("status", "UNKNOWN")

        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query

        if not answer:
            memory["last_assistant_response"] = f"SQL: {sql_query} | Records: {len(records) if records else 0}"
        else:
            memory["last_assistant_response"] = answer

        # Keep only last 10 conversations
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]

    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)



# =============================================================================
# TEMPORARY TABLE MANAGEMENT FOR FOLLOW-UP QUESTIONS
# =============================================================================

def create_temporary_table_from_dataframe(result_df, source_query):
    """Create a temporary SQLite table from a result dataframe for follow-up queries.
    
    Returns a tuple of (table_name, schema_dict) for use in follow-up queries.
    """
    try:
        import time
        table_name = f"temp_result_{int(time.time() * 1000) % 1000000}"
        
        # Connect and create temporary table
        conn = sqlite3.connect('data.db')
        result_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        
        # Build schema for this temporary table
        temp_schema = {
            "table_name": table_name,
            "columns": [
                {
                    "name": col,
                    "type": str(result_df[col].dtype),
                    "description": f"Column from previous query result"
                }
                for col in result_df.columns
            ]
        }
        
        logger.info("Created temporary table | name=%s | rows=%d | columns=%d", 
                    table_name, len(result_df), len(result_df.columns))
        
        return table_name, temp_schema
    except Exception as e:
        logger.exception("Failed to create temporary table: %s", e)
        return None, None


def drop_temporary_table(table_name):
    """Drop a temporary table from the database."""
    if not table_name:
        return True
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        conn.close()
        logger.info("Dropped temporary table | name=%s", table_name)
        return True
    except Exception as e:
        logger.exception("Failed to drop temporary table: %s", e)
        return False



def is_followup_question(user_query, conversation_history):
    """Detect if the user is asking a follow-up question.

    A query is a follow-up ONLY if it refers back to the active result set via
    pronouns / reference words, OR is a pure chart request with no new
    entity/column/filter that would require a fresh main-DB query.
    """
    if not st.session_state.temp_table_name:
        return False

    query_lower = user_query.lower()

    # --- Standard reference-word follow-ups ---
    followup_keywords = [
        'that', 'those', 'these', 'records', 'results', 'rows', 'items',
        'from that', 'from there', 'among those', 'above', 'convert', 'previous',
    ]
    if any(kw in query_lower for kw in followup_keywords):
        return True

    # --- Chart request: only treat as follow-up if it is a *pure* re-chart ---
    # i.e. "plot this", "pie chart", "bar chart" WITHOUT naming a new entity
    # (year, state, attorney, chapter, district, debtor, filing, etc.) that
    # would require going back to the main database.
    if is_chart_request(user_query):
        NEW_ENTITY_HINTS = [
            'year', 'month', 'quarter', 'state', 'district', 'chapter',
            'attorney', 'debtor', 'filing', 'case', 'judge', 'trustee',
            'top', 'bottom', 'by state', 'by year', 'by chapter', 'by district',
            'by attorney', 'over time', 'trend',
        ]
        if any(hint in query_lower for hint in NEW_ENTITY_HINTS):
            # Needs a fresh query against the main DB — NOT a follow-up
            return False
        # No new entity hints → pure re-chart of the active result set
        return True

    return False