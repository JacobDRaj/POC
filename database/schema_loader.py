
import json
import os
import sqlite3
import pandas as pd
from utils.logger import logger
from utils.dataframe import clean_column_name

def load_schema():
    """Load default schema from schema.json"""
    try:
        with open("schema.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None



def load_sample_data_knowledge(sample_csv_path: str = "sample_data.csv", n_rows: int = 20) -> str:
    """
    Read sample_data.csv using pd.read_csv and convert it to a compact CSV string
    that can be injected into LLM prompts as a data knowledge base.

    If the CSV file is missing or empty, falls back to querying the live data.db
    for the first n_rows rows and also refreshes the CSV file on disk.

    Returns a string of up to n_rows rows in CSV format, or an empty string on failure.
    """
    try:
        # Attempt to read from CSV file
        if os.path.exists(sample_csv_path) and os.path.getsize(sample_csv_path) > 0:
            df = pd.read_csv(sample_csv_path, nrows=n_rows)
            if not df.empty:
                logger.info("Loaded sample knowledge from %s (%d rows)", sample_csv_path, len(df))
                return df.to_csv(index=False)

        # Fallback: pull from live database
        logger.warning("sample_data.csv missing or empty — loading from data.db")
        try:
            table_name = get_db_table_name()
        except Exception:
            table_name = "uploaded_data"
        conn = sqlite3.connect("data.db")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {n_rows}", conn)
        conn.close()

        if not df.empty:
            # Persist to CSV so future calls are faster
            df.to_csv(sample_csv_path, index=False)
            logger.info("Refreshed %s from data.db (%d rows)", sample_csv_path, len(df))
            return df.to_csv(index=False)

    except Exception as e:
        logger.warning("load_sample_data_knowledge failed: %s", e)

    return ""



def get_db_table_name():
    """Detect active table name in SQLite database"""
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='uploaded_data' OR name='bankruptcy4')")
        res = cursor.fetchall()
        conn.close()
        if res:
            names = [r[0] for r in res]
            if "uploaded_data" in names:
                return "uploaded_data"
            return names[0]
        return "uploaded_data"
    except Exception:
        return "uploaded_data"


def get_actual_database_schema():
    """Dynamically construct schema based on the current active table in data.db"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()
        
        schema = {
            "table_name": table_name,
            "columns": []
        }
        
        descriptions = {}
        try:
            with open("schema.json", "r") as f:
                static_schema = json.load(f)
                descriptions = {col["name"].lower(): col.get("description", "") for col in static_schema.get("columns", [])}
        except Exception:
            pass
            
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            clean_name = clean_column_name(col_name)
            schema["columns"].append({
                "name": col_name,
                "type": col_type,
                "description": descriptions.get(clean_name, f"Database field: {col_name}")
            })
        return schema
    except Exception as e:
        logger.error(f"Error getting dynamic database schema: {e}")
        return None
