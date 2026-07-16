
import pandas as pd
from utils.logger import logger
import streamlit as st

def clean_column_name(name):
    """Normalize column names to standard lowercase snake_case and resolve common typos."""
    if not name or not isinstance(name, str):
        return name
    n = name.strip().lower().replace(" ", "_")
    # Clean typos / spelling inconsistencies
    n = n.replace("attorny", "attorney")   # Attorny_* → attorney_*
    n = n.replace("lastt_name", "last_name")  # _lastt_Name → _last_name
    n = n.replace("lastt", "last")           # any remaining _lastt_ variants
    n = n.replace("addl_1", "address_line_1")
    n = n.replace("addl_2", "address_line_2")
    n = n.replace("ac_no", "account_number")
    # Map bare 'client' column header to DB column 'client_name'
    if n == "client":
        n = "client_name"
    n = n.replace("creditor_time", "creditor_meeting_time")
    n = n.replace("notification_no", "notification_number")
    n = n.replace("case_no", "case_number")
    return n


def load_and_clean_csv(uploaded_file):
    """Read CSV, strip whitespace from columns, clean rows and map columns to clean schema"""
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_dataframe = df
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != '']
        df.columns = [clean_column_name(col) for col in df.columns]
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        logger.info("Loaded and cleaned CSV | shape=(%d, %d)", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.exception("Error processing uploaded CSV: %s", e)
        return None

