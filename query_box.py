import hashlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from insights_generator import generate_insights
from utils.formatter import format_table
from query.suggestion import _generate_dynamic_prompt_suggestions
from query.handler import _handle_user_query
from utils.logger import logger


load_dotenv()


def render_query_box_tab(schema, active_schema):
    """Main rendering entrypoint for the Query Box tab in app.py"""
    
    # Initialize suggested question states
    if "suggested_question_selected" not in st.session_state:
        st.session_state.suggested_question_selected = None

    # -------------------------------------------------------------------------
    # DYNAMIC AUTO PROMPT SUGGESTIONS (TEXTUAL & VISUAL) via Bedrock Haiku
    # -------------------------------------------------------------------------
    # schema_to_use = active_schema if active_schema else schema

    schema_to_use = active_schema if active_schema else schema
    st.markdown(
        """
        <style>
        div:has(.suggest-anchor-textual) button,
        div:has(.suggest-anchor-visual) button,
        div:has(.suggest-anchor-textual) .stButton > button,
        div:has(.suggest-anchor-visual) .stButton > button,
        div[data-testid="column"]:has(.suggest-anchor-textual) button,
        div[data-testid="column"]:has(.suggest-anchor-visual) button {
            background-color: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            text-align: left !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            height: auto !important;
            min-height: 44px !important;
            display: inline-block !important;
        }

        div:has(.suggest-anchor-textual) button:hover,
        div:has(.suggest-anchor-visual) button:hover,
        div:has(.suggest-anchor-textual) .stButton > button:hover,
        div:has(.suggest-anchor-visual) .stButton > button:hover,
        div[data-testid="column"]:has(.suggest-anchor-textual) button:hover,
        div[data-testid="column"]:has(.suggest-anchor-visual) button:hover {
            background-color: #f8fafc !important;
            border-color: #3b82f6 !important;
            color: #2563eb !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.05) !important;
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # chat_placeholder = st.container()
    # input_placeholder = st.container()
    # suggestion_placeholder = st.container()

    # with input_placeholder:
    user_input = st.chat_input(
        "Query database (e.g. 'Show total records by state')..."
        )

    if st.session_state.get("suggested_question_selected"):
        user_input = st.session_state.suggested_question_selected
        st.session_state.suggested_question_selected = None

    if user_input:
        if not st.session_state.data_in_db:
            st.error("No active dataset found in database. Please upload a CSV first.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "user"
    ):
        last_msg = st.session_state.messages[-1]
        _handle_user_query(
                last_msg["content"],
                schema,
                active_schema,
            )

    # with chat_placeholder:

    chat_box = st.container(height=550)
            
    with chat_box:
                    for idx, msg in enumerate(st.session_state.messages):
                        avatar_val = "blank.png" if msg["role"] == "assistant" else None
                        with st.chat_message(msg["role"], avatar=avatar_val):
                            token_info_state_key = f"token_info_state_{idx}"
                            token_info_button_key = f"token_info_btn_{idx}"
                            if token_info_state_key not in st.session_state:
                                st.session_state[token_info_state_key] = False
                            show_token_usage = st.session_state[token_info_state_key]
            
                            content = msg.get("content")
                            if isinstance(content, dict):
                                if content.get("type") == "table":
                                    st.markdown(content.get("message", "Result:"))
                                    df_res = pd.DataFrame(content.get("data", []))
                                    st.dataframe(format_table(df_res), width="stretch", hide_index=True)
                                elif content.get("type") == "text":
                                    st.markdown(content.get("message", ""))
                                else:
                                    st.write(content)
                            else:
                                st.markdown(content)
            
                        if msg["role"] == "assistant":
                            # st.write(msg.get("has_insights"))
                            # st.write(msg.get("chart_type"))
                            if msg.get("dataframe") is not None:
                                st.markdown("### 📊 Analysis Results")
                                if msg.get("has_insights"):
                                    result_tab, chart_tab = st.tabs(
                                        ["📄 Data", "📈 Insights"]
                                    )
            
                                    with result_tab:
                                        st.caption(f"Returned {len(msg['dataframe'])} records")
                                        st.dataframe(
                                            format_table(msg["dataframe"]),
                                            width="stretch",
                                            hide_index=True
                                        )
            
                                    with chart_tab:
                                        st.caption("Automatically selected best visualization")
                                        generate_insights(
                                            msg["dataframe"],
                                            chart_type=msg.get("chart_type", "auto"),
                                            user_query=msg.get("user_query", "")
                                        )
                                else:
                                    st.dataframe(
                                        format_table(msg["dataframe"]),
                                        width="stretch",
                                        hide_index=True
                                    )
            
                            if msg.get("token_usage"):
                                with st.expander("⚙️ Model Execution Details", expanded=False):
                                    for step_name, usage in msg["token_usage"].items():
                                        if isinstance(usage, dict) and usage:
                                            st.markdown(
                                                f"""**{step_name.replace("_", " ").title()}**
                                                - Input Tokens : `{usage['input_tokens']}`
                                                - Output Tokens : `{usage['output_tokens']}`
                                                - Total Tokens : `{usage['total_tokens']}`
                                                """
                                                )

    # user_input = st.chat_input("Query database (e.g. 'Show total records by state')...")

    # if st.session_state.get("suggested_question_selected"):
    #     user_input = st.session_state.suggested_question_selected
    #     st.session_state.suggested_question_selected = None

    # if user_input:
    #     if not st.session_state.data_in_db:
    #         st.error("No active dataset found in database. Please upload a CSV first.")
    #     else:
    #         st.session_state.messages.append({"role": "user", "content": user_input})
            

    # if (
    # len(st.session_state.messages) > 0
    # and st.session_state.messages[-1]["role"] == "user"
    # ):
    #     last_msg = st.session_state.messages[-1]
    #     with st.spinner("Processing request..."):
    #         _handle_user_query(
    #             last_msg["content"],
    #             schema,
    #             active_schema,
    #             )
        
# ============================================================
# Suggestions
# ============================================================

    # Prepare suggestions based on conversation history and last result
    # Show follow-up suggestions only after the first user query
    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]

    if user_messages:
        history = st.session_state.get("conversation_memory", {}).get("history", [])
        sample_df = st.session_state.get("temp_table_dataframe")
        history_len = len(history) if history else 0
        active_temp_table = st.session_state.get("temp_table_name")
        last_msg_content = st.session_state.messages[-1]["content"]

        current_state_key = (
            f"sug_{history_len}_{active_temp_table}_"
            f"{st.session_state.get('last_uploaded_file_name')}_"
            f"{hashlib.sha256(str(last_msg_content).encode()).hexdigest()}"
        )

        if (
            "suggestions_state_key" not in st.session_state
            or st.session_state.suggestions_state_key != current_state_key
            or "current_suggestions" not in st.session_state
        ):
            suggestions = _generate_dynamic_prompt_suggestions(
                schema_to_use,
                st.session_state.get("conversation_memory"),
                last_result_df=sample_df,
            )
            st.session_state.current_suggestions = suggestions
            st.session_state.suggestions_state_key = current_state_key
        else:
            suggestions = st.session_state.current_suggestions

        logger.info(f"Suggestions received: {suggestions}")
        st.markdown('<div class="suggest-anchor-textual"></div>', unsafe_allow_html=True)
        st.caption("💬 Suggested Follow-up Questions")

        for idx, txt_q in enumerate(suggestions.get("textual", [])[:3]):
            if st.button(txt_q, key=f"suggest_text_{idx}", width="stretch"):
                st.session_state.suggested_question_selected = txt_q
                st.rerun()
       