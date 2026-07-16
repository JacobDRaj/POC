import streamlit as st
from llm.service import smart_query_understanding
from query.history import (is_followup_question, _append_conversation_memory)
from insights_generator import generate_insights
from query.history import drop_temporary_table, create_temporary_table_from_dataframe
from query.pipeline import generate_sql_from_question
from query.chart import detect_chart_type, is_pure_chart_request, is_chart_request
from utils.logger import logger 
from utils.cache import _make_query_cache_key
from utils.formatter import format_table
from query.chart import should_generate_insights


# ==============================================w===============================
# USER QUERY HANDLER & CHAT INTERACTION
# =============================================================================

def _handle_user_query(user_query, schema, active_schema):
    """Handle user queries by generating, validating, executing SQL and generating insights"""
    status = st.empty()
    status.info("🧠 Understanding your question...")
    try:
        logger.info("Processing user query: %s", user_query)

        # ── Smart Query Understanding (Haiku pre-processor) ────────────────────
        schema_for_understanding = active_schema if active_schema else schema
        understanding = smart_query_understanding(
            user_query,
            schema_for_understanding,
            st.session_state.get("conversation_memory"),
        )
        logger.info(
            "Query understanding | intent=%s | answerable=%s | normalized=%s",
            understanding["intent"],
            understanding["is_answerable"],
            understanding["normalized_query"][:120],
        )

        # If the query is deemed unanswerable from schema, surface clarification
        if not understanding["is_answerable"] and understanding["clarification_needed"]:
            clarification_msg = f"❓ {understanding['clarification_needed']}"
            _append_conversation_memory(user_query, "", None, answer=clarification_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": clarification_msg
            })

            return

        # Use the normalized query for all downstream processing
        effective_query = understanding["normalized_query"] if understanding["normalized_query"] else user_query
        # ── End Smart Query Understanding ──────────────────────────────────────

        # NOTE: keyword-based detection always uses original user_query
        # effective_query (schema-normalized) is used ONLY for SQL generation + validation
        is_followup = is_followup_question(user_query, st.session_state.conversation_memory) or (understanding.get("intent") == "follow_up")
        
        if is_followup and st.session_state.temp_table_schema:
            working_schema = st.session_state.temp_table_schema
            st.info(" **Querying previous result set**")
            logger.info("Using temporary table for follow-up query | table=%s", working_schema.get('table_name'))
        else:
            working_schema = active_schema if active_schema else schema
            if any(kw in user_query.lower() for kw in ['clear', 'reset', 'new query', 'fresh', 'different']):
                if st.session_state.temp_table_name:
                    drop_temporary_table(st.session_state.temp_table_name)
                    st.session_state.temp_table_name = None
                    st.session_state.temp_table_schema = None
                    st.session_state.temp_table_source_query = None
                    st.session_state.temp_table_dataframe = None
                    st.info(" Cleared previous result set. Starting fresh...")
        
        generation_token_usage = {}
        validation_token_usage = {}
        token_usage = {
            "generation": generation_token_usage,
            "validation": validation_token_usage,
        }

        cache_key = _make_query_cache_key(user_query, working_schema, st.session_state.conversation_memory)

        if cache_key in st.session_state.query_cache:
            cached_response = st.session_state.query_cache[cache_key]
            # st.success(" Cached result found for repeated query")
            # st.dataframe(format_table(cached_response["dataframe"]), width='stretch', hide_index=True)
            if cached_response.get("has_insights"):
                # with st.expander(" View Insights and Charts", expanded=True):
                #     generate_insights(
                #         cached_response["dataframe"],
                #         chart_type=cached_response.get("chart_type", "auto"),
                #         user_query=user_query,
                #     )
                _append_conversation_memory(
                user_query,
                cached_response["sql_query"],
                cached_response["records"],
                cached_response.get("validation_result"),
            )
                st.session_state.messages.append({
                "role": "assistant",
                "content": cached_response["message"],
                "sql_query": cached_response["sql_query"],
                "validation_result": cached_response.get("validation_result"),
                "dataframe": cached_response["dataframe"],
                "user_query": user_query,
                "has_insights": cached_response.get("has_insights", False),
                "chart_type": cached_response.get("chart_type", "auto"),
                "token_usage": cached_response.get("token_usage", {}),
                })
                return

        is_visualization_fallback = False
        sql_query = None
        final_query = None
        result_df = None
        validation_result = None
        
        
        main_db_schema = active_schema if active_schema else schema
        if is_followup and st.session_state.temp_table_dataframe is not None and is_pure_chart_request(user_query):
            logger.info("Pure chart request detected. Bypassing SQL generation and reusing active dataset.")
            final_query = st.session_state.temp_table_source_query
            result_df = st.session_state.temp_table_dataframe
            validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
            is_visualization_fallback = True
            sql_query = None

        if not is_visualization_fallback:
            with st.spinner(""):
                pipeline_result = generate_sql_from_question(
                    user_query,
                    working_schema,
                    st.session_state.conversation_memory,
                    token_usage=generation_token_usage,
                    main_schema=main_db_schema,
                )
            status.info("📝 Generating SQL...")
            sql_query = pipeline_result["sql_query"]
            validation_result = pipeline_result["validation_result"]
            result_df = pipeline_result["result_df"]

            if not sql_query and is_followup and st.session_state.temp_table_dataframe is not None and is_chart_request(user_query):
                logger.info("Empty SQL generated for chart follow-up query. Re-using active dataset.")
                final_query = st.session_state.temp_table_source_query
                result_df = st.session_state.temp_table_dataframe
                validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
                is_visualization_fallback = True

        if not is_visualization_fallback:
            if not sql_query:
                error_msg = " Failed to generate SQL query from your question. Please try rephrasing."
                # st.error(error_msg)
                _append_conversation_memory(user_query, "", None, answer=error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            final_query = sql_query
            
            if not validation_result['is_valid']:
                if validation_result.get('repaired_query'):
                    # st.warning(f" SQL validation indicated issues: {validation_result['explanation']}")
                    # st.info(" Using auto-repaired query...")
                    final_query = validation_result['repaired_query']
                else:
                    error_msg = f" Sorry !!  {validation_result['explanation']}"
                    # st.error(error_msg)
                    _append_conversation_memory(user_query, sql_query, None, validation_result, answer=error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sql_query": sql_query,
                        "validation_result": validation_result
                    })
                    return
            # else:
            #     st.success(" ")
            
            if result_df is None:
                error_msg = " Hey Could you please rephase your questio properly, I am unable to understand you currently . Thanks !!!"
                # st.error(error_msg)
                _append_conversation_memory(user_query, final_query, None, validation_result, answer=error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sql_query": final_query,
                    "validation_result": validation_result
                })
                return
        
        if len(result_df) == 0:
            msg = " Query executed successfully, but no records matched your criteria."
            # st.info(msg)
            _append_conversation_memory(user_query, final_query, [], validation_result, answer=msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "user_query": user_query,
                "has_insights": should_generate_insights(user_query, result_df),
                "chart_type": detect_chart_type(user_query),
                "token_usage": token_usage,
            })
            st.session_state.query_cache[cache_key] = {
                "message": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "records": [],
                "user_query": user_query,
                "has_insights": should_generate_insights(user_query, result_df),
                "chart_type": detect_chart_type(user_query),
                "token_usage": token_usage,
            }
            return
        
        success_msg = f" Generated chart for the active dataset." if is_visualization_fallback else f" Query results. "
        # st.success("")
        # st.dataframe(format_table(result_df), width='stretch', hide_index=True)
        if result_df is not None and not result_df.empty:
            old_temp_table = st.session_state.get("temp_table_name")
            temp_table_name, temp_schema = create_temporary_table_from_dataframe(result_df, final_query)
            if temp_table_name and temp_schema:
                if old_temp_table and old_temp_table != temp_table_name:
                    drop_temporary_table(old_temp_table)
                st.session_state.temp_table_name = temp_table_name
                st.session_state.temp_table_schema = temp_schema
                st.session_state.temp_table_source_query = final_query
                st.session_state.temp_table_dataframe = result_df
                # st.info(f" Result set stored as temporary table. You can ask follow-up questions about these {len(result_df)} records.")
        
        # Use original user_query for intent/keyword detection — effective_query may be schema-rewritten
        if result_df is None:
            raise RuntimeError("Result dataframe was not generated.")
        status.info("📊 Generating insights...")
        should_insights = should_generate_insights(user_query, result_df) or is_visualization_fallback
        # Only trust Haiku's 'visualization' intent if the user also used an
        # explicit chart keyword — prevents false positives on 'show me X' queries
        if not should_insights and understanding.get("intent") == "visualization":
            EXPLICIT_CHART_WORDS = {
                "plot", "chart", "graph", "visualize", "visualise",
                "pie", "bar", "donut", "scatter", "histogram",
                "heatmap", "line chart", "area chart",
            }
            if any(cw in user_query.lower() for cw in EXPLICIT_CHART_WORDS):
                should_insights = True
        chart_type = detect_chart_type(user_query)
        
        # if should_insights and (len(result_df) >= 2 or is_visualization_fallback):
        #     with st.expander(" View Insights and Charts", expanded=True):
        #         generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        _append_conversation_memory(effective_query, final_query, result_df.to_dict('records'), validation_result)
        st.session_state.messages.append({
            "role": "assistant",
            "content": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
        })
        st.session_state.query_cache[cache_key] = {
            "message": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "records": result_df.to_dict('records'),
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
        }
        
        logger.info("Query processed successfully | rows=%d | chart_type=%s | temp_table=%s", 
                    len(result_df), chart_type, st.session_state.temp_table_name or "none")
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception("Error in user query handling: %s", e)
        st.session_state.messages.append({
        "role": "assistant",
        "content": error_msg
    })

