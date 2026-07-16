import json
from utils.logger import logger
from llm.sql_generator import (column_mapper, sql_builder, sql_validator)
from llm.intent import intent_classifier
from query.sql_executor import execute_sqlite
from llm.entity_extractor import entity_extractor

def generate_sql_from_question(user_question, schema, conversation_memory=None, token_usage=None, main_schema=None):
    """Refactored pipeline to generate, validate, and execute SQL based on user question"""
    try:
        logger.info("Executing pipeline for user question: %s", user_question)
        
        # Step 1: Intent Classifier
        intent = intent_classifier(user_question, conversation_memory, token_usage)
        logger.info("Pipeline Step 1: Intent Classifier -> %s", intent)
        
        # Step 2: Entity Extractor
        entities = entity_extractor(user_question, intent, conversation_memory, token_usage)
        logger.info("Pipeline Step 2: Entity Extractor -> %s", json.dumps(entities))
        
        # Step 3: Column Mapper
        mapping = column_mapper(entities, schema, main_schema, token_usage)
        logger.info("Pipeline Step 3: Column Mapper -> %s", json.dumps(mapping))
        
        # Step 4: SQL Builder
        sql_query = sql_builder(user_question, intent, entities, mapping, schema, main_schema, conversation_memory, token_usage)
        logger.info("Pipeline Step 4: SQL Builder -> %s", sql_query)
        
        # Step 5: SQL Validator
        validation_result = sql_validator(sql_query, user_question, schema, main_schema, token_usage)
        logger.info("Pipeline Step 5: SQL Validator -> %s", json.dumps(validation_result))
        
        # If repaired_query is provided, use it
        final_query = sql_query
        if not validation_result.get("is_valid") and validation_result.get("repaired_query"):
            final_query = validation_result["repaired_query"]
            logger.info("Pipeline Step 5: Using repaired query -> %s", final_query)
        
        # Step 6: Execute SQLite
        result_df = None
        execution_error = None

        if final_query:
            try:
                result_df = execute_sqlite(final_query)

                logger.info(
                    "Pipeline Step 6: Execute SQLite -> Success, fetched %d rows",
                    len(result_df),
                )

            except Exception as e:
                logger.exception("Database query execution failed: %s", e)
                raise

        else:
            logger.warning("Pipeline Step 6: Execute SQLite skipped (Empty query)")

        return {
            "intent": intent,
            "entities": entities,
            "mapping": mapping,
            "sql_query": final_query,
            "validation_result": validation_result,
            "result_df": result_df,
            "execution_error": execution_error,
        }