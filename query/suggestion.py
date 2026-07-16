
import re 
import json
from config import call_llm_haiku2,call_llm_haiku
from utils.logger import logger
from llm.prompt_manager import STARTER_QUESTION_PROMPT
from config import call_llm_with_cache

# def clean_and_validate_suggestions(data, schema):
#     """Post-process and validate suggestions to ensure they strictly conform to user rules."""
#     visual_kws = ["show", "draw", "display", "plot", "chart", "graph", "visualize", "visualise", "breakdown", "distribution", "trend", "trends", "view", "pie", "bar", "line", "donut", "histogram", "heatmap", "scatter", "area"]
    
#     textual = []
#     visual = []
    
#     # Extract raw suggestions
#     raw_textual = data.get("textual", []) if isinstance(data, dict) else []
#     raw_visual = data.get("visual", []) if isinstance(data, dict) else []
    
#     # 1. Process textual suggestions: remove visual words or drop them
#     for q in raw_textual:
#         q_str = str(q).strip()
#         if not q_str or len(q_str) < 5:
#             continue
#         q_lower = q_str.lower()
        
#         # Check if it has any visual keyword
#         has_visual_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in visual_kws)
#         if has_visual_kw:
#             # Try to rewrite it by replacing the visual action verb at the start
#             cleaned_q = q_str
#             cleaned_q = re.sub(r"^(show|display|draw|view|plot|visualize|visualise)\b", "", cleaned_q, flags=re.IGNORECASE).strip()
#             q_lower_new = cleaned_q.lower()
#             if any(re.search(rf"\b{kw}\b", q_lower_new) for kw in visual_kws):
#                 continue # discard it
#             else:
#                 # Rewrite leading question prefix
#                 if cleaned_q:
#                     cleaned_q = cleaned_q[0].upper() + cleaned_q[1:]
#                     if not cleaned_q.startswith(("What", "How", "List", "Filter", "Find", "Identify", "Calculate", "Compare")):
#                         cleaned_q = "What is the " + cleaned_q.lower()
#                     if not cleaned_q.endswith("?"):
#                         cleaned_q += "?"
#                     textual.append(cleaned_q)
#         else:
#             textual.append(q_str)
            
#     # 2. Process visual suggestions: ensure they contain at least one visual keyword
#     for q in raw_visual:
#         q_str = str(q).strip()
#         if not q_str or len(q_str) < 5:
#             continue
#         q_lower = q_str.lower()
        
#         has_visual_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in visual_kws)
#         if not has_visual_kw:
#             # Prepend a default chart verb
#             q_str = "Show " + q_str[0].lower() + q_str[1:]
#             q_lower = q_str.lower()

#         # Enforce bar chart for any comparison between 2 (e.g. status vs status, chapter vs chapter, X vs Y)
#         is_comparison = False
#         if any(kw in q_lower for kw in [" vs ", " versus ", "compare", "comparing", "comparison"]):
#             is_comparison = True
#         elif any(p[0] in q_lower and p[1] in q_lower for p in [
#             ("active", "closed"),
#             ("chapter 7", "13"),
#             ("chapter 7", "chapter 13"),
#             ("new", "closed"),
#             ("individual", "business"),
#             ("asset", "no-asset"),
#             ("asset", "no asset"),
#         ]):
#             is_comparison = True
            
#         if is_comparison:
#             # If it already has "bar" or "barchart" in it, it's fine.
#             if "bar" not in q_lower and "barchart" not in q_lower:
#                 # Replace other chart types with Bar chart
#                 other_charts = ["pie chart", "pie", "donut chart", "donut", "line chart", "line", "scatter plot", "scatter", "histogram", "heatmap", "area chart", "area", "chart", "graph", "plot"]
#                 replaced = False
#                 for ct in other_charts:
#                     pattern = rf"\b{ct}\b"
#                     if re.search(pattern, q_lower):
#                         q_str = re.sub(pattern, "Bar chart", q_str, flags=re.IGNORECASE)
#                         replaced = True
#                         break
#                 if not replaced:
#                     if q_str.lower().startswith("show "):
#                         q_str = "Bar chart showing " + q_str[5:]
#                     elif q_str.lower().startswith("compare "):
#                         q_str = "Bar chart comparing " + q_str[8:]
#                     else:
#                         q_str = "Bar chart of " + q_str[0].lower() + q_str[1:]
                        
#         visual.append(q_str)
        
#     # Deduplicate
#     textual = list(dict.fromkeys(textual))
#     visual = list(dict.fromkeys(visual))
    
#     # 3. Formulate fallback lists for padding
#     fallback_textual = [
#         "Filter cases by Chapter 7 filings",
#         "Identify the top 10 states by case volume",
#         "List the most recent 10 records in the dataset",
#         "Calculate the average match score for all records",
#         "Find records where state is NY",
#         "How many active cases are registered?",
#         "Compare count of active vs closed status cases"
#     ]
#     # Filter fallbacks just in case
#     fallback_textual = [f for f in fallback_textual if not any(re.search(rf"\b{kw}\b", f.lower()) for kw in visual_kws)]
    
#     fallback_visual = [
#         "Pie chart of chapter breakdown",
#         "Bar chart of cases by state",
#         "Line chart of filings over time",
#         "Horizontal bar chart of top 10 attorneys",
#         "Donut chart of case status",
#         "Histogram of match scores"
#     ]
    
#     # Pad textual to 4
#     for item in fallback_textual:
#         if len(textual) >= 4:
#             break
#         if item not in textual:
#             textual.append(item)
            
#     # Pad visual to 4
#     for item in fallback_visual:
#         if len(visual) >= 4:
#             break
#         if item not in visual:
#             visual.append(item)
            
#     return {
#         "textual": textual[:4],
#         "visual": visual[:4]
#     }


def clean_and_validate_suggestions(data, schema):
    """
    Clean and validate LLM-generated follow-up suggestions.
    Do NOT rewrite, downgrade or replace high-quality LLM suggestions.
    Only:
      - Remove empty values
      - Remove duplicates
      - Trim overly long text
      - Ensure minimum suggestions
    """

    raw_textual = data.get("textual", []) if isinstance(data, dict) else []
    raw_visual = data.get("visual", []) if isinstance(data, dict) else []

    textual = []
    visual = []

    # ----------------------------
    # Clean textual suggestions
    # ----------------------------
    for q in raw_textual:

        if not q:
            continue

        q = str(q).strip()

        if len(q) < 8:
            continue

        # Remove newlines
        q = re.sub(r"\s+", " ", q)

        # Remove trailing period
        q = q.rstrip(".")

        # Limit length for UI buttons
        if len(q) > 120:
            q = q[:117] + "..."

        textual.append(q)

    # ----------------------------
    # Clean visual suggestions
    # ----------------------------
    for q in raw_visual:

        if not q:
            continue

        q = str(q).strip()

        if len(q) < 8:
            continue

        q = re.sub(r"\s+", " ", q)

        q = q.rstrip(".")

        if len(q) > 120:
            q = q[:117] + "..."

        visual.append(q)

    # ----------------------------
    # Remove duplicates
    # ----------------------------
    textual = list(dict.fromkeys(textual))
    visual = list(dict.fromkeys(visual))

    # ----------------------------
    # Intelligent fallback ONLY if
    # LLM completely failed
    # ----------------------------
    if not textual:

        textual = [
            "Which business segments contribute most to overall results?",
            "What trend or anomaly deserves further investigation?",
            "Which categories drive the highest business impact?",
            "How does performance vary across key dimensions?"
        ]

    if not visual:

        visual = [
            "Compare business metrics across categories",
            "Analyze trends over time",
            "Visualize distribution across major groups",
            "Identify top contributors visually"
        ]

    return {
        "textual": textual[:3],
        "visual": visual[:1]
    }


def generate_dataset_starters(schema, sample_df):
    """Generate dataset-specific starter questions."""

    prompt = STARTER_QUESTION_PROMPT.format(
        schema=schema,
        sample_data=sample_df.head(5).to_string(index=False)
    )

    response = call_llm_with_cache(
        prompt,
        temperature=0
    )

    try:
        result = json.loads(response)
        return result.get("starter_questions", [])
    except Exception:
        return []


def _generate_dynamic_prompt_suggestions(schema, conversation_memory, last_result_df=None):
    """Generate dynamic textual and visual suggestions based on conversation history
    and the columns/data of the latest result DataFrame."""
    try:
        # Build result context if we have a recent DataFrame
        result_context = ""
        if last_result_df is not None and not last_result_df.empty:
            num_cols = last_result_df.select_dtypes(include='number').columns.tolist()
            cat_cols = [c for c in last_result_df.columns if c not in num_cols]
            sample_rows = last_result_df.head(10).to_dict('records')
            summary = last_result_df.describe(include="all").fillna("").to_string()
            null_counts = last_result_df.isnull().sum().to_dict()
            unique_counts = {
                c: last_result_df[c].nunique()
                for c in last_result_df.columns
                }
            
            result_context = (
    f"\nLAST QUERY RESULT SUMMARY:\n"
    f"- Rows returned: {len(last_result_df)}\n"
    f"- Numeric columns: {num_cols or 'none'}\n"
    f"- Categorical columns: {cat_cols or 'none'}\n"
    f"- Unique values: {unique_counts}\n"
    f"- Missing values: {null_counts}\n"
    f"- Dataset statistics:\n{summary[:5000]}\n"
    f"- Sample rows: {sample_rows}\n"
)

        # Format memory
        history_lines = []
        last_question_context = ""
        if conversation_memory and conversation_memory.get("history"):
            history = conversation_memory["history"]
            recent = history[-3:]
            for entry in recent:
                history_lines.append(f"User: {entry.get('user_question')}")
                if entry.get('sql_query'):
                    history_lines.append(f"SQL: {entry.get('sql_query')}")
                if entry.get('record_count') is not None:
                    history_lines.append(f"Result count: {entry.get('record_count')} rows")
            
            # Extract previous question for context
            last_q = history[-1].get("user_question")
            if last_q:
                last_question_context = (
                    f"CURRENT USER QUERY / LAST USER QUESTION: \"{last_q}\"\n"
                    f"CRITICAL REQUIREMENT:\n"
                    f"The textual suggestions MUST be a few logical follow-up questions based on the current query ('{last_q}'). "
                    f"For example, if the current query is about states and chapters, follow up on those specific states or chapters "
                    f"(e.g., 'What is the percentage of Chapter 7 in the top state?' or 'Compare Chapter 13 cases across those states'). "
                    f"Make the suggested questions feel like a natural continuation of the user's analytical journey, drilling deeper "
                    f"into the categories, filters, or time periods from the current query and its results.\n\n"
                )
        
        if history_lines:
            history_str = "Recent Conversation:\n" + "\n".join(history_lines)
        else:
            history_str = "Recent Conversation:\n(No queries asked yet. This is the start of the user's journey.)"

        # Build journey instruction before creating the prompt
        
        if history_lines:
            journey_instruction = (
        "USER JOURNEY:\n"
        "- This is an ongoing conversation.\n"
        "- Generate contextual follow-up questions that naturally extend the previous analysis.\n"
        "- Build upon the latest findings instead of starting a new analysis.\n\n"
    )
        else:
            journey_instruction = (
        "USER JOURNEY:\n"
        "- This is the first interaction.\n"
        "- Generate exploratory analytical questions instead of follow-up questions.\n"
        "- Help the user discover meaningful insights from the available data.\n"
        "- Do NOT assume any previous analysis, findings, filters, or results.\n\n"
                                )
       
        prompt = (
            "You are a Senior Business Intelligence Copilot specialized in SQL Analytics, Business Intelligence, Executive Reporting and Data Exploration."
            "Your role is to help users progress naturally from data exploration to business insight and decision-making.\n"
            "Generate recommendations that a financial analyst, operations manager, or executive would find valuable.\n\n"

            "OBJECTIVE:\n"
            "Generate intelligent, business-focused follow-up questions and visualization suggestions "
            "based on:\n"
            "1. Database schema\n"
            "2. Current query result dataset\n"
            "3. Conversation history\n"
            "4. Previously explored insights\n\n"


            f"DATABASE SCHEMA:\n{json.dumps(schema, indent=2)[:200000]}\n\n"
            f"{history_str}\n"
            f"{result_context}\n\n"
            "IMPORTANT:\n"
            "- Suggestions MUST use ONLY columns and values available in the latest query result.\n"
            "- Never invent business entities.\n"
            "- If the current result contains only two columns, generate follow-up questions using only those columns.\n"
            "- Continue drilling into the current result instead of suggesting unrelated analysis.\n\n"
            f"{last_question_context}"
            f"{journey_instruction}"
            

            "FIRST INTERACTION RULES:\n"
            "- If this is the first interaction, generate exploratory analytical questions instead of follow-up questions.\n"
            "- Do not assume previous analysis, rankings, trends, filters, or insights.\n"
            "- Recommend broad business and data exploration questions using the available schema.\n"
            "- Suggest high-level visualizations that provide an overview of the dataset.\n"
            "- Avoid drill-down, comparison, or root-cause analysis during the first interaction.\n\n"

            "SUPPORTED VISUALIZATION TYPES:\n"
            "- Bar Chart\n"
            "- Horizontal Bar Chart\n"
            "- Pie Chart\n"
            "- Donut Chart\n"
            "- Line Chart\n"
            "- Area Chart\n"
            "- Scatter Plot\n"
            "- Histogram\n"
            "- Heatmap\n\n"

            "QUESTION GENERATION RULES:\n"

            "A. FOLLOW-UP QUESTION RULES\n"
            "- Generate EXACTLY 3 follow-up questions.\n"
            "- Every question MUST be directly answerable using the current dataset.\n"
            "- Generate questions that naturally produce SQL.\n"
            "- Use actual column names and values from the latest result whenever possible.\n"
            "- Continue the user's current investigation instead of starting a new topic.\n"
            "- Questions should require GROUP BY, aggregation, ranking, filtering, joins, percentages, trends or comparisons.\n"
            "- Ask questions that a Senior BI Consultant, Data Analyst, Finance Analyst or Executive would ask.\n"
            "- Never generate beginner questions.\n"
            "- Never generate generic questions like:\n"
            "    • Top 10\n"
            "    • Show all records\n"
            "    • List records\n"
            "    • Filter Chapter 7\n"
            "    • Average value\n"
            "    • Recent records\n"
            "- Prefer:\n"
            "    • Root cause analysis\n"
            "    • KPI comparison\n"
            "    • Trend analysis\n"
            "    • Segmentation\n"
            "    • Correlation\n"
            "    • Business impact\n"
            "    • Operational performance\n"
            "    • Risk analysis\n"
            "    • Outlier detection\n"
            "- Never explain the question.\n"
            "- Maximum 20 words.\n"
            "- Return ONLY the question.\n\n"

            "B. VISUALIZATION SUGGESTIONS\n"
            "- Generate EXACTLY 4 visualization suggestions.\n"
            "- Every visualization must provide a unique business insight.\n"
            "- Never recommend multiple charts using the same dimension and measure combination.\n"
            "- Avoid duplicate topics across visualization suggestions.\n"
            "- Recommend charts only when supported by the current query result.\n"
            "- If this is the first interaction, recommend overview charts rather than drill-down visualizations.\n"
            "- Exclude NULL, blank, or empty values from every visualization.\n"
            "- Recommend only charts that can be fully populated without empty bars, slices, or categories.\n"
            "- Prioritize accuracy, readability, and business value over chart variety.\n\n"

            "CONTEXT AWARENESS:\n"
            "- Continue the user's analytical journey instead of starting a new one.\n"
            "- Prioritize insights that explain why the current result occurred.\n"
            "- Suggest the next logical business question.\n"
            "- Use only values present in the latest dataset.\n"
            "- Never repeat the previous user question.\n"
            "- Never recommend repeating the same analysis.\n"
            "- Every recommendation must extend the current investigation.\n"
            "- Prioritize business decisions over SQL exploration.\n"
            "- Assume the audience is an experienced business analyst.\n"

            "ANALYTICAL CONTEXT:\n"
            "- Do not restart the analysis.\n"
            "- Continue from the user's previous finding.\n"
            "- Assume the user wants to investigate deeper.\n"
            "- Every suggestion should build upon the last insight.\n\n"

            "FOLLOW-UP STRATEGY:\n"
            "Question 1: Drill deeper into the current metric.\n"
            "Question 2: Compare business dimensions using the current result.\n"
            "Question 3: Recommend the next executive-level KPI investigation.\n\n"


            "OUTPUT FORMAT:\n"
            "Return ONLY valid JSON.\n"
            "Do NOT return markdown, explanations, notes, comments, or additional text.\n\n"

            "{\n"
            '  "textual": [\n'
            '    "Question 1",\n'
            '    "Question 2",\n'
            '    "Question 3",\n'
            '    "Question 4"\n'
            "  ],\n"
            '  "visual": [\n'
            '    "Visual 1",\n'
            '    "Visual 2",\n'
            '    "Visual 3",\n'
            '    "Visual 4"\n'
            "  ]\n"
            "}"
        )



        response = call_llm_haiku2(prompt)
        
        logger.info("Generated suggestions via Haiku2: %s", response)

        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\n", "", cleaned_response)
                cleaned_response = re.sub(r"\n```$", "", cleaned_response)

            data = json.loads(cleaned_response)
            if isinstance(data, dict) and "textual" in data and "visual" in data:
                logger.info(f"LLM Suggestions JSON: {data}")
                return clean_and_validate_suggestions(data, schema)
        except Exception as ex:
            # logger.warning("Failed to parse Haiku JSON response: %s. Using regex fallback.", ex)
            logger.exception("Failed to parse Haiku JSON")
            logger.info(f"Raw LLM Response:\n{response}")

        # Fallback regex parsing
        textual, visual = [], []
        matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)
        for m in matches:
            if m not in ["textual", "visual"] and len(m) > 8:
                m_lower = m.lower()
                if any(kw in m_lower for kw in ["chart", "plot", "graph", "visualize", "bar", "pie", "line", "donut", "scatter", "histogram", "heatmap", "area"]):
                    if len(visual) < 4:
                        visual.append(m)
                else:
                    if len(textual) < 4:
                        textual.append(m)

        if len(textual) >= 1 or len(visual) >= 1:
            return clean_and_validate_suggestions({"textual": textual, "visual": visual}, schema)

    except Exception as e:
        logger.exception("Error generating dynamic suggestions: %s", e)

    # Fallback suggestions generated dynamically from the schema if the LLM call fails
    columns_set = {col.get("name").lower() for col in schema.get("columns", [])} if schema else set()
    fallback_textual = []
    fallback_visual = []

    if "state" in columns_set:
        fallback_textual.append("Top 10 states with the most filings")
        fallback_visual.append("Bar chart of filings by state")
    if "status" in columns_set:
        fallback_textual.append("What is active vs closed case breakdown?")
    if "chapter" in columns_set:
        fallback_visual.append("Pie chart of chapter distribution")
    if "open_date" in columns_set or "date_filed" in columns_set:
        fallback_visual.append("Line chart of filings by year")
    if "attorney_last_name" in columns_set or "attorney_first_name" in columns_set or "attorney_dba" in columns_set:
        fallback_visual.append("Horizontal bar chart of top 10 attorneys")

    # Fill in generic ones to meet target counts
    if len(fallback_textual) < 4:
        fallback_textual.append("How many total bankruptcy filings are there?")
    if len(fallback_textual) < 4:
        fallback_textual.append("List the most recent 10 records")
    if len(fallback_textual) < 4:
        fallback_textual.append("Filter cases by Chapter 7")
    if len(fallback_textual) < 4:
        fallback_textual.append("Identify the top 10 states by volume")

    if len(fallback_visual) < 4:
        fallback_visual.append("Pie chart of chapter distribution")
    if len(fallback_visual) < 4:
        fallback_visual.append("Bar chart of filings by state")
    if len(fallback_visual) < 4:
        fallback_visual.append("Line chart of filings by year")
    if len(fallback_visual) < 4:
        fallback_visual.append("Horizontal bar chart of top 10 states")

    return clean_and_validate_suggestions({"textual": [], "visual": []}, schema)
