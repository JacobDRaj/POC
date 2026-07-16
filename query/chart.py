
from pandas import DataFrame



def is_chart_request(user_query):
    """Detect if the user is asking for a chart or visualization."""
    chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'pie', 'piechart', 'bar', 'barchart', 'line', 'trend', 'donut']
    query_lower = user_query.lower()
    return any(kw in query_lower for kw in chart_keywords)


def is_pure_chart_request(user_query):
    """Detect if the query is purely asking to chart the active result set (no new filters/sorting)."""
    query_lower = user_query.lower().strip()
    
    if not is_chart_request(user_query):
        return False
        
    filter_indicators = [
        'greater', 'less', 'more than', 'fewer than', 'above', 'below', 
        'equal', 'limit', 'top', 'bottom', 'where', 'having', 'filter',
        'sort', 'order by', 'group by', 'count greater', 'count less', 
        '>', '<', '=', '>=', '<='
    ]
    
    has_filter = False
    for ind in filter_indicators:
        if ind in query_lower:
            if ind == 'above':
                if not any(x + ' above' in query_lower for x in ['data', 'results', 'table', 'items', 'rows', 'query', 'shown']):
                    has_filter = True
            else:
                has_filter = True
                
    if has_filter:
        return False
        
    return True

def detect_chart_type(user_query):
    """Detect appropriate visualization type from user query keywords"""
    q = user_query.lower()
    # Most specific matches first
    if any(k in q for k in ["donut", "doughnut"]):
        return "donut"
    if any(k in q for k in ["pie", "ratio", "proportion", "breakdown"]):
        return "pie"
    if any(k in q for k in ["scatter", "correlation"]):
        return "scatter"
    if any(k in q for k in ["heatmap", "heat map", "matrix"]):
        return "heatmap"
    if any(k in q for k in ["histogram", "frequency"]):
        return "histogram"
    if any(k in q for k in ["horizontal bar", "ranked", "ranking"]):
        return "horizontal_bar"
    if any(k in q for k in ["area chart", "area plot", "filled"]):
        return "area"
    if any(k in q for k in ["line chart", "line plot", "trend line", "over time", "trend", "growth", "year", "month", "yearly", "monthly"]):
        return "line"
    if any(k in q for k in ["bar chart", "bar plot", "bar graph", "bar", "column", "compare"]):
        return "bar"
    return "auto"



def should_generate_insights(user_query, result_df):
    """
    Determine whether a visualization should be shown.

    Rules:
    1. Always require a valid dataframe.
    2. Always show charts when the user explicitly requests one.
    3. Automatically show charts when the SQL result is suitable for visualization.
    """

    if result_df is None or result_df.empty or len(result_df) < 2:
        return False

    q_lower = user_query.lower()

    CHART_KEYWORDS = {
        "plot", "chart", "graph",
        "visualize", "visualise", "visualization",
        "pie", "bar", "line", "donut",
        "scatter", "histogram", "heatmap",
        "dashboard",
        "compare", "comparison", "vs", "versus",
        "trend", "distribution", "breakdown"
    }

    # --------------------------------------------------
    # Explicit chart request
    # --------------------------------------------------
    if any(k in q_lower for k in CHART_KEYWORDS):
        return True

    # --------------------------------------------------
    # Automatically visualize grouped SQL results
    # --------------------------------------------------
    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in result_df.columns
        if c not in numeric_cols
    ]

    # Typical SQL GROUP BY result
    if len(result_df.columns) == 2:
        if len(numeric_cols) == 1 and len(categorical_cols) == 1:
            return True

    # Multi-dimensional aggregation
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        return True

    return False