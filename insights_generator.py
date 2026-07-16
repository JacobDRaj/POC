import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
import plotly.express as px
from config import call_llm, call_llm_with_cache
logger = logging.getLogger("insights_generator")

#  Light professional chart theme (matches app UI) 
PRIMARY   = "#2563EB"   # electric blue
ACCENT2   = "#7C3AED"   # violet
ACCENT3   = "#10B981"   # emerald
BG_DARK   = "#FFFFFF"
BG_PANEL  = "#a6d1f7"
TEXT_MAIN = "#1E293B"
TEXT_MUTED= "#64748B"
GRID_LINE = "#E2E8F0"

CHART_COLORS = [
    "#2563EB", "#7C3AED", "#10B981", "#F59E0B",
    "#EF4444", "#06B6D4", "#EC4899", "#84CC16",
    "#F97316", "#8B5CF6"
]

import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_DARK,
    "axes.edgecolor":    GRID_LINE,
    "axes.labelcolor":   TEXT_MAIN,
    "axes.titlecolor":   TEXT_MAIN,
    "axes.grid":         True,
    "axes.prop_cycle":   mpl.cycler(color=CHART_COLORS),
    "grid.color":        GRID_LINE,
    "grid.linewidth":    0.7,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  BG_DARK,
    "legend.edgecolor":  GRID_LINE,
    "legend.labelcolor": TEXT_MAIN,
    "figure.dpi":        110,
})
sns.set_style("whitegrid", {
    "axes.facecolor":  BG_DARK,
    "figure.facecolor": BG_DARK,
    "grid.color":      GRID_LINE,
})


def show_centered_plot(fig, use_container_width=True, clear_figure=True, ratio=0.7):
    """Render a Matplotlib figure centered in the Streamlit interface"""
    side_ratio = (1.0 - ratio) / 2.0
    c1, c2, c3 = st.columns([side_ratio, ratio, side_ratio])
    with c2:
        st.pyplot(fig, use_container_width=True, clear_figure=clear_figure)


class DataInsightsGenerator:
    """Generate comprehensive insights from query results"""
    
    def __init__(self, result_df: pd.DataFrame):
        self.df = result_df
        raw_numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in self.df.columns:
            if col in raw_numeric_cols:
                # Distinguish between value/measure columns and identifier/low-cardinality integer categories (e.g. Chapter 7/11/13, Year 2024)
                is_value_col = any(x in str(col).lower() for x in ['count', 'total', 'frequency', 'value', 'sum', 'amount', 'pct', 'ratio'])
                is_low_cardinality_int = (
                    pd.api.types.is_integer_dtype(self.df[col]) and 
                    self.df[col].nunique() <= 20 and
                    not is_value_col
                )
                if is_low_cardinality_int and len(raw_numeric_cols) > 1:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
                
        self.date_cols = self._detect_date_columns()
    
    def _detect_date_columns(self) -> List[str]:
        """Detect columns that represent dates"""
        date_cols = []
        for col in self.df.columns:
            if any(token in str(col).lower() for token in ["date", "time", "opened", "filed", "conversion", "status_date"]):
                try:
                    pd.to_datetime(self.df[col], errors="coerce")
                    if pd.to_datetime(self.df[col], errors="coerce").notna().sum() / len(self.df) >= 0.5:
                        date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset"""
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum()
        }
        return stats
    
    def get_numeric_insights(self) -> Dict:
        """Generate detailed insights for numeric columns"""
        insights = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            insights[col] = {
                "count": len(col_data),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "skewness": col_data.skew(),
                "has_outliers": self._detect_outliers(col_data),
                "outlier_count": self._count_outliers(col_data)
            }
        
        return insights
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Detect if a series has outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count number of outliers in a series"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def get_categorical_insights(self) -> Dict:
        """Generate detailed insights for categorical columns"""
        insights = {}
        
        for col in self.categorical_cols:
            col_data = self.df[col].astype(str).value_counts(dropna=True)
            
            insights[col] = {
                "unique_count": self.df[col].nunique(),
                "most_common": col_data.index[0] if len(col_data) > 0 else None,
                "most_common_freq": int(col_data.iloc[0]) if len(col_data) > 0 else 0,
                "least_common": col_data.index[-1] if len(col_data) > 0 else None,
                "least_common_freq": int(col_data.iloc[-1]) if len(col_data) > 0 else 0,
                "top_categories": col_data.head().to_dict(),
                "missing_count": self.df[col].isnull().sum()
            }
        
        return insights
    
    def detect_trends(self) -> Optional[Tuple[str, str]]:
        """Detect if data has date and numeric columns for trend analysis"""
        if self.date_cols and self.numeric_cols:
            return self.date_cols[0], self.numeric_cols[0]
        return None
    
    def get_correlations(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix for numeric columns"""
        if len(self.numeric_cols) >= 2:
            return self.df[self.numeric_cols].corr()
        return None
    
    def get_missing_data_report(self) -> Dict:
        """Generate report on missing data"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        report = {}
        for col in self.df.columns:
            if missing[col] > 0:
                report[col] = {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
        
        return report


class InsightVisualizer:
    """Handle visualization of insights with smart recommendations"""
    
    def __init__(self, result_df: pd.DataFrame, insights_gen: DataInsightsGenerator):
        self.df = result_df
        self.insights = insights_gen

    def _aggregate_by_category(self, cat_col: str, num_col: str) -> pd.Series:
        """Helper to aggregate a numeric column by a category using sum or mean depending on the column name."""
        num_lower = str(num_col).lower()
        if any(x in num_lower for x in ["avg", "mean", "score", "pct", "ratio", "percentage"]):
            return self.df.groupby(cat_col)[num_col].mean()
        return self.df.groupby(cat_col)[num_col].sum()
    
    def _is_year_column(self, col_name: str, values: pd.Series) -> bool:
        """Detect whether a categorical column likely represents years"""
        col_lower = str(col_name).lower()
        if "year" in col_lower:
            return True

        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().all() and numeric_values.between(1900, 2100).all():
            return True

        parsed_dates = pd.to_datetime(values, errors="coerce", format="%Y")
        if parsed_dates.notna().all():
            return True

        return False

    def _sort_bar_data_by_category(self, bar_data: pd.DataFrame, cat_col: str, value_col: str) -> pd.DataFrame:
        """Sort chart categories chronologically when they represent years"""
        if self._is_year_column(cat_col, bar_data[cat_col]):
            bar_data = bar_data.copy()
            bar_data["_sort_key"] = pd.to_numeric(bar_data[cat_col], errors="coerce")
            if bar_data["_sort_key"].notna().all():
                return bar_data.sort_values("_sort_key").drop(columns=["_sort_key"])

        return bar_data.sort_values(value_col, ascending=True)

    def _build_llm_summary_prompt(
        self,
        stats: Dict,
        numeric_insights: Dict,
        categorical_insights: Dict,
    ) -> str:
        """Build a prompt for the LLM summarizing dataset trends and drivers."""
        prompt_lines = [
            "Role: You are an elite strategic credit risk advisor and business intelligence analyst.",
            "Task: Generate a highly polished, client-focused 'Trend Summary Analysis' based on the dataset characteristics provided below.",
            "",
            "CRITICAL FORMATTING RULES (follow exactly):",
            "1. Tone: Professional, executive-level, clear, and highly authoritative. Speak directly to business leaders/clients.",
            "2. Use the exact markdown structure shown below — do NOT merge sections into a single paragraph.",
            "3. Each section MUST start on its own new line with a markdown header (## or ###).",
            "4. The 'Key Drivers & Concentrations' section MUST be a bullet list using '- ' prefix for each item.",
            "5. Use **bold** to emphasize key numbers, percentages, and names.",
            "6. Do NOT write prose paragraphs where bullet lists are required.",
            "7. Keep total output under 160 words. No introductory or concluding meta-text.",
            "",
            "REQUIRED OUTPUT STRUCTURE (use this exact format):",
            "## Executive Trend Overview",
            "<1-2 sentence summary of overall volume and distribution trajectory>",
            "",
            "## Key Drivers & Concentrations",
            "- <Driver 1: e.g., top states/territories with % share>",
            "- <Driver 2: e.g., dominant bankruptcy chapter with % share>",
            "- <Driver 3: e.g., primary risk profile or debtor type>",
            "",
            "## Strategic Client Implications",
            "<1-2 sentences on portfolio management, resource allocation, or operational risk>",
            "",
            "Tone: Professional, executive-level, authoritative. Speak directly to business leaders.",
            "Focus: Translate statistics into strategic business insights. Avoid column names, SQL, or data-quality language.",
            "",
            "Dataset Metadata & Aggregated Statistics:",
        ]

        if self.insights.date_cols:
            prompt_lines.append(f"- Temporal/Date Columns detected: {', '.join(self.insights.date_cols)}")

        if self.insights.numeric_cols:
            prompt_lines.append(f"- Numeric Columns detected: {', '.join(self.insights.numeric_cols)}")
            prompt_lines.append("  Summary stats:")
            for col, col_stats in list(numeric_insights.items())[:3]:
                prompt_lines.append(
                    f"    * {col}: Average={round(col_stats['mean'], 2)}, Median={round(col_stats['median'], 2)}, "
                    f"Min={round(col_stats['min'], 2)}, Max={round(col_stats['max'], 2)}, Outliers count={col_stats['outlier_count']}"
                )

        if self.insights.categorical_cols:
            prompt_lines.append(f"- Categorical Columns detected: {', '.join(self.insights.categorical_cols)}")
            prompt_lines.append("  Top distributions:")
            for col, cat_stats in list(categorical_insights.items())[:3]:
                top_categories = cat_stats.get('top_categories', {})
                category_summary = ', '.join([
                    f"{k} (count: {v})" for k, v in list(top_categories.items())[:3]
                ])
                prompt_lines.append(
                    f"    * {col}: Unique count={cat_stats['unique_count']}, Top occurrences={category_summary}"
                )

        prompt_lines.append("")
        prompt_lines.append("Generate the Trend Summary Analysis now, adhering strictly to the Tone & Style Guidelines. Do not include any introductory or concluding meta-text.")

        return "\n".join(prompt_lines)

    def _generate_llm_summary(self, stats: Dict) -> Optional[str]:
        """Generate an LLM-based executive summary of dataset insights."""
        try:
            numeric_insights = self.insights.get_numeric_insights()
            categorical_insights = self.insights.get_categorical_insights()
            prompt = self._build_llm_summary_prompt(stats, numeric_insights, categorical_insights)
            llm_response = call_llm_with_cache(prompt, temperature=0.1)
            return llm_response.strip() if llm_response else None
        except Exception as e:
            logger.warning("LLM summary generation failed: %s", e)
            return None

    def render_executive_summary(self):
        """Render executive summary with key metrics"""
        stats = self.insights.generate_summary_statistics()

        # st.subheader(" Data Summary")

        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Total Records", stats["total_rows"])
        # with col2:
        #     st.metric("Columns", stats["total_columns"])
        # with col3:
        #     st.metric("Duplicates", stats["duplicate_rows"])
        # with col4:
        #     st.metric("Memory (MB)", stats["memory_usage_mb"])

        llm_summary = self._generate_llm_summary(stats)
        if llm_summary:
            import re
            # Scale heading font sizes to 60% of Streamlit defaults:
            # ## (h2) default ~1.75rem → 1.05rem | ### (h3) default ~1.25rem → 0.75rem
            scaled = re.sub(
                r'^## (.+)$',
                r'<h2 style="font-size:1.05rem;font-weight:700;margin:10px 0 4px 0;">\1</h2>',
                llm_summary, flags=re.MULTILINE
            )
            scaled = re.sub(
                r'^### (.+)$',
                r'<h3 style="font-size:0.75rem;font-weight:600;margin:6px 0 4px 0;">\1</h3>',
                scaled, flags=re.MULTILINE
            )
            st.markdown(scaled, unsafe_allow_html=True)

        # Missing data report
        missing_report = self.insights.get_missing_data_report()
        # if missing_report:
        #     with st.expander(" Missing Data Details", expanded=False):
        #         missing_df = pd.DataFrame([
        #             {"Column": col, "Missing Count": data["count"], "% Missing": data["percentage"]}
        #             for col, data in missing_report.items()
        #         ])
        #         st.dataframe(missing_df, use_container_width=True)
    
    def render_numeric_analysis(self):
        """Render comprehensive numeric column analysis"""
        numeric_insights = self.insights.get_numeric_insights()
        
        if not numeric_insights:
            return
        
        # st.subheader(" Numeric Columns Analysis")
        
        # Summary statistics table
        # with st.expander(" Summary Statistics", expanded=True):
        #     stats_data = []
        #     for col, stats in numeric_insights.items():
        #         stats_data.append({
        #             "Column": col,
        #             "Count": stats["count"],
        #             "Mean": round(stats["mean"], 2),
        #             "Median": round(stats["median"], 2),
        #             "Std Dev": round(stats["std"], 2),
        #             "Min": round(stats["min"], 2),
        #             "Max": round(stats["max"], 2),
        #             "Outliers": stats["outlier_count"]
        #         })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Visualizations for key numeric columns
        key_numeric = self.insights.numeric_cols[:3]  # Focus on top 3 numeric columns
        
        if len(key_numeric) > 0:
            st.subheader("Numeric Distributions (Bar Plot)")

            num_charts = len(key_numeric)
            # Use full-width single column for 1 chart, else split into columns
            use_full_width = (num_charts == 1)
            if use_full_width:
                chart_containers = [st.container()]
            else:
                chart_containers = st.columns(min(3, num_charts))
            
            for idx, col in enumerate(key_numeric):
                col_data = self.df[col].dropna()
                if len(col_data) < 2:
                    continue

                # ── Adaptive figsize ──────────────────────────────────────────
                is_aggregated = (
                    len(self.insights.categorical_cols) >= 1
                    and len(self.insights.numeric_cols) == 1
                )
                if is_aggregated:
                    cat_col = self.insights.categorical_cols[0]
                    agg_series = self._aggregate_by_category(cat_col, col)
                    bar_data = pd.DataFrame({cat_col: agg_series.index, col: agg_series.values})
                    n_bars = len(bar_data)
                else:
                    n_bars = 15  # histogram bins

                if use_full_width:
                    # Full-width: wide & tall enough to breathe
                    fig_w = 10
                    fig_h = max(4.0, min(6.0, 3.0 + n_bars * 0.12))
                else:
                    # Multi-column: compact but scale height with bar count
                    fig_w = 5.0
                    fig_h = max(3.2, min(5.5, 2.8 + n_bars * 0.10))

                # Font scale: smaller when many bars, larger when few
                label_fs  = max(6, min(9,  10 - n_bars // 6))
                tick_fs   = max(6, min(8,   9 - n_bars // 8))
                val_fs    = max(5, min(8,   9 - n_bars // 7))
                # ────────────────────────────────────────────────────────────

                ctx = chart_containers[0] if use_full_width else chart_containers[idx % 3]

                with ctx:
                    if is_aggregated:
                        if not self.insights.categorical_cols:
                            return 
                        cat_col = self.insights.categorical_cols[0]    
                        bar_data = self._aggregate_by_category(cat_col, col)

                        bar_data = pd.DataFrame({
                            cat_col: bar_data.index,
                            col: bar_data.values
                        })

                        bar_data = self._sort_bar_data_by_category(bar_data, cat_col, col)

                        fig = px.bar(
                            bar_data,
                            x=cat_col,
                            y=col,
                            text=col,
                            color=col,
                            color_continuous_scale="Blues"
                        )

                        fig.update_traces(
                            texttemplate="%{text:,.0f}",
                            textposition="outside",
                            marker_line_width=1,
                            marker_line_color="white",
                            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y}}<extra></extra>"
                        )

                        fig.update_layout(
                            title={
                                "text": f"{col} by {cat_col}",
                                "x": 0.5
                            },
                            template="plotly_white",
                            height=520,
                            xaxis_title=cat_col,
                            yaxis_title=col,
                            coloraxis_showscale=False,
                            margin=dict(l=20, r=20, t=60, b=20)
                        )

                        st.plotly_chart(fig, width="stretch")
                    
                    # fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    
                    # if is_aggregated:
                    #     bar_data = self._sort_bar_data_by_category(bar_data, cat_col, col)
                        
                    #     bars = ax.bar(range(len(bar_data)), bar_data[col].values,
                    #                  color=CHART_COLORS[0], alpha=0.95, edgecolor=BG_PANEL, linewidth=1.2)
                        
                    #     ax.set_xticks(range(len(bar_data)))
                    #     rotation = 45 if n_bars > 6 else 0
                    #     ax.set_xticklabels(bar_data[cat_col].values, rotation=rotation,
                    #                        ha='right' if rotation else 'center', fontsize=tick_fs)
                        
                    #     for bar in bars:
                    #         height = bar.get_height()
                    #         ax.text(bar.get_x() + bar.get_width() / 2., height,
                    #                 f'{int(height)}',
                    #                 ha='center', va='bottom', fontsize=val_fs,
                    #                 fontweight='bold', color=TEXT_MAIN)
                        
                    #     ax.set_title(f"{col} by {cat_col}", fontsize=11 if use_full_width else 10,
                    #                  fontweight="bold")
                    #     ax.set_xlabel(cat_col, fontsize=label_fs, fontweight='bold')
                    #     ax.set_ylabel(col, fontsize=label_fs, fontweight='bold')
                    # else:
                    #     counts, bins = np.histogram(col_data, bins=15)
                        
                    #     bars = ax.bar(range(len(counts)), counts, color=CHART_COLORS[0],
                    #                  alpha=0.99, edgecolor=BG_PANEL, linewidth=1.2)
                        
                    #     for bar in bars:
                    #         height = bar.get_height()
                    #         ax.text(bar.get_x() + bar.get_width() / 2., height,
                    #                 f'{int(height)}',
                    #                 ha='center', va='bottom', fontsize=val_fs,
                    #                 fontweight='bold', color=TEXT_MAIN)
                        
                    #     ax.set_title(f"{col} Distribution", fontsize=11 if use_full_width else 10,
                    #                  fontweight="bold")
                    #     ax.set_xlabel(f"{col} Value Ranges", fontsize=label_fs, fontweight='bold')
                    #     ax.set_ylabel("Frequency (Count)", fontsize=label_fs, fontweight='bold')
                    
                    # ax.grid(axis="y", alpha=0.3, linestyle="--")
                    # ax.set_axisbelow(True)
                    # plt.tight_layout()
                    # show_centered_plot(fig, use_container_width=True, clear_figure=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis for numeric columns"""
        if len(self.insights.numeric_cols) < 2:
            return
        
        corr_matrix = self.insights.get_correlations()
        
        if corr_matrix is None or corr_matrix.empty:
            return
        
        st.subheader(" Correlation Analysis")
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                    strong_corr.append({
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Correlation": round(corr_val, 3)
                    })
        
        # Display correlations
        if strong_corr:
            with st.expander(" Strong Correlations (|r| > 0.5)", expanded=True):
                corr_df = pd.DataFrame(strong_corr).sort_values("Correlation", key=abs, ascending=True)
                st.dataframe(corr_df, use_container_width=True)
        
        # Heatmap
        with st.expander("Correlation Heatmap", expanded=False):
            fig = px.imshow(corr_matrix,color_continuous_scale="RdBu_r",aspect="auto"
                            )
            fig.update_layout(
                title="Numeric Variables Correlation Matrix",
                template="plotly_white",
                height=500,
                coloraxis_colorbar_title="Correlation"
                            )
            st.plotly_chart(fig, width="stretch")

    
    def render_trend_analysis(self):
        """Render trend analysis if date and numeric columns exist"""
        trend_info = self.insights.detect_trends()
        
        if not trend_info:
            return
        
        date_col, numeric_col = trend_info
        
        st.subheader("Trend Analysis")
        
        try:
            trend_df = self.df[[date_col, numeric_col]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.dropna(subset=[date_col, numeric_col])
            trend_df = trend_df.sort_values(date_col)
            
            if len(trend_df) >= 2:
                fig = px.line(
                        trend_df,
                        x=date_col,
                        y=numeric_col,
                        markers=True,
                        title=f"{numeric_col} Over Time ({date_col})"
                    )

                fig.update_traces(
                        line=dict(width=3),
                        marker=dict(size=8)
                    )

                fig.update_layout(
                        template="plotly_white",
                        height=450,
                        xaxis_title=date_col,
                        yaxis_title=numeric_col,
                        title_x=0.5
                    )

                st.plotly_chart(fig, width="stretch")

        except Exception as e:
            logger.warning(f"Could not render trend analysis: {e}")
    
    # def render_categorical_analysis(self):
    #     """Render categorical column analysis with pie charts only"""
    #     cat_insights = self.insights.get_categorical_insights()
        
    #     if not cat_insights:
    #         return
        
    #     # st.subheader(" Categorical Columns Analysis")
        
    #     # Focus on key categorical columns with reasonable cardinality
    #     key_categorical = [col for col in self.insights.categorical_cols 
    #                       if cat_insights[col]["unique_count"] <= 20][:3]
        
    #     # Fallback if no low-cardinality categorical columns exist, but there are categorical columns
    #     if not key_categorical and self.insights.categorical_cols:
    #         key_categorical = self.insights.categorical_cols[:1]
        
    #     if key_categorical:
    #         st.subheader(" Category Distributions (Pie Charts)")

    #         num_pies = len(key_categorical)
    #         use_full_width = (num_pies == 1)
    #         if use_full_width:
    #             pie_containers = [st.container()]
    #         else:
    #             pie_containers = st.columns(min(3, num_pies))
            
    #         for idx, col in enumerate(key_categorical):
    #             # Resolve value column
    #             value_col = None
    #             if len(self.insights.numeric_cols) == 1 and len(self.insights.categorical_cols) <= 2:
    #                 value_col = self.insights.numeric_cols[0]
    #             if value_col is None:
    #                 possible_value_cols = [c for c in self.insights.numeric_cols 
    #                                      if any(x in c.lower() for x in ['count', 'total', 'frequency', 'value', 'sum'])]
    #                 if possible_value_cols:
    #                     value_col = possible_value_cols[0]
                
    #             if value_col:
    #                 pie_data = self._aggregate_by_category(col, value_col).sort_values(ascending=False)
    #             else:
    #                 pie_data = self.df[col].astype(str).value_counts()
                
    #             # Group high-cardinality items (if > 10 categories) into "Other"
    #             if len(pie_data) > 10:
    #                 top_n = pie_data.head(9)
    #                 other_sum = pie_data.iloc[9:].sum()
    #                 pie_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
                
    #             if len(pie_data) == 0:
    #                 continue

    #             # ── Adaptive figsize & font scale by slice count ─────────────
    #             n_slices = len(pie_data)
    #             if use_full_width:
    #                 fig_w = fig_h = 0.5 * max(6.0, min(9.0, 5.0 + n_slices * 0.2))
    #             else:
    #                 fig_w = fig_h = 0.5 * max(4.2, min(6.5, 3.8 + n_slices * 0.18))

    #             label_fs  = max(5, min(8, 9 - n_slices // 3))
    #             pct_fs    = max(5, min(7, 8 - n_slices // 4))
    #             title_fs  = 10 if use_full_width else 8
    #             # ─────────────────────────────────────────────────────────────

    #             ctx = pie_containers[0] if use_full_width else pie_containers[idx % 3]
    #             with ctx:
    #                 fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    #                 colors = CHART_COLORS[:n_slices]

    #                 # For many slices suppress labels on tiny wedges to avoid overlap
    #                 display_labels = pie_data.index.tolist()
    #                 if n_slices > 8:
    #                     threshold = pie_data.sum() * 0.03
    #                     display_labels = [
    #                         lbl if val >= threshold else ''
    #                         for lbl, val in zip(pie_data.index, pie_data.values)
    #                     ]

    #                 total = pie_data.sum()

    #                 # Custom autopct: show count on line 1, pct on line 2
    #                 def make_autopct(values):
    #                     def autopct(pct):
    #                         count = int(round(pct / 100.0 * total))
    #                         return f"{count:,}\n({pct:.1f}%)"
    #                     return autopct

    #                 wedges, texts, autotexts = ax.pie(
    #                     pie_data.values,
    #                     labels=display_labels,
    #                     autopct=make_autopct(pie_data.values),
    #                     startangle=90,
    #                     colors=colors,
    #                     textprops={'fontsize': label_fs},
    #                     pctdistance=0.78,
    #                 )
    #                 for autotext in autotexts:
    #                     autotext.set_color('white')
    #                     autotext.set_fontweight('bold')
    #                     autotext.set_fontsize(pct_fs)

    #                 title = (
    #                     f"{col} Distribution (by {value_col})"
    #                     if value_col else f"{col} Distribution"
    #                 )
    #                 ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=12)
    #                 ax.axis("equal")

    #                 # Legend with count + pct for every slice
    #                 legend_labels = [
    #                     f"{lbl}:  {int(val):,}  ({val / total * 100:.1f}%)"
    #                     for lbl, val in zip(pie_data.index, pie_data.values)
    #                 ]
    #                 ax.legend(
    #                     wedges, legend_labels,
    #                     title="Category", title_fontsize=max(5, pct_fs - 1),
    #                     loc="lower center",
    #                         bbox_to_anchor=(0.5, -0.25),
    #                     ncol=min(3, n_slices),
    #                     fontsize=max(4, pct_fs - 2),
    #                     frameon=True,
    #                     framealpha=0.85,
    #                     edgecolor=GRID_LINE,
    #                 )
    #                 plt.tight_layout()
    #                 if use_full_width:
    #                     show_centered_plot(fig, use_container_width=True, clear_figure=True, ratio=0.45)
    #                 else:
    #                     st.pyplot(fig, use_container_width=False, clear_figure=True)
    
    def render_categorical_analysis(self):
        """Render categorical analysis using interactive Plotly pie charts."""

        cat_insights = self.insights.get_categorical_insights()

        if not cat_insights:
            return

        key_categorical = [
            col for col in self.insights.categorical_cols
            if cat_insights[col]["unique_count"] <= 20
        ][:3]

        if not key_categorical and self.insights.categorical_cols:
            key_categorical = self.insights.categorical_cols[:1]

        if not key_categorical:
            return

        st.subheader("Category Distribution")

        cols = st.columns(len(key_categorical))

        for idx, col in enumerate(key_categorical):

            value_col = None

            if len(self.insights.numeric_cols) == 1:
                value_col = self.insights.numeric_cols[0]

            if value_col:
                pie_data = (
                    self.df.groupby(col)[value_col]
                    .sum()
                    .sort_values(ascending=False)
                )
            else:
                pie_data = self.df[col].astype(str).value_counts()

            if len(pie_data) > 10:
                top = pie_data.head(9)
                other = pie_data.iloc[9:].sum()
                pie_data = pd.concat([top, pd.Series({"Other": other})])

            plot_df = pd.DataFrame({
                col: pie_data.index,
                "Value": pie_data.values
            })

            with cols[idx]:

                fig = px.pie(
                    plot_df,
                    names=col,
                    values="Value",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}<extra></extra>"
                )

                fig.update_layout(
                    template="plotly_white",
                    title=f"{col} Distribution",
                    title_x=0.5,
                    height=450,
                    showlegend=True
                )

                st.plotly_chart(fig, width="stretch")

    def render_line_chart(self):
        """Render interactive line chart."""
        if not self.insights.numeric_cols:
            return

        num_col = self.insights.numeric_cols[0]
        cat_col = self.insights.categorical_cols[0] if self.insights.categorical_cols else None

        if cat_col:
            agg = self._aggregate_by_category(cat_col, num_col)

            plot_df = pd.DataFrame({
                cat_col: agg.index,
                num_col: agg.values
            })

            plot_df = self._sort_bar_data_by_category(plot_df, cat_col, num_col)

            fig = px.line(
                plot_df,
                x=cat_col,
                y=num_col,
                markers=True,
                text=num_col
            )

        else:

            plot_df = self.df.copy()

            fig = px.line(
                plot_df,
                x=plot_df.index,
                y=num_col,
                markers=True,
                text=num_col
            )

        fig.update_traces(
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=9)
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{num_col} Trend",
            title_x=0.5,
            height=500,
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    def render_grouped_bar(self):
            
        """Render grouped bar chart using Plotly."""

        if len(self.insights.categorical_cols) < 2 or not self.insights.numeric_cols:
            self.render_numeric_analysis()
            return

        c1 = self.insights.categorical_cols[0]
        c2 = self.insights.categorical_cols[1]
        val = self.insights.numeric_cols[0]

        top_c1 = self.df[c1].value_counts().head(15).index

        plot_df = self.df[self.df[c1].isin(top_c1)].copy()

        fig = px.bar(
            plot_df,
            x=c1,
            y=val,
            color=c2,
            barmode="group",
            text=val,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{val} by {c1} and {c2}",
            title_x=0.5,
            height=550,
            xaxis_title=c1,
            yaxis_title=val,
            legend_title=c2
        )

        st.plotly_chart(fig, width="stretch")

    def render_horizontal_bar(self):
        """Render interactive horizontal bar chart."""

        if not self.insights.numeric_cols or not self.insights.categorical_cols:
            self.render_numeric_analysis()
            return

        num_col = self.insights.numeric_cols[0]
        cat_col = self.insights.categorical_cols[0]

        agg = self._aggregate_by_category(cat_col, num_col).sort_values()

        bar_df = pd.DataFrame({
            cat_col: agg.index,
            num_col: agg.values
        })

        fig = px.bar(
            bar_df,
            x=num_col,
            y=cat_col,
            orientation="h",
            text=num_col,
            color=num_col,
            color_continuous_scale="Viridis"
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=f"<b>%{{y}}</b><br>{num_col}: %{{x}}<extra></extra>"
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{num_col} by {cat_col}",
            title_x=0.5,
            height=max(450, len(bar_df) * 35),
            xaxis_title=num_col,
            yaxis_title=cat_col,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, width="stretch")

    def render_scatter(self):
        """Render interactive scatter plot."""

        if len(self.insights.numeric_cols) < 2:
            st.info("Scatter plot requires at least 2 numeric columns.")
            self.render_numeric_analysis()
            return

        x_col = self.insights.numeric_cols[0]
        y_col = self.insights.numeric_cols[1]

        color_col = (
            self.insights.categorical_cols[0]
            if self.insights.categorical_cols
            else None
        )

        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=self.df.columns,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_traces(
            marker=dict(size=10, opacity=0.8)
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{x_col} vs {y_col}",
            title_x=0.5,
            height=500,
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        st.plotly_chart(fig, width="stretch")

    def render_donut(self):
        """Render interactive donut chart."""

        if not self.insights.categorical_cols:
            st.info("Donut chart needs categorical data.")
            self.render_numeric_analysis()
            return

        cat_col = self.insights.categorical_cols[0]
        val_col = self.insights.numeric_cols[0] if self.insights.numeric_cols else None

        if val_col:
            pie_data = self._aggregate_by_category(cat_col, val_col).sort_values(ascending=False)
        else:
            pie_data = self.df[cat_col].astype(str).value_counts()

        if len(pie_data) > 10:
            top = pie_data.head(9)
            other = pie_data.iloc[9:].sum()
            pie_data = pd.concat([top, pd.Series({"Other": other})])

        plot_df = pd.DataFrame({
            cat_col: pie_data.index,
            "Value": pie_data.values
        })

        fig = px.pie(
            plot_df,
            names=cat_col,
            values="Value",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}<extra></extra>"
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{cat_col} Distribution",
            title_x=0.5,
            height=500
        )

        st.plotly_chart(fig, width="stretch")

    def render_histogram(self):
        """Render interactive histograms."""

        if not self.insights.numeric_cols:
            st.info("No numeric columns found.")
            return

        cols_to_plot = self.insights.numeric_cols[:3]

        chart_cols = st.columns(len(cols_to_plot))

        for idx, col in enumerate(cols_to_plot):

            with chart_cols[idx]:

                fig = px.histogram(
                    self.df,
                    x=col,
                    nbins=20,
                    color_discrete_sequence=["#3B82F6"]
                )

                fig.update_layout(
                    template="plotly_white",
                    title=f"{col} Distribution",
                    title_x=0.5,
                    height=400,
                    xaxis_title=col,
                    yaxis_title="Frequency"
                )

                st.plotly_chart(fig, width="stretch")

    def render_heatmap(self):
        """Render interactive heatmap."""

        if len(self.insights.numeric_cols) >= 2:
            self.render_correlation_analysis()
            return

        if len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:

            c1 = self.insights.categorical_cols[0]
            c2 = self.insights.categorical_cols[1]
            val = self.insights.numeric_cols[0]

            try:
                pivot = self.df.pivot_table(
                    index=c1,
                    columns=c2,
                    values=val,
                    aggfunc="sum",
                    fill_value=0
                )

                fig = px.imshow(
                    pivot,
                    color_continuous_scale="Blues",
                    aspect="auto"
                )

                fig.update_layout(
                    template="plotly_white",
                    title=f"{val} by {c1} × {c2}",
                    title_x=0.5,
                    height=550
                )

                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                logger.warning(f"Pivot heatmap failed: {e}")
                self.render_numeric_analysis()

        else:
            st.info("Heatmap requires at least two categorical or numeric columns.")
            self.render_numeric_analysis()

    def _decide_chart_type_with_llm(self, user_query: str) -> str:
        """Use LLM to dynamically decide the best chart type based on the user's question."""
        prompt = f"""You are an expert data visualization assistant.
Based on the user's query and the data properties, decide the single most appropriate chart type to display.

USER QUERY: "{user_query}"

AVAILABLE DATA:
- Numeric columns: {', '.join(self.insights.numeric_cols) if self.insights.numeric_cols else 'None'}
- Categorical columns: {', '.join(self.insights.categorical_cols) if self.insights.categorical_cols else 'None'}
- Date columns: {', '.join(self.insights.date_cols) if self.insights.date_cols else 'None'}

CHART TYPE OPTIONS:
1. "bar"
2. "horizontal_bar"
3. "line"
4. "pie"
5. "donut"
6. "scatter"
7. "heatmap"
8. "histogram"
9. "grouped_bar"
10. "auto"

Respond with ONLY ONE of these:
bar
horizontal_bar
line
pie
donut
scatter
heatmap
histogram
grouped_bar
auto
"""
        try:
            response = call_llm_with_cache(prompt, temperature=0.1)
            if response:
                choice = response.strip().lower()
                for valid_choice in ["bar", "horizontal_bar", "line", "pie", "donut", "scatter", "heatmap", "histogram", "grouped_bar", "auto"]:
                    if valid_choice in choice:
                        return valid_choice
            return "auto"
        except Exception as e:
            logger.warning("LLM chart detection failed: %s", e)
            return "auto"

    def render_insights(self, chart_type: str = "auto", user_query: str = None):
        """Main method to render all insights"""
        # Executive summary always shown
        self.render_executive_summary()
        st.divider()

        # LLM-driven chart type selection when auto
        if chart_type == "auto" and user_query:
            llm_chart_type = self._decide_chart_type_with_llm(user_query)
            if llm_chart_type != "auto":
                st.info(f" Selecting '{llm_chart_type}' chart based on your query.")
                chart_type = llm_chart_type

        # ── Dispatch to the correct renderer ─────────────────────────────────
        if chart_type == "bar":
            if len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:
                self.render_grouped_bar()
            elif self.insights.numeric_cols:
                self.render_numeric_analysis()
            else:
                st.info("No numeric columns found. Showing categorical distributions instead.")
                self.render_categorical_analysis()

        elif chart_type == "pie":
            if self.insights.categorical_cols:
                self.render_categorical_analysis()
            else:
                st.info("No categorical columns found. Showing numeric bar charts instead.")
                self.render_numeric_analysis()

        elif chart_type == "donut":
            self.render_donut()

        elif chart_type in ("line", "area"):
            self.render_line_chart()

        elif chart_type == "horizontal_bar":
            self.render_horizontal_bar()

        elif chart_type == "scatter":
            self.render_scatter()

        elif chart_type == "histogram":
            self.render_histogram()

        elif chart_type == "heatmap":
            self.render_heatmap()

        elif chart_type == "trend":
            if self.insights.date_cols and self.insights.numeric_cols:
                self.render_trend_analysis()
            else:
                st.info("Missing date or numeric columns for trend. Showing line chart instead.")
                self.render_line_chart()

        elif chart_type == "correlation":
            if len(self.insights.numeric_cols) >= 2:
                self.render_correlation_analysis()
            else:
                st.info("Need 2+ numeric columns for correlation. Showing all relevant charts.")
                chart_type = "auto"

        if chart_type == "auto":

            selected_chart = st.radio(
                "Visualization",
                [
                    "Bar",
                    "Line",
                    "Donut",
                    "Horizontal Bar",
                    "Scatter",
                    "Heatmap",
                    "Histogram"
                ],
                horizontal=True
            )

            if selected_chart == "Bar":
                self.render_numeric_analysis()

            elif selected_chart == "Line":
                self.render_line_chart()

            elif selected_chart == "Donut":
                self.render_donut()

            elif selected_chart == "Horizontal Bar":
                self.render_horizontal_bar()

            elif selected_chart == "Scatter":
                self.render_scatter()

            elif selected_chart == "Heatmap":
                self.render_heatmap()

            elif selected_chart == "Histogram":
                self.render_histogram()




def generate_insights( result_df: pd.DataFrame,
                      chart_type: str = "auto",
                      user_query: str = None
                      ):
        if result_df is None or result_df.empty:
            return
        try:
            insights_gen = DataInsightsGenerator(result_df)
            visualizer = InsightVisualizer(result_df, insights_gen)

            if chart_type == "auto":
                chart_type = st.radio(
                    "Visualization",
                    [
                    "bar",
                    "line",
                    "donut",
                    "horizontal_bar",
                    "scatter",
                    "heatmap",
                    "histogram"
                    ],
                    horizontal=True
                    )

            visualizer.render_insights(
            chart_type=chart_type,
            user_query=user_query
            )
        except Exception as e:
            logger.exception(f"Error generating insights: {e}")
            st.error(f"Could not generate insights: {str(e)}")
