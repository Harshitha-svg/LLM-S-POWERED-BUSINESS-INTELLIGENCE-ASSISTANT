import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from io import BytesIO

st.set_page_config(page_title="SmartBI Navigator", layout="wide")

# ---- Initialize session states ----
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

# ---- Sidebar ----
st.sidebar.title("🧠 LLM'S Powered Business Intelligence Assistant")

# Clear Data and New Chat buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.df = None
        st.rerun()
with col2:
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.query_counter = 0
        st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["📂 Upload Data", "👀 Data Preview", "📊 Visualization", "📈 Insights Dashboard", "🔍 Data Query", "🧾 Export Report"]
)

# ---- Chat History in Sidebar ----
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Recent Queries")
if len(st.session_state.chat_history) == 0:
    st.sidebar.info("No queries yet.")
else:
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        with st.sidebar.expander(f"Q{i}: {q[:30]}..."):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a[:150]}...")

# ---- Helper function for chart download ----
def save_figure_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# ---- Helper function to process query ----
def process_query(query, df):
    query_lower = query.lower()
    response = ""
    result_df = None
    
    try:
        # Show first n rows
        if "first" in query_lower and "row" in query_lower:
            nums = re.findall(r'\d+', query_lower)
            num = int(nums[0]) if nums else 5
            result_df = df.head(num)
            response = f"Showing first {num} rows"
        
        # Show last n rows
        elif "last" in query_lower and "row" in query_lower:
            nums = re.findall(r'\d+', query_lower)
            num = int(nums[0]) if nums else 5
            result_df = df.tail(num)
            response = f"Showing last {num} rows"
        
        # Show columns
        elif "column" in query_lower and ("show" in query_lower or "what" in query_lower or "list" in query_lower):
            response = f"**Columns:** {', '.join(df.columns.tolist())}"
        
        # Mean
        elif "mean" in query_lower or "average" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                response = f"**Mean of {col_match[0]}:** {df[col_match[0]].mean():.2f}"
            else:
                result_df = pd.DataFrame(df.mean(numeric_only=True), columns=['Mean'])
                response = "Mean values calculated"
        
        # Median
        elif "median" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                response = f"**Median of {col_match[0]}:** {df[col_match[0]].median():.2f}"
            else:
                result_df = pd.DataFrame(df.median(numeric_only=True), columns=['Median'])
                response = "Median values calculated"
        
        # Sum/Total
        elif "sum" in query_lower or "total" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                response = f"**Sum of {col_match[0]}:** {df[col_match[0]].sum():.2f}"
            else:
                result_df = pd.DataFrame(df.sum(numeric_only=True), columns=['Sum'])
                response = "Sum values calculated"
        
        # Count
        elif "count" in query_lower or "how many" in query_lower:
            if "row" in query_lower:
                response = f"**Total rows:** {len(df)}"
            else:
                result_df = pd.DataFrame(df.count(), columns=['Count'])
                response = "Row counts per column"
        
        # Unique values
        elif "unique" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                unique_vals = df[col_match[0]].unique()
                response = f"**Unique values in {col_match[0]}:** {len(unique_vals)}\n\nValues: {', '.join(map(str, unique_vals[:20]))}"
            else:
                response = "Please specify a column name"
        
        # Max
        elif "max" in query_lower or "maximum" in query_lower or "highest" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                response = f"**Maximum of {col_match[0]}:** {df[col_match[0]].max()}"
            else:
                result_df = pd.DataFrame(df.max(numeric_only=True), columns=['Max'])
                response = "Maximum values calculated"
        
        # Min
        elif "min" in query_lower or "minimum" in query_lower or "lowest" in query_lower:
            col_match = [col for col in df.columns if col.lower() in query_lower]
            if col_match:
                response = f"**Minimum of {col_match[0]}:** {df[col_match[0]].min()}"
            else:
                result_df = pd.DataFrame(df.min(numeric_only=True), columns=['Min'])
                response = "Minimum values calculated"
        
        # Missing values
        elif "missing" in query_lower or "null" in query_lower:
            missing = df.isnull().sum()
            result_df = pd.DataFrame(missing[missing > 0], columns=['Missing Count'])
            response = f"**Total missing values:** {df.isnull().sum().sum()}"
        
        # Data types
        elif "type" in query_lower or "dtype" in query_lower:
            result_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            response = "Data types displayed"
        
        # Shape/Size
        elif "shape" in query_lower or "size" in query_lower or "dimension" in query_lower:
            response = f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns"
        
        else:
            response = "❓ Sorry, I couldn't understand that question. Try asking about: mean, sum, count, first rows, columns, unique values, max, min, missing values, data types, or shape."
        
        return response, result_df
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}", None

# ---- Upload Data ----
if page == "📂 Upload Data":
    st.title("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Data Preview under Upload
            if st.session_state.df is not None:
                st.subheader("👀 Quick Data Preview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(st.session_state.df))
                with col2:
                    st.metric("Total Columns", len(st.session_state.df.columns))
                with col3:
                    st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
                
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': st.session_state.df.columns,
                    'Type': st.session_state.df.dtypes.values,
                    'Non-Null Count': st.session_state.df.count().values,
                    'Null Count': st.session_state.df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# ---- Data Preview ----
elif page == "👀 Data Preview":
    st.title("👀 Data Preview")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Filters
        st.subheader("🔍 Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.slider("Number of rows to display", 5, min(100, len(df)), min(50, len(df)))
        with col2:
            columns_to_show = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist()[:10])
        
        if columns_to_show:
            st.dataframe(df[columns_to_show].head(show_rows), use_container_width=True)
        
        # Search functionality
        st.subheader("🔎 Search Data")
        col1, col2 = st.columns(2)
        with col1:
            search_col = st.selectbox("Select column to search", df.columns)
        with col2:
            search_term = st.text_input("Enter search term")
        
        if search_term:
            mask = df[search_col].astype(str).str.contains(search_term, case=False, na=False)
            st.write(f"Found {mask.sum()} matching rows")
            st.dataframe(df[mask], use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first from 'Upload Data'.")

# ---- Visualization ----
elif page == "📊 Visualization":
    st.title("📊 Visualization")
    if st.session_state.df is not None:
        df = st.session_state.df
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Select X-axis", columns)
        with col2:
            y_col = st.selectbox("Select Y-axis", columns)
        with col3:
            chart_type = st.selectbox("Select Chart Type", 
                                      ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", 
                                       "Histogram", "Box Plot", "Heatmap", "Area Chart"])

        if st.button("Generate Chart", type="primary"):
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                if chart_type == "Bar Chart":
                    df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax, color='skyblue')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                elif chart_type == "Line Chart":
                    df.plot(x=x_col, y=y_col, kind="line", ax=ax, marker='o')
                    
                elif chart_type == "Pie Chart":
                    df.groupby(x_col)[y_col].sum().plot(kind="pie", ax=ax, autopct="%1.1f%%")
                    ax.set_ylabel("")
                    
                elif chart_type == "Scatter Plot":
                    df.plot.scatter(x=x_col, y=y_col, ax=ax, alpha=0.6)
                    
                elif chart_type == "Histogram":
                    df[y_col].plot(kind="hist", ax=ax, bins=30, edgecolor='black')
                    ax.set_xlabel(y_col)
                    ax.set_ylabel("Frequency")
                    
                elif chart_type == "Box Plot":
                    df[[y_col]].plot(kind="box", ax=ax)
                    
                elif chart_type == "Heatmap":
                    if len(numeric_cols) > 0:
                        corr = df[numeric_cols].corr()
                        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    else:
                        st.warning("No numeric columns for heatmap")
                        
                elif chart_type == "Area Chart":
                    df.plot(x=x_col, y=y_col, kind="area", ax=ax, alpha=0.4)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Display with controls
                with st.expander("📊 Chart View", expanded=True):
                    st.pyplot(fig)
                    
                    # Download button
                    buf = save_figure_to_bytes(fig)
                    st.download_button(
                        label="📥 Download Chart",
                        data=buf,
                        file_name=f"{chart_type.replace(' ', '_')}.png",
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating chart: {str(e)}")
    else:
        st.warning("⚠️ Please upload a dataset first.")

# ---- Insights Dashboard ----
elif page == "📈 Insights Dashboard":
    st.title("📈 Insights Dashboard")
    if st.session_state.df is not None:
        df = st.session_state.df

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())

        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.subheader("📊 Summary Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            st.subheader("📈 Distribution Analysis")
            col = st.selectbox("Select column for distribution", numeric_cols)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Histogram
            df[col].hist(bins=30, ax=ax1, edgecolor='black')
            ax1.set_title(f'Histogram of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            
            # Box plot
            df[[col]].boxplot(ax=ax2)
            ax2.set_title(f'Box Plot of {col}')
            ax2.set_ylabel(col)
            
            plt.tight_layout()
            
            with st.expander("📊 Distribution Charts", expanded=True):
                st.pyplot(fig)
                buf = save_figure_to_bytes(fig)
                st.download_button(
                    label="📥 Download Distribution Charts",
                    data=buf,
                    file_name=f"{col}_distribution.png",
                    mime="image/png"
                )

        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            st.subheader("📑 Categorical Data Analysis")
            cat_col = st.selectbox("Select categorical column", cat_cols)
            value_counts = df[cat_col].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts.plot(kind='bar', ax=ax, color='coral')
            ax.set_title(f'Top 10 Values in {cat_col}')
            ax.set_xlabel(cat_col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            with st.expander("📊 Categorical Chart", expanded=True):
                st.pyplot(fig)
                buf = save_figure_to_bytes(fig)
                st.download_button(
                    label="📥 Download Categorical Chart",
                    data=buf,
                    file_name=f"{cat_col}_bar_chart.png",
                    mime="image/png"
                )
            
            st.write(f"**Unique values:** {df[cat_col].nunique()}")
            st.dataframe(value_counts, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# ---- Data Query Page ----
elif page == "🔍 Data Query":
    st.title("🔍 Data Query")
    if st.session_state.df is not None:
        st.markdown("""
        ### Ask questions about your data:
        **Examples:**
        - Show first 10 rows
        - What is the mean of [column_name]?
        - Show sum of all numeric columns
        - Count rows where [column] > 100
        - Show unique values in [column]
        - What are the columns?
        - Show data types
        - What is the shape of data?
        """)
    else:
        st.warning("⚠️ Please upload a dataset first from 'Upload Data'.")

# ---- Export Report ----
elif page == "🧾 Export Report":
    st.title("🧾 Export Report")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv,
                file_name="SmartBI_Report.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel Export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            st.download_button(
                "📥 Download as Excel",
                data=excel_data,
                file_name="SmartBI_Report.xlsx",
                mime="application/vnd.openxmlwriter.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Summary Statistics Export
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_csv = df[numeric_cols].describe().to_csv()
                st.download_button(
                    "📥 Download Summary Stats",
                    data=summary_csv,
                    file_name="Summary_Statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.warning("⚠️ Please upload a dataset first.")

# ---- GLOBAL CHAT INTERFACE (Available on all pages) ----
st.markdown("---")
st.markdown("### 💬 Ask Questions About Your Data")

if st.session_state.df is not None:
    df = st.session_state.df
    
    # Create a unique key for each query
    query_key = f"query_input_{st.session_state.query_counter}"
    query = st.text_input("Your question:", key=query_key, placeholder="e.g., Show first 10 rows, What is the mean?, Count rows...")
    
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and query:
        response, result_df = process_query(query, df)
        
        # Display response
        st.markdown("#### 📝 Answer:")
        st.info(response)
        
        if result_df is not None:
            st.dataframe(result_df, use_container_width=True)
        
        # Save to history
        st.session_state.chat_history.append((query, response))
        st.session_state.query_counter += 1
        st.rerun()
    
    # Display recent conversation
    if len(st.session_state.chat_history) > 0:
        st.markdown("#### 📜 Recent Conversation")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"Q: {q}", expanded=(i==1)):
                st.markdown(f"**A:** {a}")
else:
    st.info("📂 Please upload a dataset from the 'Upload Data' page to start asking questions.")