import plotly.express as px

def auto_plot(df):
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 2:
        return px.scatter(df, x=num_cols[0], y=num_cols[1], title="Auto Scatter Plot")
    elif len(num_cols) == 1:
        return px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
    else:
        return None
