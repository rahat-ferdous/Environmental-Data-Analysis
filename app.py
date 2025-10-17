import streamlit as st
import pandas as pd
import numpy as np

# Page setup
st.set_page_config(
    page_title="My Python Portfolio",
    page_icon="🐍",
    layout="wide"
)

# Header section
st.title("🐍 My Python Developer Portfolio")
st.markdown("### Welcome to my interactive Python portfolio!")

# About section
st.header("🚀 About Me")
st.write("""
I'm a passionate Python developer with expertise in:
- **Web Development** (FastAPI, Django, Flask)
- **Data Analysis** (Pandas, NumPy, Matplotlib)
- **Machine Learning** (Scikit-learn, TensorFlow)
- **Automation & Scripting**
""")

# Skills demonstration
st.header("🛠️ Skills Demo")

# Data analysis demo
st.subheader("📊 Data Analysis with Pandas")
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [1000, 1200, 800, 1500, 2000],
    'Growth %': [10, 20, -15, 25, 33]
})
st.dataframe(data)

# Simple visualization
st.subheader("📈 Sales Visualization")
st.bar_chart(data.set_index('Month')['Sales'])

# Python code example
st.subheader("💻 Python Code Example")
code = '''
def calculate_metrics(sales_data):
    """Calculate business metrics from sales data"""
    total_sales = sum(sales_data)
    average_sales = total_sales / len(sales_data)
    growth = ((sales_data[-1] - sales_data[0]) / sales_data[0]) * 100
    return total_sales, average_sales, growth

# Example usage
sales = [1000, 1200, 800, 1500, 2000]
total, average, growth_pct = calculate_metrics(sales)
print(f"Total: ${total}, Average: ${average:.2f}, Growth: {growth_pct:.1f}%")
'''
st.code(code, language='python')

# Contact section
st.header("📞 Contact Me")
st.write("""
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com)
- **GitHub**: [Your GitHub](https://github.com)
""")

st.success("🎉 Portfolio deployed successfully! Ready to customize.")
