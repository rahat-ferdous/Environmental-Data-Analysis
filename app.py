import streamlit as st
import pandas as pd
import numpy as np
import itertools  # Add this line

st.set_page_config(
    page_title="Flood Analysis",
    page_icon="🌊", 
    layout="wide"
)

# Header
st.title("🌊 Environmental Data Analysis")
st.markdown("### Specializing in Environmental Data Analysis & Flood Risk Assessment")

# NumPy Demonstration Section
st.header("🔬 NumPy for Scientific Computing")

# Create sample data using NumPy
st.subheader("1. Array Operations with NumPy")
col1, col2 = st.columns(2)

with col1:
    # Create arrays for flood data
    st.write("**Creating Flood Data Arrays:**")
    rainfall = np.array([45, 78, 120, 95, 60, 200])  # mm of rainfall
    river_levels = np.array([2.1, 2.8, 4.5, 3.2, 2.5, 5.8])  # meters
    
    st.write("Daily Rainfall (mm):", rainfall)
    st.write("River Levels (m):", river_levels)
    
    # Basic operations
    st.write("**Data Operations:**")
    st.write("Total Weekly Rainfall:", np.sum(rainfall), "mm")
    st.write("Average River Level:", f"{np.mean(river_levels):.2f} m")
    st.write("Peak River Level:", f"{np.max(river_levels):.2f} m")

with col2:
    # Statistical analysis
    st.write("**Statistical Analysis:**")
    st.write("Rainfall Std Dev:", f"{np.std(rainfall):.2f} mm")
    st.write("Flood Threshold Exceedances:", np.sum(river_levels > 3.0))
    st.write("Correlation:", f"{np.corrcoef(rainfall, river_levels)[0,1]:.2f}")

# Flood Risk Analysis
st.header("📊 Flood Risk Analysis Simulation")

# Generate synthetic flood data
np.random.seed(42)  # For reproducible results

# Create realistic flood dataset
days = 365
baseline_rainfall = np.random.normal(50, 20, days)  # Normal rainfall pattern

# Add flood events
flood_days = np.random.choice(days, 15, replace=False)  # 15 flood days
baseline_rainfall[flood_days] = np.random.normal(150, 40, 15)  # Heavy rainfall

# Calculate river levels (simplified model)
river_levels_sim = 2.0 + (baseline_rainfall / 50) + np.random.normal(0, 0.3, days)

# Create flood risk categories
flood_risk = np.where(river_levels_sim > 4.0, "High",
                     np.where(river_levels_sim > 3.0, "Medium", "Low"))

# Create DataFrame
flood_data = pd.DataFrame({
    'Day': range(1, days + 1),
    'Rainfall_mm': baseline_rainfall.round(1),
    'River_Level_m': river_levels_sim.round(2),
    'Flood_Risk': flood_risk
})

st.write("**Sample Flood Monitoring Data (First 30 days):**")
st.dataframe(flood_data.head(30))

# Flood Analysis Metrics
st.subheader("🌧️ Flood Analysis Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk_days = np.sum(flood_data['Flood_Risk'] == 'High')
    st.metric("High Risk Days", high_risk_days)
with col2:
    max_rainfall = np.max(flood_data['Rainfall_mm'])
    st.metric("Max Rainfall", f"{max_rainfall:.1f} mm")
with col3:
    flood_threshold_breaches = np.sum(flood_data['River_Level_m'] > 3.5)
    st.metric("Flood Threshold Breaches", flood_threshold_breaches)
with col4:
    avg_rainfall = np.mean(flood_data['Rainfall_mm'])
    st.metric("Average Rainfall", f"{avg_rainfall:.1f} mm")

# Flood Pattern Analysis
st.header("📈 Flood Pattern Detection")

# Calculate moving averages for trend analysis
window_size = 7  # 7-day moving average
rainfall_ma = np.convolve(flood_data['Rainfall_mm'], 
                         np.ones(window_size)/window_size, mode='valid')
river_ma = np.convolve(flood_data['River_Level_m'], 
                      np.ones(window_size)/window_size, mode='valid')

col1, col2 = st.columns(2)

with col1:
    st.write("**7-Day Moving Averages:**")
    ma_data = pd.DataFrame({
        'Day': range(window_size, days + 1),
        'Rainfall_MA': rainfall_ma.round(1),
        'River_Level_MA': river_ma.round(2)
    })
    st.dataframe(ma_data.head(15))

with col2:
    st.write("**Extreme Event Analysis:**")
    extreme_rain = flood_data[flood_data['Rainfall_mm'] > 100]
    extreme_river = flood_data[flood_data['River_Level_m'] > 4.0]
    
    st.write(f"Extreme Rainfall Days (>100mm): {len(extreme_rain)}")
    st.write(f"Dangerous River Levels (>4.0m): {len(extreme_river)}")
    st.write(f"Longest Flood Sequence: {max(len(list(g)) for k, g in itertools.groupby(flood_risk) if k == 'High')} days")

# Interactive Flood Simulation
st.header("🎮 Interactive Flood Risk Simulator")

st.write("Adjust parameters to see how they affect flood risk:")

col1, col2 = st.columns(2)

with col1:
    base_rainfall = st.slider("Base Rainfall (mm)", 30, 80, 50)
    rainfall_variability = st.slider("Rainfall Variability", 10, 40, 20)

with col2:
    flood_frequency = st.slider("Flood Events per Year", 5, 30, 15)
    flood_intensity = st.slider("Flood Intensity", 120, 200, 150)

if st.button("Run Flood Simulation"):
    # Generate new simulation with user parameters
    np.random.seed(123)
    new_rainfall = np.random.normal(base_rainfall, rainfall_variability, 365)
    
    # Add flood events based on user input
    flood_days_sim = np.random.choice(365, flood_frequency, replace=False)
    new_rainfall[flood_days_sim] = np.random.normal(flood_intensity, 30, flood_frequency)
    
    # Calculate river levels
    new_river_levels = 2.0 + (new_rainfall / 50) + np.random.normal(0, 0.3, 365)
    new_flood_risk = np.where(new_river_levels > 4.0, "High",
                            np.where(new_river_levels > 3.0, "Medium", "Low"))
    
    # Display results
    st.subheader("Simulation Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk Days", np.sum(new_flood_risk == 'High'))
    with col2:
        st.metric("Max River Level", f"{np.max(new_river_levels):.2f} m")
    with col3:
        st.metric("Total Flood Days", np.sum(new_flood_risk != 'Low'))
    
    # Show sample of simulated data
    sim_data = pd.DataFrame({
        'Rainfall': new_rainfall[:10].round(1),
        'River_Level': new_river_levels[:10].round(2),
        'Risk': new_flood_risk[:10]
    })
    st.write("**Sample Simulated Data:**")
    st.dataframe(sim_data)

# Advanced Flood Modeling
st.header("🔍 Advanced Flood Modeling Techniques")

st.write("""
**Python & NumPy Applications in Flood Analysis:**

1. **Statistical Analysis** - Extreme value analysis, frequency analysis
2. **Time Series Modeling** - Trend detection, seasonal patterns
3. **Risk Assessment** - Probability calculations, scenario modeling
4. **Data Processing** - Handling large environmental datasets
5. **Simulation Modeling** - Monte Carlo simulations for flood risk
""")

# Code Example: Flood Prediction Function
st.subheader("💻 Flood Prediction Algorithm")

flood_code = '''
import numpy as np

def predict_flood_risk(rainfall_data, river_levels, soil_moisture):
    """
    Predict flood risk using multiple environmental factors
    """
    # Normalize inputs
    rainfall_norm = (rainfall_data - np.mean(rainfall_data)) / np.std(rainfall_data)
    river_norm = (river_levels - np.mean(river_levels)) / np.std(river_levels)
    soil_norm = (soil_moisture - np.mean(soil_moisture)) / np.std(soil_moisture)
    
    # Calculate composite risk score (simplified)
    risk_score = (0.5 * rainfall_norm + 
                  0.3 * river_norm + 
                  0.2 * soil_norm)
    
    # Classify risk levels
    risk_levels = np.where(risk_score > 1.5, "Extreme",
                          np.where(risk_score > 0.5, "High",
                                  np.where(risk_score > -0.5, "Medium", "Low")))
    
    return risk_levels, risk_score

# Example usage
rainfall = np.array([45, 120, 80, 200, 60])  # mm
river_levels = np.array([2.1, 4.5, 3.2, 5.8, 2.5])  # meters
soil_moisture = np.array([0.3, 0.8, 0.5, 0.9, 0.4])  # saturation

risks, scores = predict_flood_risk(rainfall, river_levels, soil_moisture)
print("Flood Risk Predictions:", risks)
print("Risk Scores:", scores)
'''

st.code(flood_code, language='python')


# Footer
st.markdown("---")
st.write("Built with Python, NumPy, and Streamlit for environmental data analysis and flood risk assessment!")
