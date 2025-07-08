# MERCEDES-DASHBOARD
This project is a professional Streamlit-based Sales Intelligence Dashboard built to analyze and visualize vehicle sales data across multiple brands and years. It supports uploading Excel files with multiple sheets (e.g., Mercedes Benz, GAC, Gargour Asia) and begins with a secure login system. Users can interactively filter data by city, sales executive, model, year, and date range. The dashboard delivers dynamic insights, including the top 15 declining models year-over-year, sales trend forecasting using a Random Forest model, market share analysis, top repeated customers based on phone numbers, yearly trends, and the most requested vehicle colors. It also features an interactive map of sales locations across Lebanon. All charts and insights are rendered using Plotly and Matplotlib, and filtered data can be downloaded directly. Fuzzy matching (via RapidFuzz) enhances address-based city classification.

This dashboard is already deployed and live on Streamlit Cloud, making it easy for sales teams and managers to access insights anytime without installation.

Tech Stack Used:
	•	Frontend & Deployment: Streamlit (hosted on Streamlit Cloud), HTML (via Streamlit), Plotly
	•	Backend/Data Processing: Pandas, NumPy, Scikit-learn (Random Forest), OpenPyXL
	•	Visualization: Matplotlib, Plotly
	•	Fuzzy Matching: RapidFuzz
	•	Map Integration: Streamlit Map / Mapbox
	•	Authentication: Simple Streamlit session-based login
 
