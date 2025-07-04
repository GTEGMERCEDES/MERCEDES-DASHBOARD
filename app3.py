import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import time


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    with st.sidebar:
        if st.button("🔒 Logout"):
            st.session_state.logged_in = False
            st.rerun()

if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center;'>🔐 Welcome to the Sales Dashboard</h2>", unsafe_allow_html=True)
    st.write("Please enter your password to continue.")

    with st.form(key="login_form"):
        password_input = st.text_input("🔑 Password", type="password", placeholder="Enter password here")
        submit_btn = st.form_submit_button("🚀 Unlock Dashboard")

    if submit_btn:
        if password_input == "tony":
            st.success("✅ Access granted! Welcome back 👋")
            st.session_state.logged_in = True
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            st.stop()
    else:
        st.stop()

# ------------------ Dashboard Starts Here ------------------
st.title("📊 Welcome to the Sales Dashboard")

st.sidebar.header("📂 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Upload Excel file (with 3 sheets)", type=["xlsx"])

if uploaded_file is not None:
    sheet_option = st.sidebar.selectbox("Choose Sheet", ["Mercedes Benz", "GAC", "Gargour ASIA"])
    df = pd.read_excel(uploaded_file, sheet_name=sheet_option, engine="openpyxl")

    # Proceed with cleaning and rest of the dashboard
    df['Inv Date'] = pd.to_datetime(df['Inv Date'], errors='coerce')
    df['Unt'] = pd.to_numeric(df['Unt'], errors='coerce')
    df = df.dropna(subset=['Inv Date', 'Unt'])
    df['Year'] = df['Inv Date'].dt.year

    # Continue with rest of your dashboard logic...
else:
    st.warning("Please upload an Excel file to proceed.")

from rapidfuzz import process, fuzz

# Step 1: Extract city names from address column
df['City'] = df['AD'].astype(str).str.extract(r'City[: ]*([A-Za-z\s]+)', expand=False)
df['City'] = df['City'].str.strip().str.title()

# Step 2: Fuzzy grouping with better token-aware logic
unique_cities = df['City'].dropna().unique()
city_mapping = {}

for city in unique_cities:
    if city not in city_mapping:
        matches = process.extract(
            city, unique_cities, scorer=fuzz.token_sort_ratio, limit=10
        )
        similar = [match[0] for match in matches if match[1] >= 75]
        for match in similar:
            city_mapping[match] = city

# Step 3: Apply mapping
df['City'] = df['City'].map(city_mapping)

# --------------------- Add Coordinates ---------------------
city_coords = {
    # Beirut & Suburbs
    'Beirut': (33.8938, 35.5018),
    'Achrafieh': (33.8869, 35.5131),
    'Hamra': (33.8945, 35.4783),
    'Mar Elias': (33.8797, 35.5002),
    'Verdun': (33.8885, 35.4897),
    'Ain El Tineh': (33.8933, 35.5033),
    'Ras Beirut': (33.8950, 35.4781),
    'Mazraa': (33.8722, 35.5103),
    'Tariq El Jdideh': (33.8700, 35.5100),
    'Furn El Chebbak': (33.8623, 35.5220),
    'Bourj Hammoud': (33.8898, 35.5234),
    
    # Mount Lebanon
    'Baabda': (33.8333, 35.5333),
    'Hazmieh': (33.8651, 35.5352),
    'Jamhour': (33.8211, 35.5623),
    'Hadath': (33.8336, 35.5441),
    'Aley': (33.8106, 35.5903),
    'Bchamoun': (33.7498, 35.4582),
    'Choueifat': (33.7500, 35.4833),
    'Mansourieh': (33.8730, 35.5684),
    'Ain Saadeh': (33.8781, 35.6053),
    'Beit Mery': (33.8600, 35.5925),
    'Broummana': (33.8681, 35.6782),
    'Antelias': (33.9122, 35.5791),
    'Dbayeh': (33.9437, 35.5976),
    'Zalka': (33.9084, 35.5722),
    'Zouk Mosbeh': (33.9532, 35.6206),
    'Jounieh': (33.9703, 35.6175),
    'Kaslik': (33.9819, 35.6405),
    'Sarba': (33.9800, 35.6300),
    'Kfarchima': (33.7893, 35.5371),
    'Kornet Chehwan': (33.8915, 35.5840),
    'Qornet El Hamra': (33.9004, 35.6941),
    'Ajaltoun': (33.9973, 35.6856),
    'Ballouneh': (34.0107, 35.6850),
    'Zouk Mikael': (33.9765, 35.6186),
    'Kfarhbab': (33.9870, 35.6523),
    'Ain El Remmaneh': (33.8590, 35.5233),
    'Elissar': (33.8924, 35.5667),

    # North Lebanon
    'Tripoli': (34.4381, 35.8308),
    'El Mina': (34.4366, 35.8255),
    'Zgharta': (34.3900, 35.8936),
    'Bcharré': (34.2517, 36.0107),
    'Koura': (34.3481, 35.8164),
    'Ehden': (34.2957, 35.9743),
    'Chekka': (34.3742, 35.6481),
    'Batroun': (34.2564, 35.6586),
    'Jbeil': (34.1230, 35.6518),
    'Halba': (34.5378, 36.0833),

    # South Lebanon
    'Sidon': (33.5606, 35.3756),
    'Tyre': (33.2700, 35.2038),
    'Jezzine': (33.5416, 35.5841),
    'Qana': (33.2200, 35.3039),
    'Zahrani': (33.4667, 35.3000),

    # Nabatieh
    'Nabatieh': (33.3789, 35.4839),
    'Kfar Remen': (33.3612, 35.4822),
    'Marjeyoun': (33.3775, 35.5922),
    'Bint Jbeil': (33.2739, 35.4189),
    'Tebnine': (33.2778, 35.4081),

    # Bekaa
    'Zahle': (33.8468, 35.9020),
    'Chtaura': (33.8271, 35.8496),
    'Riyaq': (33.8574, 36.0056),
    'Jdita': (33.8337, 35.8912),
    'Taalabaya': (33.8342, 35.8874),
    'Qabb Ilyas': (33.8236, 35.8666),
    'Ferzol': (33.8416, 35.9104),
    'Niha Bekaa': (33.7896, 35.8990),

    # Baalbek-Hermel
    'Baalbek': (34.0058, 36.2181),
    'Hermel': (34.3958, 36.3861),
    'Labweh': (34.1711, 36.2703),
    'Ras Baalbek': (34.2595, 36.4402),
    'Fakeha': (34.1667, 36.2833),

    # Extra villages and alt spellings
    'Hadath': (33.8336, 35.5441),
    'Hazmiyeh': (33.8648, 35.5433),
    'Hazmieh Mar Takla': (33.8645, 35.5452),
    'Ain El Tinneh': (33.8930, 35.5040),
    'Ain El Tineh': (33.8933, 35.5033),
    'Ain El Tine': (33.8933, 35.5032),
    'Dora': (33.9035, 35.5496),
    'Sin El Fil': (33.8667, 35.5333),
    'Fanar': (33.8864, 35.5823),
    'Jal El Dib': (33.9106, 35.5873),
    'Okaibeh': (34.0204, 35.6495),
    'Mechref': (33.6830, 35.4260),
    'Bechara El Khoury': (33.8877, 35.5081)
}
df['Latitude'] = df['City'].map(lambda x: city_coords.get(x, (None, None))[0])
df['Longitude'] = df['City'].map(lambda x: city_coords.get(x, (None, None))[1])

# --------------------- Sidebar Filters ---------------------
st.sidebar.header("Filters")
cities = sorted(df['City'].dropna().unique())
executives = sorted(df['S.executive'].dropna().unique())
models = sorted(df['Vehicle description'].dropna().astype(str).unique())
years = sorted(df['Year'].dropna().unique())

city_filter = st.sidebar.selectbox("Select City", ["All"] + cities)
exec_filter = st.sidebar.selectbox("Select Sales Executive", ["All"] + executives)
model_filter = st.sidebar.selectbox("Select Vehicle Model", ["All"] + models)
year_filter = st.sidebar.selectbox("Select Year", ["All"] + list(map(str, years)))
start_date, end_date = st.sidebar.date_input("Select Date Range", [df['Inv Date'].min(), df['Inv Date'].max()])

# --------------------- Filter Data ---------------------
dff = df.copy()
if city_filter != "All":
    dff = dff[dff['City'] == city_filter]
if exec_filter != "All":
    dff = dff[dff['S.executive'] == exec_filter]
if model_filter != "All":
    dff = dff[dff['Vehicle description'].astype(str) == model_filter]
if year_filter != "All":
    dff = dff[dff['Year'] == int(year_filter)]
dff = dff[(dff['Inv Date'] >= pd.to_datetime(start_date)) & (dff['Inv Date'] <= pd.to_datetime(end_date))]

# --------------------- KPIs ---------------------
st.title("🚗 Full Sales Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Units Sold", int(dff['Unt'].sum()))
#col2.metric("Unique Customers", dff['Customer name'].nunique())
col2.metric("Unique Customers (Phone P4)", dff['P4'].nunique())
col3.metric("Sales Period", f"{dff['Inv Date'].min().date()} ➔ {dff['Inv Date'].max().date()}")

# --------------------- Forecast ---------------------
dff_monthly = dff.resample('MS', on='Inv Date')['Unt'].sum().reset_index()
dff_monthly['Date_Ordinal'] = dff_monthly['Inv Date'].map(pd.Timestamp.toordinal)
dff_monthly['Month'] = dff_monthly['Inv Date'].dt.month
dff_monthly['Year'] = dff_monthly['Inv Date'].dt.year

X = dff_monthly[['Date_Ordinal', 'Month', 'Year']]
y = dff_monthly['Unt']
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
dff_monthly['Prediction'] = model.predict(X)

future_dates = [dff_monthly['Inv Date'].max() + pd.DateOffset(months=i) for i in range(1, 25)]
future_df = pd.DataFrame({'Inv Date': future_dates})
future_df['Date_Ordinal'] = future_df['Inv Date'].map(pd.Timestamp.toordinal)
future_df['Month'] = future_df['Inv Date'].dt.month
future_df['Year'] = future_df['Inv Date'].dt.year
future_preds = model.predict(future_df[['Date_Ordinal', 'Month', 'Year']])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dff_monthly['Inv Date'], dff_monthly['Unt'], label='Actual Sales', marker='o')
ax.plot(dff_monthly['Inv Date'], dff_monthly['Prediction'], label='Prediction', linestyle='--', color='orange')
ax.plot(future_df['Inv Date'], future_preds, label='Forecast (Next 24 Months)', linestyle='-', marker='o', color='green')
ax.set_title('📊 Forecast (Random Forest)')
ax.set_xlabel('Date')
ax.set_ylabel('Units Sold')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --------------------- Top Executives ---------------------
st.subheader("👨‍💼 Top Executives by Units Sold")
exec_sales = dff.groupby('S.executive')['Unt'].sum().sort_values(ascending=True).reset_index()
fig_exec = px.bar(exec_sales, x='Unt', y='S.executive', orientation='h', title="Units Sold by Executive", color='Unt')
st.plotly_chart(fig_exec, use_container_width=True)


# --------------------- Repeated Customers by Phone Number ---------------------
st.subheader("🔁 Top 10 Most Repeated Customers (by Phone Number)")

# Drop rows with missing phone numbers to avoid counting NaNs
repeated_phones = dff.dropna(subset=['P4'])

top_phone_counts = repeated_phones['P4'].value_counts().loc[lambda x: x > 1].head(10).reset_index()
top_phone_counts.columns = ['Phone Number', 'Purchases']

# Get the most frequent customer name associated with each phone number
phone_to_name = repeated_phones.groupby('P4')['Customer name'].agg(
    lambda x: ', '.join(sorted(set(x)))
).reset_index()
top_customers_with_names = pd.merge(top_phone_counts, phone_to_name, left_on='Phone Number', right_on='P4').drop(columns='P4')
top_customers_with_names.columns = ['Phone Number', 'Purchases', 'Customer Name']

# Reorder columns
top_customers_with_names = top_customers_with_names[['Customer Name', 'Phone Number', 'Purchases']]

st.dataframe(top_customers_with_names)

# --------------------- Market Share ---------------------
st.subheader("📊 Top 10 Vehicle Market Share + Others")
model_sales = dff.groupby('Vehicle description')['Unt'].sum().sort_values(ascending=False)
top10 = model_sales.head(10)
others = model_sales[10:].sum()
market_share = pd.concat([top10, pd.Series({'Others': others})]).reset_index()
market_share.columns = ['Model', 'Units']
fig_market_share = px.pie(market_share, names='Model', values='Units', title='Top 10 Market Share + Others')
st.plotly_chart(fig_market_share, use_container_width=True)

# --------------------- Colors ---------------------
if 'Colour/WheelbaseDesc' in dff.columns:
    st.subheader("🎨 Top 10 Requested Colors")
    color_summary = dff['Colour/WheelbaseDesc'].value_counts().head(10).reset_index()
    color_summary.columns = ['Color', 'Units']
    fig_colors = px.bar(color_summary, x='Color', y='Units', title='Top 10 Colors Requested', color_discrete_sequence=['#1f77b4']*10)
    st.plotly_chart(fig_colors, use_container_width=True)

# --------------------- Yearly Trend ---------------------
st.subheader("📈 Yearly Sales Trend")
yearly_sales = dff.groupby(dff['Inv Date'].dt.year)['Unt'].sum().reset_index()
yearly_sales.columns = ['Year', 'Total Units']
fig_yearly = px.line(yearly_sales, x='Year', y='Total Units', markers=True, title="Yearly Sales Trend")
st.plotly_chart(fig_yearly, use_container_width=True)

# --------------------- Top Selling Models by Year ---------------------
st.subheader("📈 Top 10 Selling Models per Year")

# Calculate total units per model
top_models = dff.groupby('Vehicle description')['Unt'].sum().nlargest(10).index

# Filter data for only top 10 models
model_year = dff[dff['Vehicle description'].isin(top_models)]
model_year = model_year.groupby([dff['Vehicle description'].astype(str), dff['Inv Date'].dt.year])['Unt'].sum().reset_index()
model_year.columns = ['Model', 'Year', 'Units']

# Plot
fig_model_year = px.line(model_year, x='Year', y='Units', color='Model', title='Yearly Sales for Top 10 Models')
st.plotly_chart(fig_model_year, use_container_width=True)

# --------------------- Top 15 Declining Models ---------------------
# Ensure the uploaded file is already processed
if uploaded_file is not None and 'Vehicle description' in df.columns:
    st.subheader("📉 Top 15 Declining Models (YoY Comparison)")

    # Allow the user to choose years dynamically from available data
    year_options = sorted(df['Year'].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox("Select Base Year (Previous)", year_options, index=len(year_options)-2)
    with col2:
        year2 = st.selectbox("Select Comparison Year (Current)", year_options, index=len(year_options)-1)

    if year1 >= year2:
        st.warning("⚠️ Base year must be earlier than comparison year.")
    else:
        # Group by model and year
        decline_df = df[['Vehicle description', 'Year', 'Unt']].dropna()
        grouped = decline_df.groupby(['Vehicle description', 'Year'])['Unt'].sum().reset_index()
        pivot_df = grouped.pivot(index='Vehicle description', columns='Year', values='Unt').fillna(0)

        if year1 in pivot_df.columns and year2 in pivot_df.columns:
            # Calculate YoY decline
            pivot_df['YoY Growth (%)'] = ((pivot_df[year2] - pivot_df[year1]) / pivot_df[year1].replace(0, 1)) * 100
            top_declines = pivot_df.sort_values('YoY Growth (%)').head(15).reset_index()

            # Display table
            st.dataframe(top_declines[['Vehicle description', year1, year2, 'YoY Growth (%)']])

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(top_declines['Vehicle description'], top_declines['YoY Growth (%)'], color='red')
            ax.set_ylabel("YoY Growth (%)")
            ax.set_title(f"Top 15 Declining Models from {year1} to {year2}")
            ax.set_xticklabels(top_declines['Vehicle description'], rotation=90)
            ax.grid(axis='y')
            st.pyplot(fig)
        else:
            st.warning(f"🚫 One or both selected years ({year1}, {year2}) not found in the data.")

# --------------------- Engines ---------------------
if 'Engine number' in dff.columns:
    st.subheader("⚙️ Top 10 Engines by Total Units")
    engine_sales = dff.groupby('Engine number')['Unt'].sum().sort_values(ascending=False).head(10).reset_index()
    engine_sales.columns = ['Engine Number', 'Units']
    st.dataframe(engine_sales)

# --------------------- Map ---------------------
if 'Latitude' in dff.columns and dff['Latitude'].notna().any():
    st.subheader("🗺️ Sales Distribution by City")
    map_data = dff[['Latitude', 'Longitude', 'Unt']].dropna().rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map(map_data)
    

# --------------------- Download ---------------------
csv = dff.to_csv(index=False)
st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")
