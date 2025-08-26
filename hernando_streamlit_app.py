# streamlit_app_github.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO, StringIO
import requests
import base64
import plotly.express as px
st.title("Hernando Billing Report Analysis")

# --- Sidebar inputs (modify rates) ---
st.sidebar.header("Modify Water & Sewer Base Rates")
ires_base = st.sidebar.number_input("Inside City Residential (IRES) base price:", value=12.50)
icomm_base = st.sidebar.number_input("Inside City Commercial (ICOMM) base price:", value=12.50)
ores_base = st.sidebar.number_input("Outside City (ORES) base price:", value=16.00)
ocomm_base = st.sidebar.number_input("Outside City (OCOMM) base price:", value=16.00)

st.sidebar.header("Modify Water & Sewer Variable Rates")
ires_2_5 = st.sidebar.number_input("Inside City Residential (IRES) price/1000 gallons (2k–5k):", value=3.15)
ires_5   = st.sidebar.number_input("Inside City Residential (IRES) price/1000 gallons (>5k):", value=3.50)

icomm_2_5 = st.sidebar.number_input("Inside City Commercial (ICOMM) price/1000 gallons (2k–5k):", value=3.15)
icomm_5   = st.sidebar.number_input("Inside City Commercial (ICOMM) price/1000 gallons (>5k):", value=3.50)

ores_2_5= st.sidebar.number_input("Outside City (ORES) price/1000 gallons (3k–5k):", value=3.50)
ores_5  = st.sidebar.number_input("Outside City (ORES) price/1000 gallons (>5k):", value=3.95)

ocomm_2_5= st.sidebar.number_input("Outside City (OCOMM) price/1000 gallons (3k–5k):", value=3.50)
ocomm_5  = st.sidebar.number_input("Outside City (OCOMM) price/1000 gallons (>5k):", value=3.95)


# --- GitHub private repo details ---
GITHUB_OWNER = "jadwin1997"
GITHUB_REPO  = "hernando_streamlit_app_data"
CSV_PATH     = "Hernando-NewInfo.csv"  # path inside the repo
GITHUB_TOKEN = st.secrets["github"]["token"]

# --- Fetch CSV from GitHub ---
@st.cache_data
def load_csv_from_github(owner, repo, path, token, branch="main"):
    """
    Fetches a CSV file from a private GitHub repository using the raw URL and a personal access token.
    
    Parameters:
        owner (str): GitHub username or organization.
        repo (str): Repository name.
        path (str): Path to the CSV file relative to the repo root.
        token (str): GitHub personal access token with repo permissions.
        branch (str): Branch name (default: 'main').
    
    Returns:
        pd.DataFrame: CSV loaded as a DataFrame with 'Period' column parsed as datetime.
    """
    # Raw download URL
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download CSV. Status: {response.status_code} {response.text}")

    # Read CSV into DataFrame
    df = pd.read_csv(StringIO(response.text))
    df['Period'] = pd.to_datetime(df['Period'])
    return df


raw = load_csv_from_github(GITHUB_OWNER, GITHUB_REPO, CSV_PATH, GITHUB_TOKEN)

# --- Helpers ---
def clean_amt(x):
    return float(str(x).replace(',', '').replace('$', ''))

def check_actual(row):
    return clean_amt(row['Wtr Amt']) + clean_amt(row['Swr Amt']) + clean_amt(row['DCRUA Amt'])

def check_estimated(row):
    gallons = int(str(row["Billing Cons"]).replace(',',''))
    wtr_rate = str(row["Wtr Rate"]).upper().strip()
    swr_rate = str(row["Swr Rate"]).upper().strip()
    water_charge = 0
    if 'ACTIVE' in str(row['Status'])[:6]:
        # WATER
        if wtr_rate in ["IRES", "ICOMM"]:
            if gallons <= 2:
                water_charge = 12.50
            elif gallons <= 5:
                water_charge = 12.50 + (gallons - 2) * 3.15
            else:
                water_charge = 12.50 + (3 * 3.15) + (gallons - 5) * 3.50
        elif wtr_rate in ["ORES", "OCOMM"]:
            if gallons <= 3:
                water_charge = 16.00
            elif gallons <= 5:
                water_charge = 16.00 + (gallons - 3) * 3.50
            else:
                water_charge = 16.00 + (2 * 3.50) + (gallons - 5) * 3.95
        else:
            return check_actual(row)
        # SEWER
        dcrua = clean_amt(row['DCRUA Amt'])
        if swr_rate in ["IRES", "ICOMM"]:
            sewer_charge = max(water_charge / 2, 6.25) + dcrua
        elif swr_rate in ["ORES", "OCOMM"]:
            sewer_charge = max(water_charge / 2, 8.00) + dcrua
        else:
            return check_actual(row)
        return round(water_charge + sewer_charge, 2)
    return 0

def make_modified_fn(ires_base,icomm_base,ores_base,ocomm_base,ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5):
    def _fn(row):
        gallons = int(str(row["Billing Cons"]).replace(',',''))
        wtr_rate = str(row["Wtr Rate"]).upper().strip()
        swr_rate = str(row["Swr Rate"]).upper().strip()
        water_charge = 0
        if 'ACTIVE' in str(row['Status'])[:6]:
            # WATER (with user-modifiable rates)
            if wtr_rate in ["IRES", "ICOMM"]:
                if gallons <= 2:
                    if(wtr_rate == "IRES"):
                        water_charge = ires_base
                    else:
                        water_charge = icomm_base
                elif gallons <= 5:
                    if(wtr_rate == "IRES"):
                        water_charge = ires_base + (gallons - 2) * ires_2_5
                    else:
                        water_charge = icomm_base + (gallons - 2) * icomm_2_5
                else:
                    if(wtr_rate == "IRES"):
                        water_charge = ires_base + (3 * ires_2_5) + (gallons - 5) * ires_5
                    else:
                        water_charge = icomm_base + (3 * icomm_2_5) + (gallons - 5) * icomm_5
            elif wtr_rate in ["ORES", "OCOMM"]:
                if gallons <= 3:
                    if(wtr_rate == "ORES"):
                        water_charge = ores_base
                    else:
                        water_charge = ocomm_base
                elif gallons <= 5:
                    if(wtr_rate == "ORES"):
                        water_charge = ores_base + (gallons - 3) * ores_2_5
                    else:
                        water_charge = ocomm_base + (gallons - 3) * ocomm_2_5

                else:
                    if(wtr_rate == "ORES"):
                        water_charge = ores_base + (2 * ores_2_5) + (gallons - 5) * ores_5
                    else: 
                        water_charge = ocomm_base + (2 * ocomm_2_5) + (gallons - 5) * ocomm_5
            else:
                return check_actual(row)
            dcrua = clean_amt(row['DCRUA Amt'])
            if swr_rate in ["IRES", "ICOMM"]:
                sewer_charge = max(water_charge / 2, 6.25) + dcrua
            elif swr_rate in ["ORES", "OCOMM"]:
                sewer_charge = max(water_charge / 2, 8.00) + dcrua
            else:
                return check_actual(row)
            return round(water_charge + sewer_charge, 2)
        return 0
    return _fn

@st.cache_data
def preprocess(df,ires_base,icomm_base,ores_base,ocomm_base, ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5):
    df = df.copy()
    df['Actual_Total_Bill'] = df.apply(check_actual, axis=1)
    df['Estimated_Total_Bill'] = df.apply(check_estimated, axis=1)
    df['Modified_Total_Estimated_Bill'] = df.apply(make_modified_fn(ires_base,icomm_base,ores_base,ocomm_base,ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5), axis=1)
    df['Actual_Estimated_Diff'] = df['Actual_Total_Bill'] - df['Estimated_Total_Bill']
    df['Relative_Error_%'] = (df['Actual_Estimated_Diff'] / df['Actual_Total_Bill']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

file = preprocess(raw,ires_base,icomm_base,ores_base,ocomm_base, ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5)

# --- Display ---
st.subheader("Sample of Billing Data")
st.dataframe(file.head())

# --- Monthly totals & line chart ---
monthly_totals = file.groupby(file['Period'].dt.to_period('M')).agg({
    'Actual_Total_Bill':'sum',
    'Estimated_Total_Bill':'sum',
    'Modified_Total_Estimated_Bill':'sum'
})

st.subheader("Monthly Revenue (Actual vs Estimated vs Modified)")
fig, ax = plt.subplots(figsize=(10,5))
monthly_totals['Actual_Total_Bill'].plot(ax=ax, marker='o', label='Actual')
monthly_totals['Estimated_Total_Bill'].plot(ax=ax, marker='s', linestyle='--', label='Estimated')
monthly_totals['Modified_Total_Estimated_Bill'].plot(ax=ax, marker='d', linestyle='--', label='Modified')
ax.set_ylabel("Total Bill ($)")
ax.set_xlabel("Month")
ax.legend()
st.pyplot(fig)

# --- Revenue summary ---
st.subheader("Revenue Summary")
st.write(f"Actual Total Revenue: ${monthly_totals['Actual_Total_Bill'].sum():,.2f}")
st.write(f"Estimated Total Revenue: ${monthly_totals['Estimated_Total_Bill'].sum():,.2f}")
st.write(f"Modified Total Revenue: ${monthly_totals['Modified_Total_Estimated_Bill'].sum():,.2f}")
diff_est = monthly_totals['Actual_Total_Bill'].sum() - monthly_totals['Estimated_Total_Bill'].sum()
diff_mod = monthly_totals['Modified_Total_Estimated_Bill'].sum() - monthly_totals['Actual_Total_Bill'].sum()
st.write(f"Difference (Actual - Estimated): ${diff_est:,.2f}")
st.write(f"Difference (Modified - Actual): ${diff_mod:,.2f}")


# --- Profits by Rate Class ---
st.subheader("Revenue by Water Rate Class (Actual Revenue)")
file['Wtr Rate'] = file['Wtr Rate'].str.strip()
# Group by water rate
profit_by_rate = file[(file['Wtr Rate']!='METER') & (file['Wtr Rate']!='125 MTR') & (file['Wtr Rate']!='FIREHYDR')].groupby('Wtr Rate')['Actual_Total_Bill'].sum()
# Matplotlib Pie Chart
fig2, ax2 = plt.subplots()
ax2.pie(
    profit_by_rate, 
    labels=profit_by_rate.index, 
    autopct='%1.1f%%', 
    startangle=90
)
ax2.set_title("Profit Distribution by Water Rate Class")
st.pyplot(fig2)





# --- Profits by Rate Class ---
st.subheader("Revenue by Water Rate Class (Estimated Revenue)")
file['Wtr Rate'] = file['Wtr Rate'].str.strip()
# Group by water rate
profit_by_rate = file[(file['Wtr Rate']!='METER') & (file['Wtr Rate']!='125 MTR') & (file['Wtr Rate']!='FIREHYDR')].groupby('Wtr Rate')['Modified_Total_Estimated_Bill'].sum()

# Matplotlib Pie Chart
fig3, ax3 = plt.subplots()
ax3.pie(
    profit_by_rate, 
    labels=profit_by_rate.index, 
    autopct='%1.1f%%', 
    startangle=90
)
ax3.set_title("Profit Distribution by Water Rate Class")
st.pyplot(fig3)





# --- Distribution of Revenue by Usage Range ---
st.subheader("IRES Water Usage By Usage Tiers (gallons)")

# Convert Billing Cons (assumed in thousands of gallons) to numeric
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')

# Define usage range bins
def usage_range(g):
    if g <= 2:
        return "0–2k"
    elif g <= 5:
        return "2–5k"
    else:
        return "5k+"

file['UsageRange'] = file[file['Wtr Rate']=='IRES']['Billing Cons'].apply(usage_range)

# Group and sum thousands of gallons
revenue_by_usage = file.groupby("UsageRange")["Billing Cons"].sum().reindex(["0–2k", "2–5k", "5k+"])

# Pie chart
fig4, ax4 = plt.subplots()
ax4.pie(
    revenue_by_usage, 
    labels=revenue_by_usage.index, 
    autopct='%1.1f%%',
    startangle=90
)
ax4.set_title("Water Usage Distribution by Usage Tiers")
st.pyplot(fig4)



# --- Distribution of Revenue by Usage Range ---
st.subheader("ICOMM Distribution Water Usage by Usage Tiers (gallons)")

# Convert Billing Cons (assumed in thousands of gallons) to numeric
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')

file['UsageRange'] = file[file['Wtr Rate']=='ICOMM']['Billing Cons'].apply(usage_range)

# Group and sum revenue
revenue_by_usage = file.groupby("UsageRange")['Billing Cons'].sum().reindex(["0–2k", "2–5k", "5k+"])

# Pie chart
fig5, ax5 = plt.subplots()
ax5.pie(
    revenue_by_usage, 
    labels=revenue_by_usage.index, 
    autopct='%1.1f%%',
    startangle=90
)
ax5.set_title("Water Usage Distribution by Usage Tiers")
st.pyplot(fig5)




# --- Distribution of Revenue by Usage Range ---
st.subheader("ORES Distribution Water Usage by Usage Tiers  (gallons)")

# Convert Billing Cons (assumed in thousands of gallons) to numeric
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')



file['UsageRange'] = file[file['Wtr Rate']=='ORES']['Billing Cons'].apply(usage_range)

# Group and sum revenue
revenue_by_usage = file.groupby("UsageRange")['Billing Cons'].sum().reindex(["0–2k", "2–5k", "5k+"])

# Pie chart
fig6, ax6 = plt.subplots()
ax6.pie(
    revenue_by_usage, 
    labels=revenue_by_usage.index, 
    autopct='%1.1f%%',
    startangle=90
)
ax6.set_title("Water Usage Distribution by Usage Tiers")
st.pyplot(fig6)

# --- Distribution of Revenue by Usage Range ---
st.subheader("OCOMM Distribution Water Usage by Usage Tiers  (gallons)")

# Convert Billing Cons (assumed in thousands of gallons) to numeric
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')
file['UsageRange'] = file[file['Wtr Rate']=='OCOMM']['Billing Cons'].apply(usage_range)

# Group and sum revenue
revenue_by_usage = file.groupby("UsageRange")['Billing Cons'].sum().reindex(["0–2k", "2–5k", "5k+"])

# Pie chart
fig7, ax7 = plt.subplots()
ax7.pie(
    revenue_by_usage, 
    labels=revenue_by_usage.index, 
    autopct='%1.1f%%',
    startangle=90
)
ax7.set_title("Water Usage Distribution by Usage Tiers")
st.pyplot(fig7)









# --- Combined Distribution by Class + Usage ---
st.subheader("Revenue Distribution by Water Rate Class + Usage Range")

# Ensure Billing Cons is numeric gallons
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')

# Define usage ranges
def usage_range(gallons):
    if gallons < 2000:
        return "0–2k"
    elif gallons < 5000:
        return "2–5k"
    else:
        return "5k+"

# Apply usage category only for valid classes
valid_classes = ["IRES", "ORES", "ICOMM", "OCOMM"]
mask = file['Wtr Rate'].isin(valid_classes)

file.loc[mask, "UsageRange"] = file.loc[mask, "Billing Cons"].apply(usage_range)

# Combine Class + Range
file.loc[mask, "Class+Usage"] = file.loc[mask, "Wtr Rate"] + " " + file.loc[mask, "UsageRange"]

# Group and sum revenue
revenue_by_class_usage = (
    file.loc[mask]
    .groupby("Class+Usage")["Actual_Total_Bill"]
    .sum()
    .sort_values(ascending=False)
)

# Pie chart
fig8, ax8 = plt.subplots()
ax8.pie(
    revenue_by_class_usage,
    labels=revenue_by_class_usage.index,
    autopct='%1.1f%%',
    startangle=90
)
ax8.set_title("Revenue Distribution by Class + Usage Tier")
st.pyplot(fig8)



# --- Bar chart ---
fig9, ax9 = plt.subplots(figsize=(10,6))
revenue_by_class_usage.plot(
    kind="bar",
    ax=ax9,
    color="skyblue",
    edgecolor="black"
)

ax9.set_title("Profit by Class + Usage Tier")
ax9.set_xlabel("Class + Usage Tier")
ax9.set_ylabel("Profit ($)")
ax9.set_xticklabels(revenue_by_class_usage.index, rotation=45, ha="right")

st.pyplot(fig9)


















