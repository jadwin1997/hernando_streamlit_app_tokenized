# streamlit_app_github.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO, StringIO
import requests
import base64

st.title("Hernando Billing Report Analysis")

# --- Sidebar inputs (modify rates) ---
st.sidebar.header("Modify Water & Sewer Rates")
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

def make_modified_fn(p_in_2_5, p_in_5, p_out_2_5, p_out_5):
    def _fn(row):
        gallons = int(str(row["Billing Cons"]).replace(',',''))
        wtr_rate = str(row["Wtr Rate"]).upper().strip()
        swr_rate = str(row["Swr Rate"]).upper().strip()
        water_charge = 0
        if 'ACTIVE' in str(row['Status'])[:6]:
            # WATER (with user-modifiable rates)
            if wtr_rate in ["IRES", "ICOMM"]:
                if gallons <= 2:
                    water_charge = 12.50
                elif gallons <= 5:
                    if(wtr_rate == "IRES"):
                        water_charge = 12.50 + (gallons - 2) * ires_2_5
                    else:
                        water_charge = 12.50 + (gallons - 2) * icomm_2_5
                else:
                    if(wtr_rate == "IRES"):
                        water_charge = 12.50 + (3 * p_in_2_5) + (gallons - 5) * ires_5
                    else:
                        water_charge = 12.50 + (3 * p_in_2_5) + (gallons - 5) * icomm_5
            elif wtr_rate in ["ORES", "OCOMM"]:
                if gallons <= 3:
                    water_charge = 16.00
                elif gallons <= 5:
                    if(wtr_rate == "ORES"):
                        water_charge = 16.00 + (gallons - 3) * ores_2_5
                    else:
                        water_charge = 16.00 + (gallons - 3) * ocomm_2_5

                else:
                    if(wtr_rate == "ORES"):
                        water_charge = 16.00 + (2 * p_out_2_5) + (gallons - 5) * ores_5
                    else: 
                        water_charge = 16.00 + (2 * p_out_2_5) + (gallons - 5) * ocomm_5
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
def preprocess(df, p_in_2_5, p_in_5, p_out_2_5, p_out_5):
    df = df.copy()
    df['Actual_Total_Bill'] = df.apply(check_actual, axis=1)
    df['Estimated_Total_Bill'] = df.apply(check_estimated, axis=1)
    df['Modified_Total_Estimated_Bill'] = df.apply(make_modified_fn(p_in_2_5, p_in_5, p_out_2_5, p_out_5), axis=1)
    df['Actual_Estimated_Diff'] = df['Actual_Total_Bill'] - df['Estimated_Total_Bill']
    df['Relative_Error_%'] = (df['Actual_Estimated_Diff'] / df['Actual_Total_Bill']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

file = preprocess(raw, ppg_inside_2_5, ppg_inside_5, ppg_outside_2_5, ppg_outside_5)

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
st.write(str(profit_by_rate.index))
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
st.write(str(profit_by_rate.index))
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