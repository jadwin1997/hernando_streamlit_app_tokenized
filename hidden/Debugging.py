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


st.markdown("""
**Instructions:**
- Graphs/Sections labeled "Actual" use actual billing amounts from the data.
- Graphs/Sections labeled "Estimated" use the estimated billing amounts from the data.
- Graphs/Sections labeled "Modified" use user-modified rates to recalculate estimated billing amounts.         
- Please note that changing Usage Tier rates will affect both Estimated and Modified graphs/sections.
            
- To access the modifiable rates, click the arrow on the top left to open the sidebar. Go to the Modify Rates section and input new rates as desired.
""")


 # --- Sidebar inputs (modify rates) ---
st.sidebar.header("Modify Water Base Rates")#st.sidebar.slider("Inside City Residential (IRES) base price:", 0.00, 50.00, 12.50,.01, key='ires_base_price')#
ires_base = st.sidebar.number_input("Inside City Residential (IRES) base price ($):", value=12.50, key='ires_base_price')
icomm_base = st.sidebar.number_input("Inside City Commercial (ICOMM) base price ($):", value=12.50, key='icomm_base_price')
ores_base = st.sidebar.number_input("Outside City (ORES) base price ($):", value=16.00, key='ores_base_price')
ocomm_base = st.sidebar.number_input("Outside City (OCOMM) base price ($):", value=16.00, key='ocomm_base_price')

st.sidebar.header("Modify Water Variable Rates")
ires_2_5 = st.sidebar.number_input("Inside City Residential (IRES) price/1000 gallons (tier 1)(default 2k–5k):", value=3.15, key='ires_t1_price')
ires_5   = st.sidebar.number_input("Inside City Residential (IRES) price/1000 gallons (tier 2)(default >5k):", value=3.50, key='ires_t2_price')

icomm_2_5 = st.sidebar.number_input("Inside City Commercial (ICOMM) price/1000 gallons (tier 1)(default 2k–5k):", value=3.15, key='icomm_t1_price')
icomm_5   = st.sidebar.number_input("Inside City Commercial (ICOMM) price/1000 gallons (tier 2)(default >5k):", value=3.50, key='icomm_t2_price')

ores_2_5= st.sidebar.number_input("Outside City (ORES) price/1000 gallons (tier 1)(default 3k–5k):", value=3.50, key='ores_t1_price')
ores_5  = st.sidebar.number_input("Outside City (ORES) price/1000 gallons (tier 2)(default >5k):", value=3.95, key='ores_t2_price')

ocomm_2_5= st.sidebar.number_input("Outside City (OCOMM) price/1000 gallons (tier 1)(default 3k–5k):", value=3.50, key='ocomm_t1_price')
ocomm_5  = st.sidebar.number_input("Outside City (OCOMM) price/1000 gallons (tier 2)(default >5k):", value=3.95, key='ocomm_t2_price')

#Sidebar inputs Usage Tier definitions
st.sidebar.header("Usage Tier Definitions")

# IRES + ICOMM (inside city) usage tiers
ires_tier1 = st.sidebar.number_input("IRES Tier 1 max (k gallons):", value=2, step=1, key='ires_tier1_amount')
ires_tier2 = st.sidebar.number_input("IRES Tier 2 max (k gallons):", value=5, step=1, key='ires_tier2_amount')
ICOMM_tier1 = st.sidebar.number_input("ICOMM Tier 1 max (k gallons):", value=2, step=1, key='icomm_tier1_amount')
ICOMM_tier2 = st.sidebar.number_input("ICOMM Tier 2 max (k gallons):", value=5, step=1, key='icomm_tier2_amount')
# ORES + OCOMM (outside city) usage tiers
ORES_tier1 = st.sidebar.number_input("ORES Tier 1 max (k gallons):", value=3, step=1, key='ores_tier1_amount')
ORES_tier2 = st.sidebar.number_input("ORES Tier 2 max (k gallons):", value=5, step=1, key='ores_tier2_amount')
OCOMM_tier1 = st.sidebar.number_input("OCOMM Tier 1 max (k gallons):", value=3, step=1, key='ocomm_tier1_amount')
OCOMM_tier2 = st.sidebar.number_input("OCOMM Tier 2 max (k gallons):", value=5, step=1, key='ocomm_tier2_amount')

#Sidebar inputs for 
st.sidebar.header("Sewer and DECRUA Adjustments")
sewer_rate = st.sidebar.number_input("Sewer Rate (dollars per thousand gallons):", value=2.06, step=0.01, key='sewer_rate')
check_box_sewer_multiplier_enable = st.sidebar.checkbox("Enable Sewer Rate Multiplier (Sewer Rate = water charge * multiplier)", value=True, key='sewer_rate_multiplier_enable')
sewer_multiplier_rate = st.sidebar.number_input("Water Charge Multiplier for Sewer Bill:", value=0.5, step=0.1, key='sewer_multiplier_rate', label_visibility="visible")
DCRUA_rate = st.sidebar.number_input("DCRUA Rate (dollars per thousand gallons):", value=3.84, step=0.1, key='DCRUA_rate')

#sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate

# --- GitHub private repo details ---
GITHUB_OWNER = "jadwin1997"
GITHUB_REPO  = "hernando_streamlit_app_data"
CSV_PATH     = "Hernando-NewInfo.csv"  # path inside the repo
GITHUB_TOKEN = st.secrets["github"]["token"]
summed_dcrua_actual = 0
summed_sewer_actual = 0
summed_water_charge_actual = 0
summed_dcrua_modified = 0
summed_sewer_modified = 0
summed_water_charge_modified = 0
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
def clean_amt(series):
    try:
        return (
            series.astype(str)
            .str.replace('[\$,]', '', regex=True)
            .astype(float)
        )
    except:
        return float(str(series).replace(',', '').replace('$', ''))
# --- Helpers ---
def clean_amt_row(x):
    return float(str(x).replace(',', '').replace('$', ''))
def check_actual(row):
    return clean_amt(row['Wtr Amt']) + clean_amt(row['Swr Amt']) + clean_amt(row['DCRUA Amt'])

def check_actual_wtr(row):
    try:
    
        return clean_amt(row['Wtr Amt'])
    except:
        return clean_amt_row(row['Wtr Amt'])


def check_actual_swr(row):
    try:
    
        return clean_amt(row['Swr Amt'])
    except:
        return clean_amt_row(row['Swr Amt'])

def check_actual_dcrua(row):
    try:
    
        return clean_amt(row['DCRUA Amt'])
    except:
        return clean_amt_row(row['DCRUA Amt'])

def check_estimated_vectorized_final(df):
    df = df.copy()

    # --- Get individual estimated components ---
    water_est = get_water_rate_estimated_vectorized(df)
    sewer_est = get_sewer_rate_estimated_vectorized(df)
    dcrua_est = df.apply(get_dcrua_rate_estimated, axis=1)

    # --- Add component columns ---
    df['Water_Charge'] = water_est
    df['Sewer_Charge'] = sewer_est
    df['DCRUA_Charge'] = dcrua_est

    # --- Clean and normalize status ---
    df['Status'] = df['Status'].astype(str).str.upper()
    active = df['Status'].str.startswith('ACTIVE')

    # --- Total Estimated Bill ---
    df['Estimated_Total_Bill'] = np.where(
        active,
        df['Water_Charge'] + df['Sewer_Charge'] + df['DCRUA_Charge'],
        0
    )

    return df



def get_water_rate_estimated_vectorized(df):
    # Clean and prepare columns
    df = df.copy()
    df['Gallons'] = df['Billing Cons'].astype(str).str.replace(',', '').astype(int)
    df['Wtr_Rate'] = df['Wtr Rate'].astype(str).str.upper().str.strip()
    df['Status'] = df['Status'].astype(str).str.upper()
    
    # Initialize the water charge column
    df['Estimated_Water_Charge'] = 0.0
    
    # Only active accounts
    active_mask = df['Status'].str.startswith('ACTIVE')
    
    # IRES / ICOMM rates
    ires_mask = active_mask & df['Wtr_Rate'].isin(['IRES', 'ICOMM'])
    ires_gallons = df.loc[ires_mask, 'Gallons']
    ires_charge = np.where(
        ires_gallons <= 2,
        12.50,
        np.where(
            ires_gallons <= 5,
            12.50 + (ires_gallons - 2) * 3.15,
            12.50 + 3 * 3.15 + (ires_gallons - 5) * 3.50
        )
    )
    df.loc[ires_mask, 'Estimated_Water_Charge'] = ires_charge
    
    # ORES / OCOMM rates
    ores_mask = active_mask & df['Wtr_Rate'].isin(['ORES', 'OCOMM'])
    ores_gallons = df.loc[ores_mask, 'Gallons']
    ores_charge = np.where(
        ores_gallons <= 3,
        16.00,
        np.where(
            ores_gallons <= 5,
            16.00 + (ores_gallons - 3) * 3.50,
            16.00 + 2 * 3.50 + (ores_gallons - 5) * 3.95
        )
    )
    df.loc[ores_mask, 'Estimated_Water_Charge'] = ores_charge
    
    # Optional: for other rates, fallback to check_actual_wtr
    other_mask = active_mask & ~df['Wtr_Rate'].isin(['IRES','ICOMM','ORES','OCOMM'])
    if other_mask.any():
        df.loc[other_mask, 'Estimated_Water_Charge'] = df.loc[other_mask].apply(check_actual_wtr, axis=1)
    
    return df['Estimated_Water_Charge']






def get_sewer_rate_estimated_vectorized(df):
    df = df.copy()
    
    # --- Clean numeric columns ---
    df['Billing_Cons_Num'] = df['Billing Cons'].astype(str).str.replace(',', '').astype(float)
    df['Swr_Amt_Num'] = df['Swr Amt'].apply(lambda x: float(str(x).replace(',', '').replace('$','')) if pd.notna(x) else 0)

    # --- Uppercase/strip ---
    df['Wtr_Rate'] = df['Wtr Rate'].astype(str).str.upper().str.strip()
    df['Swr_Rate'] = df['Swr Rate'].astype(str).str.upper().str.strip()
    df['Status_Active'] = df['Status'].astype(str).str[:6] == 'ACTIVE'

    # --- Water charge ---
    wtr_charge = pd.Series(0, index=df.index, dtype=float)
    mask_i = df['Status_Active'] & df['Wtr_Rate'].isin(['IRES','ICOMM'])
    mask_o = df['Status_Active'] & df['Wtr_Rate'].isin(['ORES','OCOMM'])

    # IRES/ICOMM
    wtr_charge.loc[mask_i & (df['Billing_Cons_Num'] <= 2)] = 12.50
    wtr_charge.loc[mask_i & (df['Billing_Cons_Num'] > 2) & (df['Billing_Cons_Num'] <= 5)] = \
        12.50 + (df.loc[mask_i & (df['Billing_Cons_Num'] > 2) & (df['Billing_Cons_Num'] <= 5), 'Billing_Cons_Num'] - 2) * 3.15
    wtr_charge.loc[mask_i & (df['Billing_Cons_Num'] > 5)] = 12.50 + 3*3.15 + \
        (df.loc[mask_i & (df['Billing_Cons_Num'] > 5), 'Billing_Cons_Num'] - 5) * 3.50

    # ORES/OCOMM
    wtr_charge.loc[mask_o & (df['Billing_Cons_Num'] <= 3)] = 16.00
    wtr_charge.loc[mask_o & (df['Billing_Cons_Num'] > 3) & (df['Billing_Cons_Num'] <= 5)] = \
        16.00 + (df.loc[mask_o & (df['Billing_Cons_Num'] > 3) & (df['Billing_Cons_Num'] <= 5), 'Billing_Cons_Num'] - 3) * 3.50
    wtr_charge.loc[mask_o & (df['Billing_Cons_Num'] > 5)] = 16.00 + 2*3.50 + \
        (df.loc[mask_o & (df['Billing_Cons_Num'] > 5), 'Billing_Cons_Num'] - 5) * 3.95

    # --- Sewer charge ---
    sewer_charge = pd.Series(0, index=df.index, dtype=float)

    # IRES/ICOMM sewer
    mask_s_i = df['Status_Active'] & df['Swr_Rate'].isin(['IRES','ICOMM'])
    sewer_charge.loc[mask_s_i] = (wtr_charge / 2).clip(lower=6.25)

    # ORES/OCOMM sewer
    mask_s_o = df['Status_Active'] & df['Swr_Rate'].isin(['ORES','OCOMM'])
    sewer_charge.loc[mask_s_o] = (wtr_charge / 2).clip(lower=8.00)

    # Unknown water rates but known sewer rates: use actual sewer if water_charge = 0
    mask_water_zero = df['Status_Active'] & (wtr_charge == 0) & df['Swr_Rate'].isin(['IRES','ICOMM','ORES','OCOMM'])
    sewer_charge.loc[mask_water_zero] = df.loc[mask_water_zero, 'Swr_Amt_Num']

    # Unknown sewer rates: use actual sewer amount
    mask_unknown_swr = df['Status_Active'] & ~df['Swr_Rate'].isin(['IRES','ICOMM','ORES','OCOMM'])
    sewer_charge.loc[mask_unknown_swr] = df.loc[mask_unknown_swr, 'Swr_Amt_Num']

    return sewer_charge


def get_sewer_rate_estimated(row):
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
            return check_actual_swr(row)
        # SEWER
        dcrua = clean_amt(row['DCRUA Amt'])
        if swr_rate in ["IRES", "ICOMM"]:
            sewer_charge = max(water_charge / 2, 6.25)
        elif swr_rate in ["ORES", "OCOMM"]:
            sewer_charge = max(water_charge / 2, 8.00)
        else:
            return check_actual_swr(row)
        return sewer_charge
    return 0

def get_dcrua_rate_estimated(row):
    # gallons = int(str(row["Billing Cons"]).replace(',',''))
    # wtr_rate = str(row["Wtr Rate"]).upper().strip()
    # swr_rate = str(row["Swr Rate"]).upper().strip()
    # water_charge = 0
    # if 'ACTIVE' in str(row['Status'])[:6]:
    #     return round(check_actual_dcrua(row), 2)
    # return 0
    return(clean_amt(row['DCRUA Amt']))


import numpy as np
import pandas as pd

def compute_modified_bill(df,
                          ires_base, icomm_base, ores_base, ocomm_base,
                          ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate,
                          icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate,
                          ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate,
                          ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate,
                          base_sewer_rate, sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate):
    

     # --- Clean numeric amount columns ---
    for col, raw in [
        ('Wtr_Amt_Num', 'Wtr Amt'),
        ('Swr_Amt_Num', 'Swr Amt'),
        ('DCRUA_Num', 'DCRUA Amt')
    ]:
        if col not in df.columns:
            if raw in df.columns:
                df[col] = (
                    df[raw]
                    .astype(str)
                    .str.replace(r'[\$,]', '', regex=True)
                    .replace('', '0')
                    .astype(float)
                )
            else:
                df[col] = 0.0

    df = df.copy()

    # --- Preprocess ---
    df['Gallons'] = df['Billing Cons'].astype(str).str.replace(',', '').astype(int)
    df['Wtr Rate'] = df['Wtr Rate'].astype(str).str.upper().str.strip()
    df['Swr Rate'] = df['Swr Rate'].astype(str).str.upper().str.strip()
    df['Status'] = df['Status'].astype(str)

    # Clean numeric amount columns if needed
    for col in ['Wtr_Amt_Num', 'Swr_Amt_Num', 'DCRUA_Num']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    active_mask = df['Status'].str.startswith('ACTIVE')

    # --- Rate masks ---
    valid_wtr = df['Wtr Rate'].isin(['IRES', 'ICOMM', 'ORES', 'OCOMM'])
    valid_swr = df['Swr Rate'].isin(['IRES', 'ICOMM', 'ORES', 'OCOMM'])
    valid_both = valid_wtr & valid_swr

    ires_mask = df['Wtr Rate'] == "IRES"
    icomm_mask = df['Wtr Rate'] == "ICOMM"
    ores_mask = df['Wtr Rate'] == "ORES"
    ocomm_mask = df['Wtr Rate'] == "OCOMM"

    # --- Water Charges ---
    def tiered_charge(base, t1_max, t2_max, t2_rate, t3_rate):
        return np.select(
            [
                df['Gallons'] <= t1_max,
                (df['Gallons'] > t1_max) & (df['Gallons'] <= t2_max),
                df['Gallons'] > t2_max
            ],
            [
                base,
                base + (df['Gallons'] - t1_max) * t2_rate,
                base + (t2_max - t1_max) * t2_rate + (df['Gallons'] - t2_max) * t3_rate
            ],
            default=0
        )

    df['Water_Charge'] = np.select(
        [ires_mask, icomm_mask, ores_mask, ocomm_mask],
        [
            tiered_charge(ires_base, ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate),
            tiered_charge(icomm_base, icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate),
            tiered_charge(ores_base, ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate),
            tiered_charge(ocomm_base, ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate)
        ],
        default=0
    )

    # --- Sewer Charges ---
    ires_icomm_swr = df['Swr Rate'].isin(['IRES', 'ICOMM'])
    ores_ocomm_swr = df['Swr Rate'].isin(['ORES', 'OCOMM'])

    sewer_i = np.where(
        sewer_multiplier_enable,
        np.maximum(df['Water_Charge'] * sewer_multiplier_rate, 6.25) + df['Gallons'] * DCRUA_rate,
        np.maximum(df['Gallons'] * base_sewer_rate, 6.25) + df['Gallons'] * DCRUA_rate
    )

    sewer_o = np.where(
        sewer_multiplier_enable,
        np.maximum(df['Water_Charge'] * sewer_multiplier_rate, 8.00) + df['Gallons'] * DCRUA_rate,
        np.maximum(df['Gallons'] * base_sewer_rate, 8.00) + df['Gallons'] * DCRUA_rate
    )

    df['Sewer_Charge'] = np.select(
        [ires_icomm_swr, ores_ocomm_swr],
        [sewer_i, sewer_o],
        default=0
    )

    # --- DCRUA Charge ---
    #df['DCRUA_Charge'] = df['Gallons'] * DCRUA_rate

    # --- Default total for valid rate combos ---
    df['Modified_Total_Estimated_Bill'] = np.where(
        active_mask & valid_both,
        df['Water_Charge'] + df['Sewer_Charge'],
        0
    )

    # --- Fallback: use actual row totals where rate missing ---
    missing_rate_mask = active_mask & ~valid_both
    df.loc[missing_rate_mask, 'Modified_Total_Estimated_Bill'] = (
        df.loc[missing_rate_mask, 'Wtr_Amt_Num']
        + df.loc[missing_rate_mask, 'Swr_Amt_Num']
        + df.loc[missing_rate_mask, 'DCRUA_Num']
    )

    return df


# def make_modified_fn(
#     ires_base, icomm_base, ores_base, ocomm_base,
#     ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate,
#     icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate,
#     ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate,
#     ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate, base_sewer_rate, sewer_multiplier_enable, sewer_multiplier, DCRUA_base_rate
# ):
#     def _fn(row):
#         gallons = int(str(row["Billing Cons"]).replace(',', ''))
#         wtr_rate = str(row["Wtr Rate"]).upper().strip()
#         swr_rate = str(row["Swr Rate"]).upper().strip()
#         water_charge = 0

#         if 'ACTIVE' in str(row['Status'])[:6]:
#             # WATER (with user-modifiable rates)
#             if wtr_rate == "IRES":
#                 if gallons <= ires_t1_max:
#                     water_charge = ires_base
#                 elif gallons <= ires_t2_max:
#                     water_charge = ires_base + (gallons - ires_t1_max) * ires_t2_rate
#                 else:
#                     water_charge = (
#                         ires_base
#                         + (ires_t2_max - ires_t1_max) * ires_t2_rate
#                         + (gallons - ires_t2_max) * ires_t3_rate
#                     )

#             elif wtr_rate == "ICOMM":
#                 if gallons <= icomm_t1_max:
#                     water_charge = icomm_base
#                 elif gallons <= icomm_t2_max:
#                     water_charge = icomm_base + (gallons - icomm_t1_max) * icomm_t2_rate
#                 else:
#                     water_charge = (
#                         icomm_base
#                         + (icomm_t2_max - icomm_t1_max) * icomm_t2_rate
#                         + (gallons - icomm_t2_max) * icomm_t3_rate
#                     )

#             elif wtr_rate == "ORES":
#                 if gallons <= ores_t1_max:
#                     water_charge = ores_base
#                 elif gallons <= ores_t2_max:
#                     water_charge = ores_base + (gallons - ores_t1_max) * ores_t2_rate
#                 else:
#                     water_charge = (
#                         ores_base
#                         + (ores_t2_max - ores_t1_max) * ores_t2_rate
#                         + (gallons - ores_t2_max) * ores_t3_rate
#                     )

#             elif wtr_rate == "OCOMM":
#                 if gallons <= ocomm_t1_max:
#                     water_charge = ocomm_base
#                 elif gallons <= ocomm_t2_max:
#                     water_charge = ocomm_base + (gallons - ocomm_t1_max) * ocomm_t2_rate
#                 else:
#                     water_charge = (
#                         ocomm_base
#                         + (ocomm_t2_max - ocomm_t1_max) * ocomm_t2_rate
#                         + (gallons - ocomm_t2_max) * ocomm_t3_rate
#                     )

#             else:
#                 return check_actual(row)

#             # SEWER (same as before)
#             dcrua = clean_amt(row['DCRUA Amt'])
#             if swr_rate in ["IRES", "ICOMM"]:
#                 if(sewer_multiplier_enable):
#                     sewer_charge = max(water_charge * sewer_multiplier_rate, 6.25) + gallons * DCRUA_rate#add dynamic dcrua, min sewer charge, and sewer charge
#                 else:
#                     sewer_charge = max(gallons * base_sewer_rate, 6.25) + gallons * DCRUA_rate
#             elif swr_rate in ["ORES", "OCOMM"]:
#                 if(sewer_multiplier_enable):
#                     sewer_charge = max(water_charge * sewer_multiplier_rate, 8.00) + gallons * DCRUA_rate
#                 else:
#                     sewer_charge = max(gallons * base_sewer_rate, 8.00) + gallons * DCRUA_rate
#             else:
#                 return check_actual(row)

#             return water_charge + sewer_charge

#         return 0
#     return _fn

def get_modified_water_charge(
    ires_base, icomm_base, ores_base, ocomm_base,
    ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate,
    icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate,
    ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate,
    ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate, base_sewer_rate, sewer_multiplier_enable, sewer_multiplier, DCRUA_base_rate
):
    def _fn(row):
        gallons = int(str(row["Billing Cons"]).replace(',', ''))
        wtr_rate = str(row["Wtr Rate"]).upper().strip()
        swr_rate = str(row["Swr Rate"]).upper().strip()
        water_charge = 0

        if 'ACTIVE' in str(row['Status'])[:6]:
            # WATER (with user-modifiable rates)
            if wtr_rate == "IRES":
                if gallons <= ires_t1_max:
                    water_charge = ires_base
                elif gallons <= ires_t2_max:
                    water_charge = ires_base + (gallons - ires_t1_max) * ires_t2_rate
                else:
                    water_charge = (
                        ires_base
                        + (ires_t2_max - ires_t1_max) * ires_t2_rate
                        + (gallons - ires_t2_max) * ires_t3_rate
                    )

            elif wtr_rate == "ICOMM":
                if gallons <= icomm_t1_max:
                    water_charge = icomm_base
                elif gallons <= icomm_t2_max:
                    water_charge = icomm_base + (gallons - icomm_t1_max) * icomm_t2_rate
                else:
                    water_charge = (
                        icomm_base
                        + (icomm_t2_max - icomm_t1_max) * icomm_t2_rate
                        + (gallons - icomm_t2_max) * icomm_t3_rate
                    )

            elif wtr_rate == "ORES":
                if gallons <= ores_t1_max:
                    water_charge = ores_base
                elif gallons <= ores_t2_max:
                    water_charge = ores_base + (gallons - ores_t1_max) * ores_t2_rate
                else:
                    water_charge = (
                        ores_base
                        + (ores_t2_max - ores_t1_max) * ores_t2_rate
                        + (gallons - ores_t2_max) * ores_t3_rate
                    )

            elif wtr_rate == "OCOMM":
                if gallons <= ocomm_t1_max:
                    water_charge = ocomm_base
                elif gallons <= ocomm_t2_max:
                    water_charge = ocomm_base + (gallons - ocomm_t1_max) * ocomm_t2_rate
                else:
                    water_charge = (
                        ocomm_base
                        + (ocomm_t2_max - ocomm_t1_max) * ocomm_t2_rate
                        + (gallons - ocomm_t2_max) * ocomm_t3_rate
                    )

            else:
                return check_actual_wtr(row)
            return water_charge

        return 0
    return _fn


def get_modified_sewer_charge(
    ires_base, icomm_base, ores_base, ocomm_base,
    ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate,
    icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate,
    ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate,
    ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate, base_sewer_rate, sewer_multiplier_enable, sewer_multiplier, DCRUA_base_rate
):
    def _fn(row):
        gallons = int(str(row["Billing Cons"]).replace(',', ''))
        wtr_rate = str(row["Wtr Rate"]).upper().strip()
        swr_rate = str(row["Swr Rate"]).upper().strip()
        water_charge = 0

        if 'ACTIVE' in str(row['Status'])[:6]:
            # WATER (with user-modifiable rates)
            if wtr_rate == "IRES":
                if gallons <= ires_t1_max:
                    water_charge = ires_base
                elif gallons <= ires_t2_max:
                    water_charge = ires_base + (gallons - ires_t1_max) * ires_t2_rate
                else:
                    water_charge = (
                        ires_base
                        + (ires_t2_max - ires_t1_max) * ires_t2_rate
                        + (gallons - ires_t2_max) * ires_t3_rate
                    )

            elif wtr_rate == "ICOMM":
                if gallons <= icomm_t1_max:
                    water_charge = icomm_base
                elif gallons <= icomm_t2_max:
                    water_charge = icomm_base + (gallons - icomm_t1_max) * icomm_t2_rate
                else:
                    water_charge = (
                        icomm_base
                        + (icomm_t2_max - icomm_t1_max) * icomm_t2_rate
                        + (gallons - icomm_t2_max) * icomm_t3_rate
                    )

            elif wtr_rate == "ORES":
                if gallons <= ores_t1_max:
                    water_charge = ores_base
                elif gallons <= ores_t2_max:
                    water_charge = ores_base + (gallons - ores_t1_max) * ores_t2_rate
                else:
                    water_charge = (
                        ores_base
                        + (ores_t2_max - ores_t1_max) * ores_t2_rate
                        + (gallons - ores_t2_max) * ores_t3_rate
                    )

            elif wtr_rate == "OCOMM":
                if gallons <= ocomm_t1_max:
                    water_charge = ocomm_base
                elif gallons <= ocomm_t2_max:
                    water_charge = ocomm_base + (gallons - ocomm_t1_max) * ocomm_t2_rate
                else:
                    water_charge = (
                        ocomm_base
                        + (ocomm_t2_max - ocomm_t1_max) * ocomm_t2_rate
                        + (gallons - ocomm_t2_max) * ocomm_t3_rate
                    )

            else:
                return check_actual_swr(row)

            # SEWER (same as before)
            dcrua = clean_amt(row['DCRUA Amt'])
            if swr_rate in ["IRES", "ICOMM"]:
                if(sewer_multiplier_enable):
                    sewer_charge = max(water_charge * sewer_multiplier_rate, 6.25) #add dynamic dcrua, min sewer charge, and sewer charge
                else:
                    sewer_charge = max(gallons * base_sewer_rate, 6.25) 
            elif swr_rate in ["ORES", "OCOMM"]:
                if(sewer_multiplier_enable):
                    sewer_charge = max(water_charge * sewer_multiplier_rate, 8.00) 
                else:
                    sewer_charge = max(gallons * base_sewer_rate, 8.00)
            else:
                return check_actual_swr(row)

            return sewer_charge

        return 0
    return _fn

def get_modified_dcrua(
    ires_base, icomm_base, ores_base, ocomm_base,
    ires_t1_max, ires_t2_max, ires_t2_rate, ires_t3_rate,
    icomm_t1_max, icomm_t2_max, icomm_t2_rate, icomm_t3_rate,
    ores_t1_max, ores_t2_max, ores_t2_rate, ores_t3_rate,
    ocomm_t1_max, ocomm_t2_max, ocomm_t2_rate, ocomm_t3_rate, base_sewer_rate, sewer_multiplier_enable, sewer_multiplier, DCRUA_base_rate
):
    def _fn(row):
        gallons = int(str(row["Billing Cons"]).replace(',', ''))
        wtr_rate = str(row["Wtr Rate"]).upper().strip()
        swr_rate = str(row["Swr Rate"]).upper().strip()
        water_charge = 0

        if 'ACTIVE' in str(row['Status'])[:6]:
            # WATER (with user-modifiable rates)
            if wtr_rate == "IRES":
                if gallons <= ires_t1_max:
                    water_charge = ires_base
                elif gallons <= ires_t2_max:
                    water_charge = ires_base + (gallons - ires_t1_max) * ires_t2_rate
                else:
                    water_charge = (
                        ires_base
                        + (ires_t2_max - ires_t1_max) * ires_t2_rate
                        + (gallons - ires_t2_max) * ires_t3_rate
                    )

            elif wtr_rate == "ICOMM":
                if gallons <= icomm_t1_max:
                    water_charge = icomm_base
                elif gallons <= icomm_t2_max:
                    water_charge = icomm_base + (gallons - icomm_t1_max) * icomm_t2_rate
                else:
                    water_charge = (
                        icomm_base
                        + (icomm_t2_max - icomm_t1_max) * icomm_t2_rate
                        + (gallons - icomm_t2_max) * icomm_t3_rate
                    )

            elif wtr_rate == "ORES":
                if gallons <= ores_t1_max:
                    water_charge = ores_base
                elif gallons <= ores_t2_max:
                    water_charge = ores_base + (gallons - ores_t1_max) * ores_t2_rate
                else:
                    water_charge = (
                        ores_base
                        + (ores_t2_max - ores_t1_max) * ores_t2_rate
                        + (gallons - ores_t2_max) * ores_t3_rate
                    )

            elif wtr_rate == "OCOMM":
                if gallons <= ocomm_t1_max:
                    water_charge = ocomm_base
                elif gallons <= ocomm_t2_max:
                    water_charge = ocomm_base + (gallons - ocomm_t1_max) * ocomm_t2_rate
                else:
                    water_charge = (
                        ocomm_base
                        + (ocomm_t2_max - ocomm_t1_max) * ocomm_t2_rate
                        + (gallons - ocomm_t2_max) * ocomm_t3_rate
                    )

            else:
                return check_actual_dcrua(row)

            # SEWER (same as before)
            dcrua = clean_amt(row['DCRUA Amt'])
            if swr_rate in ["IRES", "ICOMM"]:
                if(sewer_multiplier_enable):
                    dcrua_charge =gallons * DCRUA_rate#add dynamic dcrua, min sewer charge, and sewer charge
                else:
                    dcrua_charge =gallons * DCRUA_rate
            elif swr_rate in ["ORES", "OCOMM"]:
                if(sewer_multiplier_enable):
                    dcrua_charge =gallons * DCRUA_rate
                else:
                    dcrua_charge =gallons * DCRUA_rate
            else:
                return check_actual_dcrua(row)

            return dcrua_charge

        return 0
    return _fn

@st.cache_data
def preprocess(df,ires_base,icomm_base,ores_base,ocomm_base, ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5, ires_tier1, ires_tier2, ICOMM_tier1, ICOMM_tier2, ORES_tier1, ORES_tier2, OCOMM_tier1, OCOMM_tier2, sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate):
    df = df.copy()
    df['Wtr Amt_clean'] = clean_amt(df['Wtr Amt'])
    df['Swr Amt_clean'] = clean_amt(df['Swr Amt'])
    df['DCRUA Amt_clean'] = clean_amt(df['DCRUA Amt'])

    df['Actual_Total_Bill'] = (
    df['Wtr Amt_clean'] +
    df['Swr Amt_clean']+
    df['DCRUA Amt_clean']
    )
    df = check_estimated_vectorized_final(df)



    #df.apply(check_actual, axis=1)
    #df['Estimated_Total_Bill'] = df.apply(check_estimated, axis=1)
    #call the refactored make_modified_fn with dynamic tier values

    df = compute_modified_bill(
    df,
    ires_base, icomm_base, ores_base, ocomm_base,
    ires_tier1, ires_tier2, ires_2_5, ires_5,
    ICOMM_tier1, ICOMM_tier2, icomm_2_5, icomm_5,
    ORES_tier1, ORES_tier2, ores_2_5, ores_5,
    OCOMM_tier1, OCOMM_tier2, ocomm_2_5, ocomm_5,
    sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate
)


    # Apply the function row-wise to your DataFrame
    #df['Modified_Total_Estimated_Bill'] = df.apply(modified_fn, axis=1)
    df['Actual_Estimated_Diff'] = df['Actual_Total_Bill'] - df['Estimated_Total_Bill']
    df['Relative_Error_%'] = (df['Actual_Estimated_Diff'] / df['Actual_Total_Bill']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

file = preprocess(raw,ires_base,icomm_base,ores_base,ocomm_base, ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5, ires_tier1, ires_tier2, ICOMM_tier1, ICOMM_tier2, ORES_tier1, ORES_tier2, OCOMM_tier1, OCOMM_tier2, sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate)

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

# # --- Revenue summary ---
# st.subheader("Revenue Summary")
# st.write(f"Actual Total Revenue: ${monthly_totals['Actual_Total_Bill'].sum():,.2f}")
# st.write(f"Actual Total Water Charges: ${file['Wtr Amt'].apply(clean_amt).sum():,.2f}")
# st.write(f"Actual Total Sewer Charges: ${file['Swr Amt'].apply(clean_amt).sum():,.2f}")
# st.write(f"Actual Total DCRUA Charges: ${file['DCRUA Amt'].apply(clean_amt).sum():,.2f}")
# st.divider()
# st.write(f"Estimated Total Revenue: ${monthly_totals['Estimated_Total_Bill'].sum():,.2f}")
# st.write(f"Estimated Total Water Charges: ${file.apply(get_water_rate_estimated, axis=1).sum():,.2f}")
# st.write(f"Estimated Total Sewer Charges: ${file.apply(get_sewer_rate_estimated, axis=1).sum():,.2f}")
# st.write(f"Estimated Total DCRUA Charges: ${file.apply(get_dcrua_rate_estimated, axis=1).sum():,.2f}")
# st.divider()
# st.write(f"Modified Total Revenue: ${monthly_totals['Modified_Total_Estimated_Bill'].sum():,.2f}")

# --- Revenue summary ---
st.subheader("Revenue Summary")

# Compute all totals first
actual_total_revenue = monthly_totals['Actual_Total_Bill'].sum()
actual_water = clean_amt(file['Wtr Amt']).sum()
actual_sewer = clean_amt(file['Swr Amt']).sum()
actual_dcrua = clean_amt(file['DCRUA Amt']).sum()
#estimated_total_revenue = file['Estimated_Total_Bill'].sum()
# estimated_total_revenue = file["Estimated_Total_Bill"] = (
#     file.apply(get_water_rate_estimated, axis=1).sum()
#     + file.apply(get_sewer_rate_estimated, axis=1).sum()
#     + file.apply(get_dcrua_rate_estimated, axis=1).sum()
# ).round(2)
#file['Estimated_Total_Bill'].sum()#monthly_totals['Estimated_Total_Bill'].sum()
estimated_water = get_water_rate_estimated_vectorized(file).sum()

#file.apply(get_water_rate_estimated, axis=1).sum()
estimated_sewer = get_sewer_rate_estimated_vectorized(file).sum()
#estimated_sewer = estimated_sewer['Estimated_Total_Bill'].sum()
#estimated_sewer = file.apply(get_sewer_rate_estimated, axis=1).sum()
estimated_dcrua = clean_amt(file['DCRUA Amt']).sum()#file.apply(get_dcrua_rate_estimated, axis=1).sum()
estimated_total_revenue = check_estimated_vectorized_final(file)["Estimated_Total_Bill"].sum()#estimated_sewer+estimated_water+estimated_dcrua#check_estimated_vectorized_final(file).sum()#estimated_water+estimated_sewer+estimated_dcrua
modified_total_revenue = monthly_totals['Modified_Total_Estimated_Bill'].sum()

# Create a dataframe for display
import pandas as pd



# st.write("---- DEBUG SUMMARY ----")
# st.write(f"Estimated total: ${total_estimated:,.2f}")
# st.write(f"Component sum: ${(water_est_sum + sewer_est_sum + dcrua_est_sum):,.2f}")
# st.write(f"Difference: ${(total_estimated - (water_est_sum + sewer_est_sum + dcrua_est_sum)):,.2f}")
# # Compare per-row totals
# file["est_sum_components"] = (
#     file.apply(get_water_rate_estimated, axis=1)
#     + file.apply(get_sewer_rate_estimated, axis=1)
#     + file.apply(get_dcrua_rate_estimated, axis=1)
# )

# file["diff_per_row"] = file["Estimated_Total_Bill"] - file["est_sum_components"]

# st.write("Mean diff per row:", file["diff_per_row"].mean())
# st.write("Max diff per row:", file["diff_per_row"].max())
# st.write("Min diff per row:", file["diff_per_row"].min())
# # Find the rows with the largest absolute difference
# outliers = file.loc[file["diff_per_row"].abs() > 10, [
#     "AccountNumber",  # or whatever identifies your customer
#     "Estimated_Total_Bill",
#     "est_sum_components",
#     "diff_per_row"
# ]]

# st.write("⚠️ Outlier rows (diff > $10):")
# st.dataframe(outliers)
 


modified_wtr_rate = get_modified_water_charge(
        ires_base=ires_base,
        icomm_base=icomm_base,
        ores_base=ores_base,
        ocomm_base=ocomm_base,
        
        ires_t1_max=ires_tier1,
        ires_t2_max=ires_tier2,
        ires_t2_rate=ires_2_5,
        ires_t3_rate=ires_5,
        
        icomm_t1_max=ICOMM_tier1,
        icomm_t2_max=ICOMM_tier2,
        icomm_t2_rate=icomm_2_5,
        icomm_t3_rate=icomm_5,
        
        ores_t1_max=ORES_tier1,
        ores_t2_max=ORES_tier2,
        ores_t2_rate=ores_2_5,
        ores_t3_rate=ores_5,
        
        ocomm_t1_max=OCOMM_tier1,
        ocomm_t2_max=OCOMM_tier2,
        ocomm_t2_rate=ocomm_2_5,
        ocomm_t3_rate=ocomm_5,
        base_sewer_rate = sewer_rate, 
        sewer_multiplier_enable = check_box_sewer_multiplier_enable, 
        sewer_multiplier = sewer_multiplier_rate, 
        DCRUA_base_rate = DCRUA_rate
        
    )
modified_sewer_rate = get_modified_sewer_charge(
        ires_base=ires_base,
        icomm_base=icomm_base,
        ores_base=ores_base,
        ocomm_base=ocomm_base,
        
        ires_t1_max=ires_tier1,
        ires_t2_max=ires_tier2,
        ires_t2_rate=ires_2_5,
        ires_t3_rate=ires_5,
        
        icomm_t1_max=ICOMM_tier1,
        icomm_t2_max=ICOMM_tier2,
        icomm_t2_rate=icomm_2_5,
        icomm_t3_rate=icomm_5,
        
        ores_t1_max=ORES_tier1,
        ores_t2_max=ORES_tier2,
        ores_t2_rate=ores_2_5,
        ores_t3_rate=ores_5,
        
        ocomm_t1_max=OCOMM_tier1,
        ocomm_t2_max=OCOMM_tier2,
        ocomm_t2_rate=ocomm_2_5,
        ocomm_t3_rate=ocomm_5,
        base_sewer_rate = sewer_rate, 
        sewer_multiplier_enable = check_box_sewer_multiplier_enable, 
        sewer_multiplier = sewer_multiplier_rate, 
        DCRUA_base_rate = DCRUA_rate
        
    )

modified_dcrua = get_modified_dcrua(
        ires_base=ires_base,
        icomm_base=icomm_base,
        ores_base=ores_base,
        ocomm_base=ocomm_base,
        
        ires_t1_max=ires_tier1,
        ires_t2_max=ires_tier2,
        ires_t2_rate=ires_2_5,
        ires_t3_rate=ires_5,
        
        icomm_t1_max=ICOMM_tier1,
        icomm_t2_max=ICOMM_tier2,
        icomm_t2_rate=icomm_2_5,
        icomm_t3_rate=icomm_5,
        
        ores_t1_max=ORES_tier1,
        ores_t2_max=ORES_tier2,
        ores_t2_rate=ores_2_5,
        ores_t3_rate=ores_5,
        
        ocomm_t1_max=OCOMM_tier1,
        ocomm_t2_max=OCOMM_tier2,
        ocomm_t2_rate=ocomm_2_5,
        ocomm_t3_rate=ocomm_5,
        base_sewer_rate = sewer_rate, 
        sewer_multiplier_enable = check_box_sewer_multiplier_enable, 
        sewer_multiplier = sewer_multiplier_rate, 
        DCRUA_base_rate = DCRUA_rate
        
    )
modified_water = file.apply(modified_wtr_rate, axis=1).sum()
modified_sewer_rate = file.apply(modified_sewer_rate, axis=1).sum()
modified_dcrua = file.apply(modified_dcrua, axis=1).sum()
modified_total_revenue = compute_modified_bill(
    file,
    ires_base, icomm_base, ores_base, ocomm_base,
    ires_tier1, ires_tier2, ires_2_5, ires_5,
    ICOMM_tier1, ICOMM_tier2, icomm_2_5, icomm_5,
    ORES_tier1, ORES_tier2, ores_2_5, ores_5,
    OCOMM_tier1, OCOMM_tier2, ocomm_2_5, ocomm_5,
    sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate
)['Modified_Total_Estimated_Bill'].sum()#(modified_water+modified_sewer_rate+modified_dcrua).round(2)
summary_table = pd.DataFrame({
    "Category": [
        "Total Revenue",
        "Water Charges",
        "Sewer Charges",
        "DCRUA Charges"
    ],
    "Actual": [
        actual_total_revenue,
        actual_water,
        actual_sewer,
        actual_dcrua
    ],
    "Estimated": [
        estimated_total_revenue,
        estimated_water,
        estimated_sewer,
        estimated_dcrua
    ],
    "Modified": [
        modified_total_revenue,
        modified_water,  # no breakdown provided for modified water/sewer/dcrua
        modified_sewer_rate,
        modified_dcrua
    ]
})

# Format numbers nicely with commas and two decimals
summary_table = summary_table.map(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)




st.markdown("""
Actual: Actual total revenue, water charges, sewer, and DCRUA charges are calculated by summing all of the customers bills.
            
Estimated: Estimated total revenue, water charges, and sewer charges are calculated using the amount of gallons used multiplied by the current water and sewer rates. DCRUA totals remain as the actual amount charged in this column. 

Modified: Modified total revenue, water charges, sewer charges, and DCRUA charges are calculated using the amount of gallons used multiplied by the proposed water, sewer, and DCRUA rates set by the user in the sidebar's input boxes
""")




# Display the table in Streamlit
st.table(summary_table)


st.divider()
# --- Differences ---
diff_est = actual_total_revenue - estimated_total_revenue
diff_mod = actual_total_revenue - modified_total_revenue
diff_est_reverse = estimated_total_revenue - actual_total_revenue
diff_mod_reverse = modified_total_revenue - actual_total_revenue

# --- Percent differences ---
pct_diff_est = (diff_est / actual_total_revenue) * 100
pct_diff_mod = (diff_mod / actual_total_revenue) * 100

# --- Build comparison table ---
diff_table = pd.DataFrame({
    "Comparison": [
        "Actual Total Revenue − Estimated Total Revenue",
        "Actual Total Revenue − Modified Total Revenue",
        "Estimated Total Revenue − Actual Total Revenue ",
        "Modified Total Revenue − Actual Total Revenue "
    ],
    "Dollar Difference": [
        diff_est,
        diff_mod,
        diff_est_reverse,
        diff_mod_reverse
    ],
    "Percent Difference": [
        pct_diff_est,
        pct_diff_mod,
        -pct_diff_est,
        -pct_diff_mod
    ]
})

# --- Format table for readability ---
diff_table = diff_table.map(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
diff_table["Percent Difference"] = diff_table["Percent Difference"].apply(
    lambda x: f"{float(x.strip('$')):.2f}%" if isinstance(x, str) else x
)

# --- Display ---
st.subheader("Revenue Difference Summary")
st.table(diff_table)

# --- Profits by Rate Class ---
st.subheader("Revenue by Water Rate Class (Actual Revenue)")
file['Wtr Rate'] = file['Wtr Rate'].str.strip()
# Group by water rate
profit_by_rate = file[(file['Wtr Rate']!='METER') & (file['Wtr Rate']!='125 MTR') & (file['Wtr Rate']!='FIREHYDR')].groupby('Wtr Rate')['Actual_Total_Bill'].sum()


# Define the desired order
desired_order = ['IRES', 'ICOMM', 'ORES', 'OCOMM', 'NONE']

# Reindex the Series to enforce that order
profit_by_rate = profit_by_rate.reindex(desired_order).fillna(0)


# Matplotlib Pie Chart
fig2, ax2 = plt.subplots()
ax2.pie(
    profit_by_rate, 
    labels=profit_by_rate.index, 
    autopct='%1.1f%%', 
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)
ax2.set_title("Profit Distribution by Water Rate Class")
st.pyplot(fig2)


# --- Profits by Rate Class ---
st.subheader("Revenue by Water Rate Class (Estimated Revenue)")
file['Wtr Rate'] = file['Wtr Rate'].str.strip()
# Group by water rate
profit_by_rate = file[(file['Wtr Rate']!='METER') & (file['Wtr Rate']!='125 MTR') & (file['Wtr Rate']!='FIREHYDR')].groupby('Wtr Rate')['Estimated_Total_Bill'].sum()


# Define the desired order
desired_order = ['IRES', 'ICOMM', 'ORES', 'OCOMM', 'NONE']

# Reindex the Series to enforce that order
profit_by_rate = profit_by_rate.reindex(desired_order).fillna(0)

# Matplotlib Pie Chart
fig2_5, ax2_5 = plt.subplots()
ax2_5.pie(
    profit_by_rate, 
    labels=profit_by_rate.index, 
    autopct='%1.1f%%', 
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)
ax2_5.set_title("Profit Distribution by Water Rate Class")
st.pyplot(fig2_5)


# --- Profits by Rate Class ---
st.subheader("Revenue by Water Rate Class (Modified Revenue)")
file['Wtr Rate'] = file['Wtr Rate'].str.strip()
# Group by water rate
profit_by_rate = file[(file['Wtr Rate']!='METER') & (file['Wtr Rate']!='125 MTR') & (file['Wtr Rate']!='FIREHYDR')].groupby('Wtr Rate')['Modified_Total_Estimated_Bill'].sum()
# Define the desired order
desired_order = ['IRES', 'ICOMM', 'ORES', 'OCOMM', 'NONE']

# Reindex the Series to enforce that order
profit_by_rate = profit_by_rate.reindex(desired_order).fillna(0)
# Matplotlib Pie Chart
fig3, ax3 = plt.subplots()
ax3.pie(
    profit_by_rate, 
    labels=profit_by_rate.index, 
    autopct='%1.1f%%', 
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)
ax3.set_title("Profit Distribution by Water Rate Class")
st.pyplot(fig3)





    
def usage_range_dynamic(g, t1, t2):
    """Generic usage tier assignment using dynamic thresholds."""
    if pd.isna(g):
        return None
    if g <= t1:
        return f"0–{t1}k"
    elif g <= t2:
        return f"{t1}–{t2}k"
    else:
        return f"{t2}k+"

def plot_usage_distribution(df, rate_class, title_prefix, t1, t2):
    """
    Filter df for a given water rate class, assign usage ranges (using dynamic tiers),
    and plot pie chart with legend.
    """
    subset = df[df['Wtr Rate'] == rate_class].copy()
    subset['UsageRange'] = subset['Billing Cons'].apply(lambda g: usage_range_dynamic(g, t1, t2))

    usage_totals = (
        subset.groupby("UsageRange")["Billing Cons"]
        .sum()
        .reindex([f"0–{t1}k", f"{t1}–{t2}k", f"{t2}k+"])
    )

    fig, ax = plt.subplots()
    wedges, _ = ax.pie(
        usage_totals,
        labels=None,
        autopct=None,
        startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
    )

    total = usage_totals.sum()
    legend_labels = [
        f"{label}: {value:,.0f} ({value/total:.1%})"
        for label, value in zip(usage_totals.index, usage_totals)
        if pd.notna(value)
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Usage Range",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    ax.set_title(f"{title_prefix} Water Usage Distribution by Usage Tiers")
    st.pyplot(fig)
def plot_revenue_distribution(df, rate_class, title_prefix, t1, t2, revenue_col="Actual_Total_Bill"):
    """
    Filter df for a given water rate class, assign usage ranges dynamically,
    and plot pie chart with legend showing REVENUE distribution.
    """
    subset = df[df['Wtr Rate'] == rate_class].copy()
    subset['UsageRange'] = subset['Billing Cons'].apply(lambda g: usage_range_dynamic(g, t1, t2))

    revenue_totals = (
        subset.groupby("UsageRange")[revenue_col]
        .sum()
        .reindex([f"0–{t1}k", f"{t1}–{t2}k", f"{t2}k+"])
    )

    fig, ax = plt.subplots()
    wedges, _ = ax.pie(
        revenue_totals,
        labels=None,
        autopct=None,
        startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
    )

    total = revenue_totals.sum()
    legend_labels = [
        f"{label}: ${value:,.0f} ({value/total:.1%})"
        for label, value in zip(revenue_totals.index, revenue_totals)
        if pd.notna(value)
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Usage Range",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    ax.set_title(f"{title_prefix} Revenue Distribution by Usage Tiers")
    st.pyplot(fig)





# Ensure Billing Cons numeric (thousands of gallons)
file['Billing Cons'] = pd.to_numeric(file['Billing Cons'].astype(str).str.replace(',',''), errors='coerce')



# Usage distributions (Actual Usage)
st.subheader("Water Usage Distributions by Usage Tier (Actual Usage + Dynamic Usage Tiers)")
plot_usage_distribution(file, "IRES",  "IRES",  ires_tier1, ires_tier2)
plot_usage_distribution(file, "ICOMM", "ICOMM", ICOMM_tier1, ICOMM_tier2)
plot_usage_distribution(file, "ORES",  "ORES",  ORES_tier1, ORES_tier2)
plot_usage_distribution(file, "OCOMM", "OCOMM", OCOMM_tier1, OCOMM_tier2)

# Revenue distributions (Actual Revenue)
st.subheader("Revenue Distributions by Usage Tier (Actual Revenue + Dynamic Usage Tiers)")
plot_revenue_distribution(file, "IRES",  "IRES",  ires_tier1, ires_tier2)
plot_revenue_distribution(file, "ICOMM", "ICOMM", ICOMM_tier1, ICOMM_tier2)
plot_revenue_distribution(file, "ORES",  "ORES",  ORES_tier1, ORES_tier2)
plot_revenue_distribution(file, "OCOMM", "OCOMM", OCOMM_tier1, OCOMM_tier2)

# --- Combined Distribution by Class + Usage ---
st.subheader("Revenue Distribution by Water Rate Class + Dynamic Usage Tiers")

# Apply usage categories for valid classes
valid_classes = ["IRES", "ORES", "ICOMM", "OCOMM"]
mask = file['Wtr Rate'].isin(valid_classes)

file.loc[mask, "UsageRange"] = file.loc[mask].apply(
    lambda row: usage_range_dynamic(
        row["Billing Cons"],
        ORES_tier1 if row["Wtr Rate"] == "ORES" else 
        OCOMM_tier1 if row["Wtr Rate"] == "OCOMM" else 
        ires_tier1 if row["Wtr Rate"] == "IRES" else 
        ICOMM_tier1,
        ORES_tier2 if row["Wtr Rate"] == "ORES" else 
        OCOMM_tier2 if row["Wtr Rate"] == "OCOMM" else 
        ires_tier2 if row["Wtr Rate"] == "IRES" else 
        ICOMM_tier2
    ),
    axis=1
)


# Build combined category
file.loc[mask, "Class+Usage"] = (
    file.loc[mask, "Wtr Rate"] + " " + file.loc[mask, "UsageRange"]
)

# --- Revenue by Class + Usage ---
revenue_by_class_usage = (
    file.loc[mask]
    .groupby("Class+Usage")["Actual_Total_Bill"]
    .sum()
    .sort_values(ascending=False)
)

# Pie chart
fig8, ax8 = plt.subplots()
wedges, _ = ax8.pie(
    revenue_by_class_usage,
    labels=None,
    autopct=None,
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)

# Legend with dollar values + percentages
total = revenue_by_class_usage.sum()
legend_labels = [
    f"{label}: ${value:,.0f} ({value/total:.1%})"
    for label, value in zip(revenue_by_class_usage.index, revenue_by_class_usage)
    if pd.notna(value)
]

ax8.legend(
    wedges,
    legend_labels,
    title="Class + Usage",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
ax8.set_title("Revenue Distribution by Class + Usage Tier")
st.pyplot(fig8)
# --- Usage by Class + Dynamic Usage Tiers ---
st.subheader("Water Usage Distribution by Water Rate Class + Dynamic Usage Tiers")

# Use the same mask and "Class+Usage" you already created
# Aggregate by usage instead of revenue
usage_by_class_usage = (
    file.loc[mask]
    .groupby("Class+Usage")["Billing Cons"]  # <- change here
    .sum()
    .sort_values(ascending=False)
)

# Pie chart for usage
fig_usage, ax_usage = plt.subplots()
wedges, _ = ax_usage.pie(
    usage_by_class_usage,
    labels=None,
    autopct=None,
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)

# Legend with values + percentages
total_usage = usage_by_class_usage.sum()
legend_labels = [
    f"{label}: {value:,.0f}k gallons ({value/total_usage:.1%})"
    for label, value in zip(usage_by_class_usage.index, usage_by_class_usage)
    if pd.notna(value)
]

ax_usage.legend(
    wedges,
    legend_labels,
    title="Class + Usage",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
ax_usage.set_title("Water Usage Distribution by Class + Usage Tier")
st.pyplot(fig_usage)

# --- Bar chart of revenue ---
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




# --- Usage by Class + Usage ---
usage_by_class_usage = (
    file.loc[mask]
    .groupby("Class+Usage")["Billing Cons"]
    .sum()
    .sort_values(ascending=False)
)

# Bar chart of usage
fig10, ax10 = plt.subplots(figsize=(10,6))
usage_by_class_usage.plot(
    kind="bar",
    ax=ax10,
    color="skyblue",
    edgecolor="black"
)
ax10.set_title("Usage by Class + Usage Tier")
ax10.set_xlabel("Class + Usage Tier")
ax10.set_ylabel("Usage Amount (Thousands of Gallons)")
ax10.set_xticklabels(usage_by_class_usage.index, rotation=45, ha="right")
st.pyplot(fig10)



st.divider()
st.header("Start of Section 2: Using Modified Revenue Define by User")

# --- Usage Distribution by Class (Using Modified Values) ---
st.subheader("Revenue Distributions by Usage Tier (Using Modified Revenue + Dynamic Usage Tiers)")
plot_revenue_distribution(file, "IRES",  "IRES",ires_tier1,ires_tier2, revenue_col="Modified_Total_Estimated_Bill")
plot_revenue_distribution(file, "ICOMM", "ICOMM",ICOMM_tier1,ICOMM_tier2, revenue_col="Modified_Total_Estimated_Bill")
plot_revenue_distribution(file, "ORES",  "ORES", ORES_tier1,ORES_tier2,revenue_col="Modified_Total_Estimated_Bill")
plot_revenue_distribution(file, "OCOMM", "OCOMM",OCOMM_tier1,OCOMM_tier2, revenue_col="Modified_Total_Estimated_Bill")



# --- Combined Distribution by Class + Usage ---
st.subheader("Revenue Distribution by Water Rate Class + Dynamic Usage Tiers")

# Apply usage categories for valid classes
valid_classes = ["IRES", "ORES", "ICOMM", "OCOMM"]
mask = file['Wtr Rate'].isin(valid_classes)

file.loc[mask, "UsageRange"] = file.loc[mask].apply(
    lambda row: usage_range_dynamic(
        row["Billing Cons"],
        ORES_tier1 if row["Wtr Rate"] == "ORES" else 
        OCOMM_tier1 if row["Wtr Rate"] == "OCOMM" else 
        ires_tier1 if row["Wtr Rate"] == "IRES" else 
        ICOMM_tier1,
        ORES_tier2 if row["Wtr Rate"] == "ORES" else 
        OCOMM_tier2 if row["Wtr Rate"] == "OCOMM" else 
        ires_tier2 if row["Wtr Rate"] == "IRES" else 
        ICOMM_tier2
    ),
    axis=1
)


# Build combined category
file.loc[mask, "Class+Usage"] = (
    file.loc[mask, "Wtr Rate"] + " " + file.loc[mask, "UsageRange"]
)

# --- Revenue by Class + Usage ---
revenue_by_class_usage = (
    file.loc[mask]
    .groupby("Class+Usage")["Modified_Total_Estimated_Bill"]
    .sum()
    .sort_values(ascending=False)
)

# Pie chart
fig8, ax8 = plt.subplots()
wedges, _ = ax8.pie(
    revenue_by_class_usage,
    labels=None,
    autopct=None,
    startangle=90,      # rotation of the first slice
    counterclock=False  # make slices go clockwise
)

# Legend with dollar values + percentages
total = revenue_by_class_usage.sum()
legend_labels = [
    f"{label}: ${value:,.0f} ({value/total:.1%})"
    for label, value in zip(revenue_by_class_usage.index, revenue_by_class_usage)
    if pd.notna(value)
]

ax8.legend(
    wedges,
    legend_labels,
    title="Class + Usage",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
ax8.set_title("Revenue Distribution by Class + Usage Tier")
st.pyplot(fig8) 

# --- Bar chart of revenue ---
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
