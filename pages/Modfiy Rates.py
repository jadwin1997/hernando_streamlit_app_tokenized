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
st.sidebar.header("Modify Water & Sewer Base Rates")#st.sidebar.slider("Inside City Residential (IRES) base price:", 0.00, 50.00, 12.50,.01, key='ires_base_price')#
ires_base = st.sidebar.number_input("Inside City Residential (IRES) base price:", value=12.50, key='ires_base_price')
icomm_base = st.sidebar.number_input("Inside City Commercial (ICOMM) base price:", value=12.50, key='icomm_base_price')
ores_base = st.sidebar.number_input("Outside City (ORES) base price:", value=16.00, key='ores_base_price')
ocomm_base = st.sidebar.number_input("Outside City (OCOMM) base price:", value=16.00, key='ocomm_base_price')

st.sidebar.header("Modify Water & Sewer Variable Rates")
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
sewer_rate = st.sidebar.number_input("Rate per 1000 gallons:", value=2, step=1, key='sewer_rate')
check_box_sewer_multiplier_enable = st.sidebar.checkbox("Enable Sewer Rate Multiplier", value=False, key='sewer_rate_multiplier_enable')
sewer_multiplier_rate = st.sidebar.number_input("Multiple of Water Charge:", value=2, step=1, key='sewer_multiplier_rate', label_visibility="visible")
DCRUA_rate = st.sidebar.number_input("DCRUA Rate (unit):", value=2, step=1, key='DCRUA_rate')

#sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate

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


def make_modified_fn(
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
                return check_actual(row)

            # SEWER (same as before)
            dcrua = clean_amt(row['DCRUA Amt'])
            if swr_rate in ["IRES", "ICOMM"]:
                sewer_charge = max(water_charge * sewer_multiplier_rate, 6.25) + dcrua#add dynamic dcrua, min sewer charge, and sewer charge
            elif swr_rate in ["ORES", "OCOMM"]:
                sewer_charge = max(water_charge * sewer_multiplier_rate, 8.00) + dcrua
            else:
                return check_actual(row)

            return round(water_charge + sewer_charge, 2)

        return 0
    return _fn

@st.cache_data
def preprocess(df,ires_base,icomm_base,ores_base,ocomm_base, ires_2_5, ires_5, ores_2_5, ores_5, icomm_2_5, icomm_5, ocomm_2_5, ocomm_5, ires_tier1, ires_tier2, ICOMM_tier1, ICOMM_tier2, ORES_tier1, ORES_tier2, OCOMM_tier1, OCOMM_tier2, sewer_rate, check_box_sewer_multiplier_enable, sewer_multiplier_rate, DCRUA_rate):
    df = df.copy()
    df['Actual_Total_Bill'] = df.apply(check_actual, axis=1)
    df['Estimated_Total_Bill'] = df.apply(check_estimated, axis=1)
    #call the refactored make_modified_fn with dynamic tier values
    modified_fn = make_modified_fn(
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

    # Apply the function row-wise to your DataFrame
    df['Modified_Total_Estimated_Bill'] = df.apply(modified_fn, axis=1)
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
