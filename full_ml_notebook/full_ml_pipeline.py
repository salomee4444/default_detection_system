import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Run this in a new cell for complete compatibility
!pip install "pandas>=2.1.0" "numpy>=1.24.0,<2.0" --force-reinstall

# Check current versions and compatibility
import numpy as np
import pandas as pd

print("🔍 Current Environment:")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Check if numpy.matrix exists
if hasattr(np, 'matrix'):
    print("✅ numpy.matrix: Available")
else:
    print("❌ numpy.matrix: NOT AVAILABLE (NumPy 2.0+)")

# Test basic pandas operations
try:
    test_series = pd.Series([1, 2, 3])
    test_result = test_series.reset_index()
    print("✅ Pandas reset_index: Working")
except Exception as e:
    print(f"❌ Pandas reset_index: FAILED - {str(e)}")

!pip install kagglehub
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mishra5001/credit-card")

print("Path to dataset files:", path)

app_df = pd.read_csv(path+"/application_data.csv")
app_df.head()

app_df.info()

prev_app_df = pd.read_csv(path+"/previous_application.csv")
prev_app_df.head()

merged_df = app_df.merge(prev_app_df, on='SK_ID_CURR', how='inner')
for col in merged_df.columns:
    if merged_df[col].dtype in ['float64', 'object']:
        if merged_df[col].dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
            merged_df[col] = merged_df[col].astype(int, errors='ignore')
merged_df.info()
merged_df.head()

merged_df.dtypes

transactions = merged_df[['SK_ID_CURR', 'SK_ID_PREV']].copy()
transactions = transactions.rename(columns={'SK_ID_PREV': 'SK_ID_RECEIVER'})

class UnionFind:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, element):
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element]) # Path compression
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self.find(element1)
        root2 = self.find(element2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

unique_accounts = pd.concat([transactions['SK_ID_CURR'], transactions['SK_ID_RECEIVER']]).unique()
uf = UnionFind(unique_accounts)

for index, row in transactions.iterrows():
    uf.union(row['SK_ID_CURR'], row['SK_ID_RECEIVER'])

merged_df.T

def drop_high_null_features(merged_df, threshold=0.4):
    """
    Drops all columns with more than 'threshold' proportion of null values.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Fraction of null values above which columns are dropped (default = 0.4)
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns removed
    """
    # Calculate the fraction of nulls per column
    null_fraction = merged_df.isnull().mean()
    
    # Columns to drop
    cols_to_drop = null_fraction[null_fraction > threshold].index
    
    print(f"🧹 Dropping {len(cols_to_drop)} columns with more than {threshold*100}% null values")
    
    # Drop and return
    df_cleaned = merged_df.drop(columns=cols_to_drop)
    return df_cleaned

output_path = "remaining_columns.txt"

with open(output_path, "w") as f:
    for col in merged_df_clean.columns:
        f.write(col + "\n")

print(f"✅ Saved {len(merged_df_clean.columns)} column names to '{output_path}'")

def display_unique_value_counts(df, sort_by_count=False):
    """
    Displays the count of unique values for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze
        sort_by_count (bool): If True, sorts by number of unique values (descending)

    Returns:
        pd.DataFrame: A summary table of column names and their unique counts
    """
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ['Column', 'Unique_Values']

    if sort_by_count:
        unique_counts = unique_counts.sort_values(by='Unique_Values', ascending=False)

    print(f"📊 Total columns analyzed: {len(df.columns)}")
    display(unique_counts)  # works well in Jupyter/Colab

    return unique_counts

unique_summary = display_unique_value_counts(merged_df_clean, sort_by_count=True)

columns_description = {
    "SK_ID_CURR": "ID of loan in our sample",
    "TARGET": "Target variable (1 - client with payment difficulties; 0 - all other cases)",
    "NAME_CONTRACT_TYPE_x": "Identification if loan is cash or revolving (current application)",
    "CODE_GENDER": "Gender of the client",
    "FLAG_OWN_CAR": "Flag if the client owns a car",
    "FLAG_OWN_REALTY": "Flag if client owns a house or flat",
    "CNT_CHILDREN": "Number of children the client has",
    "AMT_INCOME_TOTAL": "Income of the client",
    "AMT_CREDIT_x": "Credit amount of the loan (current application)",
    "AMT_ANNUITY_x": "Loan annuity (current application)",
    "AMT_GOODS_PRICE_x": "Price of goods for which the loan is given (current application)",
    "NAME_TYPE_SUITE_x": "Who was accompanying client when applying for the current loan",
    "NAME_INCOME_TYPE": "Client’s income type (businessman, working, maternity leave, etc.)",
    "NAME_EDUCATION_TYPE": "Level of highest education the client achieved",
    "NAME_FAMILY_STATUS": "Family status of the client",
    "NAME_HOUSING_TYPE": "Housing situation of the client (renting, living with parents, etc.)",
    "REGION_POPULATION_RELATIVE": "Normalized population of region where client lives (higher = more populated)",
    "DAYS_BIRTH": "Client's age in days at the time of application",
    "DAYS_EMPLOYED": "Days before application the person started current employment",
    "DAYS_REGISTRATION": "Days before application client changed registration",
    "DAYS_ID_PUBLISH": "Days before application client changed the identity document used for the loan",
    "FLAG_MOBIL": "Did client provide mobile phone (1=YES, 0=NO)",
    "FLAG_EMP_PHONE": "Did client provide work phone (1=YES, 0=NO)",
    "FLAG_WORK_PHONE": "Did client provide home phone (1=YES, 0=NO)",
    "FLAG_CONT_MOBILE": "Was mobile phone reachable (1=YES, 0=NO)",
    "FLAG_PHONE": "Did client provide home phone (1=YES, 0=NO)",
    "FLAG_EMAIL": "Did client provide email (1=YES, 0=NO)",
    "OCCUPATION_TYPE": "Kind of occupation the client has",
    "CNT_FAM_MEMBERS": "How many family members the client has",
    "REGION_RATING_CLIENT": "Rating of the region where client lives (1,2,3)",
    "REGION_RATING_CLIENT_W_CITY": "Rating of the region where client lives with city taken into account (1,2,3)",
    "WEEKDAY_APPR_PROCESS_START_x": "Day of the week when client applied for current loan",
    "HOUR_APPR_PROCESS_START_x": "Hour when client applied for current loan (rounded)",
    "REG_REGION_NOT_LIVE_REGION": "Flag if permanent address ≠ contact address (region level)",
    "REG_REGION_NOT_WORK_REGION": "Flag if permanent address ≠ work address (region level)",
    "LIVE_REGION_NOT_WORK_REGION": "Flag if contact address ≠ work address (region level)",
    "REG_CITY_NOT_LIVE_CITY": "Flag if permanent address ≠ contact address (city level)",
    "REG_CITY_NOT_WORK_CITY": "Flag if permanent address ≠ work address (city level)",
    "LIVE_CITY_NOT_WORK_CITY": "Flag if contact address ≠ work address (city level)",
    "ORGANIZATION_TYPE": "Type of organization where client works",
    "EXT_SOURCE_2": "Normalized score from external data source",
    "EXT_SOURCE_3": "Normalized score from external data source",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Number of observations of client's social circle with 30 DPD (days past due) default",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Number of client’s social circle defaults on 30 DPD",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Number of observations of client's social circle with 60 DPD default",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Number of client’s social circle defaults on 60 DPD",
    "DAYS_LAST_PHONE_CHANGE": "Days before application client changed phone",
    "FLAG_DOCUMENT_2": "Did client provide document 2",
    "FLAG_DOCUMENT_3": "Did client provide document 3",
    "FLAG_DOCUMENT_4": "Did client provide document 4",
    "FLAG_DOCUMENT_5": "Did client provide document 5",
    "FLAG_DOCUMENT_6": "Did client provide document 6",
    "FLAG_DOCUMENT_7": "Did client provide document 7",
    "FLAG_DOCUMENT_8": "Did client provide document 8",
    "FLAG_DOCUMENT_9": "Did client provide document 9",
    "FLAG_DOCUMENT_10": "Did client provide document 10",
    "FLAG_DOCUMENT_11": "Did client provide document 11",
    "FLAG_DOCUMENT_12": "Did client provide document 12",
    "FLAG_DOCUMENT_13": "Did client provide document 13",
    "FLAG_DOCUMENT_14": "Did client provide document 14",
    "FLAG_DOCUMENT_15": "Did client provide document 15",
    "FLAG_DOCUMENT_16": "Did client provide document 16",
    "FLAG_DOCUMENT_17": "Did client provide document 17",
    "FLAG_DOCUMENT_18": "Did client provide document 18",
    "FLAG_DOCUMENT_19": "Did client provide document 19",
    "FLAG_DOCUMENT_20": "Did client provide document 20",
    "FLAG_DOCUMENT_21": "Did client provide document 21",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Number of credit bureau enquiries one hour before application",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Number of credit bureau enquiries one day before application (excluding last hour)",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Number of credit bureau enquiries one week before application (excluding last day)",
    "AMT_REQ_CREDIT_BUREAU_MON": "Number of credit bureau enquiries one month before application (excluding last week)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Number of credit bureau enquiries three months before application (excluding last month)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Number of credit bureau enquiries one year before application (excluding last three months)",
    "SK_ID_PREV": "ID of previous credit related to loan in our sample",
    "NAME_CONTRACT_TYPE_y": "Contract product type (Cash loan, consumer loan, etc.) of previous application",
    "AMT_ANNUITY_y": "Annuity of previous application",
    "AMT_APPLICATION": "Amount of credit requested on previous application",
    "AMT_CREDIT_y": "Final credit amount granted on previous application",
    "AMT_GOODS_PRICE_y": "Goods price in previous application",
    "WEEKDAY_APPR_PROCESS_START_y": "Day of the week of previous application",
    "HOUR_APPR_PROCESS_START_y": "Hour of previous application (rounded)",
    "FLAG_LAST_APPL_PER_CONTRACT": "Flag if it was last application for the previous contract",
    "NFLAG_LAST_APPL_IN_DAY": "Flag if the application was the last of the day for the client",
    "NAME_CASH_LOAN_PURPOSE": "Purpose of the cash loan in previous application",
    "NAME_CONTRACT_STATUS": "Contract status (approved, cancelled, etc.) of previous application",
    "DAYS_DECISION": "Days relative to current application when previous application decision was made",
    "NAME_PAYMENT_TYPE": "Payment method chosen for previous application",
    "CODE_REJECT_REASON": "Reason why the previous application was rejected",
    "NAME_CLIENT_TYPE": "Old or new client flag for previous application",
    "NAME_GOODS_CATEGORY": "Type of goods applied for in previous application",
    "NAME_PORTFOLIO": "Portfolio type of previous application (CASH, POS, CAR, etc.)",
    "NAME_PRODUCT_TYPE": "Product type (x-sell or walk-in) of previous application",
    "CHANNEL_TYPE": "Channel through which client applied for previous application",
    "SELLERPLACE_AREA": "Selling area of seller place for previous application",
    "NAME_SELLER_INDUSTRY": "Industry of the seller in previous application",
    "CNT_PAYMENT": "Term of previous credit at application time",
    "NAME_YIELD_GROUP": "Grouped interest rate into small, medium, and high for previous application",
    "PRODUCT_COMBINATION": "Detailed product combination of previous application",
    "DAYS_FIRST_DRAWING": "Days relative to current application when first disbursement occurred (previous application)",
    "DAYS_FIRST_DUE": "Days relative to current application when first due date was scheduled (previous application)",
    "DAYS_LAST_DUE_1ST_VERSION": "Days relative to current application when first due date occurred (previous application)",
    "DAYS_LAST_DUE": "Days relative to current application when last due date occurred (previous application)",
    "DAYS_TERMINATION": "Days relative to current application when expected termination occurred (previous application)",
    "NFLAG_INSURED_ON_APPROVAL": "Flag if client requested insurance during previous application",
    "community_id": "Synthetic or external community identifier used for grouping or linking records"
}

def find_non_numeric_binaries(df: pd.DataFrame, exclude_bools: bool = True):
    """
    Find columns with exactly two unique non-null values where those values
    are NOT equivalent to {0,1}. Useful to spot string-binary fields that
    need categorical encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    exclude_bools : bool, default True
        If True, columns with boolean {True, False} are treated as binary 0/1 and excluded.

    Returns
    -------
    cols : list[str]
        List of column names that have exactly two unique non-null values and are not {0,1}.
    details : dict[str, list]
        Mapping of column -> its two unique values (as seen in the data).
    """
    cols = []
    details = {}

    for col in df.columns:
        # unique non-null values
        uniq = pd.unique(df[col].dropna())

        # must be exactly two unique values
        if len(uniq) != 2:
            continue

        # normalize values to detect 0/1 forms
        norm = set()
        for v in uniq:
            # Booleans
            if isinstance(v, (bool, np.bool_)):
                if exclude_bools:
                    norm.add(int(v))  # True->1, False->0
                else:
                    norm.add(v)
                continue

            # Numeric (ints/floats)
            if isinstance(v, (int, np.integer, float, np.floating)):
                # treat 0.0/1.0 as 0/1
                if float(v).is_integer() and int(v) in (0, 1):
                    norm.add(int(v))
                else:
                    norm.add(v)
                continue

            # Strings: check if '0' or '1'
            if isinstance(v, str):
                s = v.strip()
                if s in {"0", "1"}:
                    norm.add(int(s))
                else:
                    norm.add(s.lower())  # case-insensitive compare
                continue

            # Fallback
            norm.add(v)

        # If normalized set equals {0,1}, treat as proper binary → skip
        if norm == {0, 1}:
            continue

        # Otherwise this is a two-level non 0/1 field → flag it
        cols.append(col)
        details[col] = list(uniq)

    return cols, details

cols_to_encode, sample_values = find_non_numeric_binaries(merged_df_clean)

print("Columns with exactly two unique values that are NOT 0/1:")
print(cols_to_encode)
print("\nSample values per column:")
for c, vals in sample_values.items():
    print(f"{c}: {vals}")

def encode_selected_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode specific binary and categorical features into numeric model-ready form.
    Returns a new DataFrame with the encoded columns.
    """
    df = df.copy()

    # 1️⃣ NAME_CONTRACT_TYPE_x → is_CashLoan
    df["is_CashLoan"] = df["NAME_CONTRACT_TYPE_x"].map({
        "Cash loans": 1,
        "Revolving loans": 0
    })

    # 2️⃣ FLAG_OWN_CAR → 1/0
    df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map({
        "Y": 1,
        "N": 0
    })

    # 3️⃣ FLAG_OWN_REALTY → 1/0
    df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({
        "Y": 1,
        "N": 0
    })

    # 4️⃣ FLAG_LAST_APPL_PER_CONTRACT → 1/0
    df["FLAG_LAST_APPL_PER_CONTRACT"] = df["FLAG_LAST_APPL_PER_CONTRACT"].map({
        "Y": 1,
        "N": 0
    })

    # 5️⃣ CODE_GENDER → M=1, F=0, XNA=0.5
    df["CODE_GENDER"] = df["CODE_GENDER"].map({
        "M": 1,
        "F": 0,
        "XNA": 0.5
    })

    # Handle missing or unseen values (keep numeric consistency)
    df["is_CashLoan"] = df["is_CashLoan"].fillna(0)
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_LAST_APPL_PER_CONTRACT", "CODE_GENDER"]:
        df[col] = df[col].fillna(0)

    return df

merged_df_clean = encode_selected_binaries(merged_df_clean)
# Confirm encoding
merged_df_clean[[
    "is_CashLoan",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "FLAG_LAST_APPL_PER_CONTRACT",
    "CODE_GENDER"
]].head()

#save to csv
merged_df_clean.to_csv("merged_clean.csv", index=False)

fig = px.histogram(merged_df_clean, x='TARGET', color='TARGET', 
                   title="Distribution of TARGET (Default vs Non-Default)",
                   text_auto=True)
fig.update_layout(bargap=0.2)
fig.show()

fig = px.histogram(merged_df_clean, x='AMT_CREDIT_x', nbins=50, color='TARGET',
                   title="Loan Credit Amount Distribution by TARGET", marginal="box")
fig.show()

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for the credit default dataset with domain-aware rules.
    Returns a new DataFrame.
    """
    df = df.copy()

    # ---- 1) Social-circle features: fill 0 + missing flags ----
    social_circle = [
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE'
    ]
    for col in social_circle:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)

    # ---- 2) Credit bureau enquiry features: fill 0 + missing flags ----
    bureau = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    for col in bureau:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)

    # ---- 3) Previous-application timeline features: fill 0 + missing flags ----
    prev_timeline = [
        'DAYS_FIRST_DRAWING',
        'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE',
        'DAYS_TERMINATION'
    ]
    for col in prev_timeline:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)

    # ---- 4) Previous-application financials: fill 0 + missing flags ----
    prev_financial_zero = [
        'AMT_GOODS_PRICE_y',
        'CNT_PAYMENT',
        'AMT_ANNUITY_y'
    ]
    for col in prev_financial_zero:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)

    # ---- 5) Insurance flag on approval: fill 0 + missing flag ----
    if 'NFLAG_INSURED_ON_APPROVAL' in df.columns:
        df['NFLAG_INSURED_ON_APPROVAL_missing'] = df['NFLAG_INSURED_ON_APPROVAL'].isnull().astype(int)
        df['NFLAG_INSURED_ON_APPROVAL'] = df['NFLAG_INSURED_ON_APPROVAL'].fillna(0)

    # ---- 6) External scores: fill median + missing flags ----
    for col in ['EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # ---- 7) Current-application numeric: fill median + flags where useful ----
    # AMT_GOODS_PRICE_x tends to be 0 for cash loans; use median + flag
    if 'AMT_GOODS_PRICE_x' in df.columns:
        df['AMT_GOODS_PRICE_x_missing'] = df['AMT_GOODS_PRICE_x'].isnull().astype(int)
        df['AMT_GOODS_PRICE_x'] = df['AMT_GOODS_PRICE_x'].fillna(df['AMT_GOODS_PRICE_x'].median())

    if 'AMT_ANNUITY_x' in df.columns:
        df['AMT_ANNUITY_x_missing'] = df['AMT_ANNUITY_x'].isnull().astype(int)
        df['AMT_ANNUITY_x'] = df['AMT_ANNUITY_x'].fillna(df['AMT_ANNUITY_x'].median())

    # ---- 8) Categoricals: fill with 'Unknown' ----
    categorical_unknown = [
        'NAME_TYPE_SUITE_x',
        'PRODUCT_COMBINATION',
        'OCCUPATION_TYPE'
    ]
    for col in categorical_unknown:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    return df

def save_remaining_features(csv_path="merged_clean.csv", output_path="remaining_columns.txt"):
    """
    Reads a CSV file, extracts all column names (features),
    saves them to a text file (one per line),
    and returns the list of features.
    """
    # Read just the header (no need to load all data)
    df = pd.read_csv(csv_path, nrows=0)
    
    # Extract feature names
    features = df.columns.tolist()
    
    # Save to a text file
    with open(output_path, "w") as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    print(f"✅ Saved {len(features)} feature names to '{output_path}'")
    return features


# Example usage
features = save_remaining_features()

def add_ratio_features(df):
    """
    Adds ratio-based financial features to the dataframe.
    """
    df = df.copy()

    # Avoid division by zero or NaN issues
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['AMT_GOODS_PRICE_x'] = df['AMT_GOODS_PRICE_x'].replace(0, np.nan)
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df['AMT_CREDIT_x'] = df['AMT_CREDIT_x'].replace(0, np.nan)

    # 1️⃣ CREDIT_INCOME_RATIO = AMT_CREDIT_x / AMT_INCOME_TOTAL
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT_x'] / df['AMT_INCOME_TOTAL']

    # 2️⃣ ANNUITY_INCOME_RATIO = AMT_ANNUITY_x / AMT_INCOME_TOTAL
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY_x'] / df['AMT_INCOME_TOTAL']

    # 3️⃣ CREDIT_GOODS_RATIO = AMT_CREDIT_x / AMT_GOODS_PRICE_x
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT_x'] / df['AMT_GOODS_PRICE_x']

    # 4️⃣ INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # 5️⃣ CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # 6️⃣ PAYMENT_RATE = AMT_ANNUITY_x / AMT_CREDIT_x
    df['PAYMENT_RATE'] = df['AMT_ANNUITY_x'] / df['AMT_CREDIT_x']

    # Replace inf values with NaN (in case of divisions by zero)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Added ratio-based financial features successfully.")
    return df

# Apply the function
merged_df_clean = add_ratio_features(merged_df_clean)

def add_time_features(df, placeholder_value=365243):
    """
    Adds time/age features derived from DAYS_* columns:
      - AGE_YEARS
      - EMPLOYED_YEARS
      - REGISTRATION_YEARS_AGO
      - ID_PUBLISH_YEARS_AGO
      - PHONE_CHANGE_YEARS_AGO
      - EMPLOYMENT_TO_AGE_RATIO
    Treats 365243 in DAYS_* as missing (NaN), per Home Credit convention.
    """
    df = df.copy()

    days_cols = [
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE'
    ]

    # Ensure columns exist; if not, create as NaN so code is safe
    for c in days_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Replace placeholder 365243 with NaN
    for c in days_cols:
        df[c] = df[c].replace(placeholder_value, np.nan)

    # Convert to years (note: DAYS_* are negative "days before today")
    to_years = lambda s: (-s / 365.0)

    df['AGE_YEARS'] = to_years(df['DAYS_BIRTH']).round(1)

    df['EMPLOYED_YEARS'] = to_years(df['DAYS_EMPLOYED']).clip(lower=0)

    df['REGISTRATION_YEARS_AGO'] = to_years(df['DAYS_REGISTRATION']).clip(lower=0)

    df['ID_PUBLISH_YEARS_AGO'] = to_years(df['DAYS_ID_PUBLISH']).clip(lower=0)

    df['PHONE_CHANGE_YEARS_AGO'] = to_years(df['DAYS_LAST_PHONE_CHANGE']).clip(lower=0)

    # Ratio: employment stability relative to age
    df['EMPLOYMENT_TO_AGE_RATIO'] = (
        df['EMPLOYED_YEARS'] / df['AGE_YEARS'].replace(0, np.nan)
    )

    # Clean infinities if any
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Added time-based / age-related features.")
    return df

# Apply to your dataframe
merged_df_clean = add_time_features(merged_df_clean)

def add_behavioral_features(df):
    """
    Adds behavioral and stability indicator features to the dataframe.
    """
    df = df.copy()

    # 1️⃣ NUM_ACTIVE_CONTACTS = FLAG_MOBIL + FLAG_PHONE + FLAG_EMAIL + FLAG_WORK_PHONE
    df['NUM_ACTIVE_CONTACTS'] = (
        df[['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_WORK_PHONE']]
        .fillna(0)
        .sum(axis=1)
    )

    # 2️⃣ HAS_WORK_CONTACT = 1 if FLAG_WORK_PHONE == 1 or FLAG_EMP_PHONE == 1 else 0
    df['HAS_WORK_CONTACT'] = np.where(
        (df['FLAG_WORK_PHONE'] == 1) | (df['FLAG_EMP_PHONE'] == 1),
        1,
        0
    )

    # 3️⃣ HAS_ALL_DOCS = sum(FLAG_DOCUMENT_2 ... FLAG_DOCUMENT_21)
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    df['HAS_ALL_DOCS'] = df[doc_cols].fillna(0).sum(axis=1)

    # 4️⃣ STABILITY_SCORE = (FLAG_CONT_MOBILE + FLAG_PHONE + FLAG_EMAIL) / 3
    df['STABILITY_SCORE'] = (
        df[['FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']]
        .fillna(0)
        .mean(axis=1)
    )

    print("✅ Added behavioral & stability indicator features successfully.")
    return df


# Apply the function
merged_df_clean = add_behavioral_features(merged_df_clean)

def add_regional_features(df, urban_threshold=0.05):
    """
    Adds:
      - URBAN_RURAL (1 = urban if REGION_POPULATION_RELATIVE > threshold else 0)
      - CITY_REGION_MISMATCH_SCORE (sum of region/city mismatch flags)
    """
    df = df.copy()

    # --- URBAN_RURAL ---
    # Treat missing REGION_POPULATION_RELATIVE as 0 (conservative = non-urban)
    rel_pop = df['REGION_POPULATION_RELATIVE'].fillna(0)
    df['URBAN_RURAL'] = (rel_pop > urban_threshold).astype(int)

    # --- CITY_REGION_MISMATCH_SCORE ---
    mismatch_cols = [
        'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY'
    ]
    # Ensure columns exist and are numeric; missing -> 0
    existing = [c for c in mismatch_cols if c in df.columns]
    if len(existing) != len(mismatch_cols):
        missing = set(mismatch_cols) - set(existing)
        # create any missing columns as zeros so the sum works
        for m in missing:
            df[m] = 0

    df['CITY_REGION_MISMATCH_SCORE'] = (
        df[mismatch_cols]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .sum(axis=1)
        .astype(int)
    )

    print("✅ Added regional & demographic indicator features.")
    return df

# Apply to your dataframe
merged_df_clean = add_regional_features(merged_df_clean, urban_threshold=0.05)

def add_bureau_features(df):
    """
    Adds:
      - BUREAU_QUERY_INTENSITY
      - SHORT_TERM_BUREAU_RATIO
    Handles missing columns by creating them as zeros and avoids div-by-zero.
    """
    df = df.copy()

    long_cols  = ['AMT_REQ_CREDIT_BUREAU_YEAR',
                  'AMT_REQ_CREDIT_BUREAU_QRT',
                  'AMT_REQ_CREDIT_BUREAU_MON']
    short_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR',
                  'AMT_REQ_CREDIT_BUREAU_DAY',
                  'AMT_REQ_CREDIT_BUREAU_WEEK']

    # Ensure all columns exist
    for c in long_cols + short_cols:
        if c not in df.columns:
            df[c] = 0

    # Cast to numeric and fill NaNs with 0 for counts
    df[long_cols + short_cols] = df[long_cols + short_cols].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)

    # 1) How frequently the client's credit is checked (longer horizons)
    df['BUREAU_QUERY_INTENSITY'] = (
        df['AMT_REQ_CREDIT_BUREAU_YEAR']
        + df['AMT_REQ_CREDIT_BUREAU_QRT']
        + df['AMT_REQ_CREDIT_BUREAU_MON']
    )

    # 2) Short-term vs long-term request ratio
    short_sum = (
        df['AMT_REQ_CREDIT_BUREAU_HOUR']
        + df['AMT_REQ_CREDIT_BUREAU_DAY']
        + df['AMT_REQ_CREDIT_BUREAU_WEEK']
    )
    denom = (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1)  # +1 to stabilize zeros
    df['SHORT_TERM_BUREAU_RATIO'] = short_sum / denom

    # Clean any accidental infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Added bureau-based features.")
    return df

# Apply to your dataframe
merged_df_clean = add_bureau_features(merged_df_clean)

def add_application_pair_features(df):
    """
    Adds features comparing current (_x) vs previous (_y) application fields:
      - CREDIT_DIFF
      - ANNUITY_DIFF
      - GOODS_PRICE_DIFF
      - CREDIT_TO_GOODS_DELTA_RATIO
      - SAME_CONTRACT_TYPE
      - SAME_WEEKDAY_APPR
      - HOUR_APPR_DIFF
    All numeric inputs coerced; missing handled safely.
    """
    df = df.copy()

    num_cols = [
        'AMT_CREDIT_x','AMT_CREDIT_y',
        'AMT_ANNUITY_x','AMT_ANNUITY_y',
        'AMT_GOODS_PRICE_x','AMT_GOODS_PRICE_y',
        'HOUR_APPR_PROCESS_START_x','HOUR_APPR_PROCESS_START_y'
    ]
    # Create any missing numeric columns, then coerce to numeric
    for c in num_cols:
        if c not in df.columns: df[c] = np.nan
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # 1) Simple deltas
    df['CREDIT_DIFF']  = df['AMT_CREDIT_y']  - df['AMT_CREDIT_x']
    df['ANNUITY_DIFF'] = df['AMT_ANNUITY_y'] - df['AMT_ANNUITY_x']
    df['GOODS_PRICE_DIFF'] = df['AMT_GOODS_PRICE_y'] - df['AMT_GOODS_PRICE_x']

    # 2) Scale-aware delta: credit change relative to current goods price
    denom = df['AMT_GOODS_PRICE_x'].replace(0, np.nan)
    df['CREDIT_TO_GOODS_DELTA_RATIO'] = df['CREDIT_DIFF'] / denom

    # 3) Consistency flags for categorical fields
    for c in ['NAME_CONTRACT_TYPE_x','NAME_CONTRACT_TYPE_y',
              'WEEKDAY_APPR_PROCESS_START_x','WEEKDAY_APPR_PROCESS_START_y']:
        if c not in df.columns: df[c] = np.nan

    df['SAME_CONTRACT_TYPE'] = (
        (df['NAME_CONTRACT_TYPE_x'].astype('string')
         .str.strip()
         .fillna(''))
        .eq(
        df['NAME_CONTRACT_TYPE_y'].astype('string')
         .str.strip()
         .fillna(''))
    ).astype(int)

    df['SAME_WEEKDAY_APPR'] = (
        (df['WEEKDAY_APPR_PROCESS_START_x'].astype('string').str.strip().fillna(''))
        .eq(df['WEEKDAY_APPR_PROCESS_START_y'].astype('string').str.strip().fillna(''))
    ).astype(int)

    # 4) Hour-of-day difference (previous - current)
    df['HOUR_APPR_DIFF'] = df['HOUR_APPR_PROCESS_START_y'] - df['HOUR_APPR_PROCESS_START_x']

    # Clean infinities if any
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Added application pair features (x vs y).")
    return df

# Apply
merged_df_clean = add_application_pair_features(merged_df_clean)

def add_loan_lifecycle_features(df, placeholder_value=365243):
    """
    Adds previous-credit lifecycle features:
      - CREDIT_DURATION_DAYS          = DAYS_LAST_DUE - DAYS_FIRST_DUE
      - TIME_TO_FIRST_PAYMENT_DAYS    = DAYS_FIRST_DUE - DAYS_FIRST_DRAWING
      - TIME_TO_TERMINATION_DAYS      = DAYS_TERMINATION - DAYS_DECISION
      - OVERLAP_WITH_CURRENT          = 1 if DAYS_TERMINATION > DAYS_DECISION else 0

    Notes:
    - Replaces 365243 placeholders with NaN (Home Credit convention).
    - Durations are clipped to be >= 0.
    - If DAYS_LAST_DUE is absent, uses DAYS_LAST_DUE_1ST_VERSION.
    """
    df = df.copy()

    # Ensure needed columns exist
    needed = [
        'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_TERMINATION', 'DAYS_DECISION'
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # Replace placeholder 365243 -> NaN
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace(placeholder_value, np.nan)

    # Choose LAST_DUE column (prefer the final version, fallback to 1st version)
    last_due = df['DAYS_LAST_DUE'].where(~df['DAYS_LAST_DUE'].isna(), df['DAYS_LAST_DUE_1ST_VERSION'])

    # --- Durations (clip to non-negative) ---
    credit_duration = (last_due - df['DAYS_FIRST_DUE'])
    df['CREDIT_DURATION_DAYS'] = credit_duration.clip(lower=0)

    time_to_first_payment = (df['DAYS_FIRST_DUE'] - df['DAYS_FIRST_DRAWING'])
    df['TIME_TO_FIRST_PAYMENT_DAYS'] = time_to_first_payment.clip(lower=0)

    time_to_termination = (df['DAYS_TERMINATION'] - df['DAYS_DECISION'])
    df['TIME_TO_TERMINATION_DAYS'] = time_to_termination.clip(lower=0)

    # --- Overlap flag: previous credit still active at decision time ---
    df['OVERLAP_WITH_CURRENT'] = (
        (df['DAYS_TERMINATION'] > df['DAYS_DECISION'])
        .fillna(False)
        .astype(int)
    )

    # Clean any infinities just in case
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Added Section 7: loan lifecycle duration features.")
    return df

# Apply to your dataframe
merged_df_clean = add_loan_lifecycle_features(merged_df_clean)

from sklearn.model_selection import StratifiedKFold

# Assuming merged_df_clean exists and TARGET is your label
y = merged_df_clean['TARGET'].values
# Exclude TARGET from features; optionally drop IDs if you don’t want them as features
X = merged_df_clean.drop(columns=['TARGET'])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Take the FIRST split as your 80/20 train/test
train_idx, test_idx = next(skf.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Shapes ->",
      "X_train:", X_train.shape, "X_test:", X_test.shape,
      "y_train:", y_train.shape, "y_test:", y_test.shape)

# Quick class-balance sanity check
def pct(pos): 
    return f"{(pos.mean()*100):.2f}%"
print("Class balance:",
      "train:", pct(y_train),
      "| test:", pct(y_test))

train_df = X_train.copy()
train_df["TARGET"] = y_train

test_df = X_test.copy()
test_df["TARGET"] = y_test

print(train_df.shape, test_df.shape)

def save_feature_list(df, filename="features_list.txt", exclude_cols=["TARGET"]):
    """
    Save all feature column names from the given DataFrame into a text file.

    Parameters
    ----------
    df : pandas.DataFrame
        The training dataframe containing feature columns and possibly a target column.
    filename : str, optional
        Name of the output text file. Default is 'features_list.txt'.
    exclude_cols : list of str, optional
        Columns to exclude (e.g., ['TARGET']).

    Returns
    -------
    features : list
        List of feature names that were saved.
    """
    # Filter out unwanted columns
    features = [col for col in df.columns if col not in exclude_cols]

    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(features))

    print(f"✅ Saved {len(features)} features to '{filename}'")
    return features


# ---- Run it ----
features = save_feature_list(train_df, filename="train_features.txt")

import pandas as pd
import numpy as np
import json
from pathlib import Path

def suggest_encoders_to_dict(
    df: pd.DataFrame,
    txt_path: str = "/Users/shehab/Desktop/default_detection_system/encoder_suggestions.txt",
    high_cardinality_threshold: int = 20,
    binary_threshold: int = 2,
):
    """
    Analyze each feature and suggest an encoder, then save as a JSON-formatted
    dict in a .txt file. Uses an absolute path by default to avoid notebook CWD issues.
    """
    encoders_dict = {}

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        unique_vals = int(s.nunique(dropna=True))

        # default
        encoder, note = "none", "Continuous or discrete numeric feature."

        # numeric?
        if np.issubdtype(s.dropna().dtype, np.number):
            if unique_vals <= binary_threshold:
                encoder, note = "binary", "Numeric binary variable (0/1)."
        else:
            # categorical
            if unique_vals <= binary_threshold:
                encoder, note = "binary", "Binary categorical (e.g. Yes/No)."
            elif unique_vals <= high_cardinality_threshold:
                encoder, note = "onehot", f"Low-cardinality categorical ({unique_vals} unique)."
            else:
                encoder, note = "target", f"High-cardinality categorical ({unique_vals} unique)."

            if any(x in col.lower() for x in ["grade", "level", "rank", "score"]):
                encoder, note = "ordinal", "Possible ordinal feature."

        encoders_dict[col] = {
            "encoder": encoder,
            "unique_values": unique_vals,
            "dtype": dtype,
            "note": note,
        }

    # --- write to disk (robust) ---
    out_path = Path(txt_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(encoders_dict, indent=4, ensure_ascii=False))

    # verify
    exists = out_path.exists()
    size = out_path.stat().st_size if exists else 0
    print(f"✅ Saved encoder suggestions to: {out_path}")
    print(f"   Exists: {exists} | Size: {size} bytes")

    # optional: show first few lines
    preview = out_path.read_text(encoding="utf-8").splitlines()[:8]
    print("   Preview:")
    for line in preview:
        print("   ", line)

    return encoders_dict, str(out_path)

# ---- Run it on your dataframe ----
_ = suggest_encoders_to_dict(merged_df_clean)

# --- Imports & globals --------------------------------------------------------
import json
import pickle
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_MISSING_TOKEN = "__MISSING__"

# --- Helpers: load spec, validate columns ------------------------------------
def load_spec_from_txt(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the encoder spec from a JSON-serialized .txt file.
    """
    with open(path, "r") as f:
        return json.load(f)


def validate_spec_columns(
    spec: Dict[str, Any],
    df: pd.DataFrame,
    target_col: str = "TARGET",
    show_examples: int = 10,
) -> None:
    """
    Print a quick report of mismatches between spec and dataframe.
    """
    spec_cols = set(spec.keys()) - {target_col}
    df_cols = set(df.columns)

    missing_in_df = sorted(spec_cols - df_cols)
    extra_in_df   = sorted((df_cols - set(spec.keys())) - {target_col})

    if missing_in_df:
        print(
            f"⚠️ Columns from spec not found in df (will be skipped): "
            f"{missing_in_df[:show_examples]}{' ...' if len(missing_in_df)>show_examples else ''}"
        )
    if extra_in_df:
        print(
            f"ℹ️ Columns present in df but not in spec (ignored unless handled explicitly): "
            f"{extra_in_df[:show_examples]}{' ...' if len(extra_in_df)>show_examples else ''}"
        )


# --- The encoder --------------------------------------------------------------
class SpecEncoder(BaseEstimator, TransformerMixin):
    """
    Encode a DataFrame according to a provided spec:
      - 'none'   -> pass through numeric columns unchanged
      - 'binary' -> ensure columns are 0/1 (object two-level mapped to 0/1; numeric two-level coerced to {0,1})
      - 'onehot' -> one-hot encode using categories seen at fit; missing/unseen -> _MISSING_TOKEN bucket
      - 'target' -> mean target encoding with smoothing; unseen -> global mean

    Notes:
    - Fit on TRAIN ONLY to avoid leakage (esp. for 'target').
    - Transform train and test with the same fitted state to keep columns aligned.
    """

    def __init__(
        self,
        spec: Dict[str, Dict[str, Any]],
        target_col: str = "TARGET",
        smoothing: float = 20.0,
        exclude_cols: Optional[List[str]] = None,   # e.g. ["SK_ID_CURR", "SK_ID_PREV", "community_id"]
    ):
        self.spec = spec
        self.target_col = target_col
        self.smoothing = float(smoothing)
        self.exclude_cols = set(exclude_cols or [])

        # Fitted state
        self.binary_maps_: Dict[str, Dict[Any, int]] = {}
        self.target_stats_: Dict[str, Dict[str, Any]] = {}  # col -> {'mapping': {cat: enc}, 'global_mean': float}
        self.onehot_levels_: Dict[str, pd.Index] = {}       # col -> categories seen at fit
        self.ohe_columns_: List[str] = []                   # final onehot column names (sorted)
        self.passthrough_cols_: List[str] = []              # 'none'/'binary'/'target' final column names
        self.fitted_ = False

    # ---------- binary helpers ----------
    @staticmethod
    def _to_binary_mapping(series: pd.Series) -> Dict[Any, int]:
        """Create a stable 0/1 mapping for a binary series (object or numeric)."""
        uniq = pd.Series(series.dropna().unique())

        if len(uniq) <= 1:
            # Degenerate → map the single value to 0
            return {uniq.iloc[0]: 0} if len(uniq) == 1 else {}

        if pd.api.types.is_numeric_dtype(series):
            # If values are two numbers, map min->0, max->1
            s = sorted(uniq.tolist())
            return {s[0]: 0, s[1]: 1}

        # Object-like heuristics
        lowered = {str(v).strip().lower() for v in uniq}
        if lowered.issubset({"y", "n"}):
            return {v: (0 if str(v).strip().lower()=="n" else 1) for v in uniq}
        if lowered.issubset({"yes", "no"}):
            return {v: (0 if str(v).strip().lower()=="no" else 1) for v in uniq}
        if lowered.issubset({"m", "f"}):
            return {v: (0 if str(v).strip().lower()=="f" else 1) for v in uniq}

        # Fallback: sort as strings, first->0, second->1
        s = sorted(uniq.astype(str).tolist())
        return {v: (0 if str(v) == s[0] else 1) for v in uniq}

    # ---------- target encoding ----------
    def _fit_target_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Mean target encoding with smoothing; missing treated as a category."""
        y = df[self.target_col].astype(float)
        x = df[col].astype("object").fillna(_MISSING_TOKEN)

        global_mean = y.mean()
        counts = x.value_counts()
        means = y.groupby(x).mean()

        enc = (means * counts + self.smoothing * global_mean) / (counts + self.smoothing)
        mapping = enc.to_dict()

        self.target_stats_[col] = {
            "mapping": mapping,
            "global_mean": float(global_mean),
        }

    def _apply_target_encoder(self, s: pd.Series, col: str) -> pd.Series:
        st = self.target_stats_[col]
        x = s.astype("object").fillna(_MISSING_TOKEN)
        return x.map(st["mapping"]).fillna(st["global_mean"]).astype(float)

    # ---------- core API ----------
    def fit(self, df: pd.DataFrame) -> "SpecEncoder":
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in DataFrame.")

        # Reset fitted state
        self.binary_maps_.clear()
        self.target_stats_.clear()
        self.onehot_levels_.clear()
        self.ohe_columns_.clear()
        self.passthrough_cols_.clear()

        for col, meta in self.spec.items():
            if col == self.target_col or col in self.exclude_cols:
                continue
            if col not in df.columns:
                # Skip silently if not found
                continue

            enc = meta.get("encoder", "none")

            if enc == "binary":
                mapping = self._to_binary_mapping(df[col])
                self.binary_maps_[col] = mapping

            elif enc == "target":
                self._fit_target_encoder(df, col)

            elif enc == "onehot":
                # Remember categories as seen during fit (including missing token)
                s = df[col].astype("object").fillna(_MISSING_TOKEN)
                cats = pd.Index(sorted(s.astype(str).unique()))
                self.onehot_levels_[col] = cats

            # 'none' has no fitting state

        # Build and store the full list of one-hot column names (stable/lexicographic)
        if self.onehot_levels_:
            tmp = []
            for col, cats in self.onehot_levels_.items():
                for cat in cats:
                    tmp.append(f"{col}__{cat}")
            self.ohe_columns_ = sorted(tmp)

        # Record passthrough cols (original names for none/binary/target)
        passthrough = []
        for col, meta in self.spec.items():
            if col == self.target_col or col in self.exclude_cols or col not in df.columns:
                continue
            enc = meta.get("encoder", "none")
            if enc in ("none", "binary", "target"):
                passthrough.append(col)
        self.passthrough_cols_ = sorted(passthrough)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("SpecEncoder is not fitted. Call .fit(df) first.")

        out: Dict[str, pd.Series] = {}

        # 1) 'none' → pass through (as numeric)
        for col, meta in self.spec.items():
            if col in self.exclude_cols or col == self.target_col or col not in df.columns:
                continue
            if meta.get("encoder", "none") == "none":
                out[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) 'binary' → map via fitted mapping
        for col, mapping in self.binary_maps_.items():
            if col not in df.columns or col in self.exclude_cols:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                mapped = s.map(mapping)
                if mapped.isna().any():
                    # Fallback numeric rule: nonzero -> 1, zero/NaN -> 0
                    mapped = (s.fillna(0) != 0).astype(int)
            else:
                mapped = s.map(mapping)
                if mapped.isna().any():
                    # Unseen strings → map to 0
                    mapped = mapped.fillna(0).astype(int)
            out[col] = mapped.astype(int)

        # 3) 'target' → apply mean encoding
        for col in self.target_stats_.keys():
            if col in df.columns and col not in self.exclude_cols:
                out[col] = self._apply_target_encoder(df[col], col)

        # 4) 'onehot' → create columns for each seen category; unseen → _MISSING_TOKEN
        for col, cats in self.onehot_levels_.items():
            if col in self.exclude_cols:
                # Create zeros if excluded
                for cat in cats:
                    out[f"{col}__{cat}"] = pd.Series(0, index=df.index, dtype=int)
                continue

            if col not in df.columns:
                # Column entirely missing at transform time → zeros
                for cat in cats:
                    out[f"{col}__{cat}"] = pd.Series(0, index=df.index, dtype=int)
                continue

            s = df[col].astype("object").fillna(_MISSING_TOKEN)
            valid_set = set(cats)
            s = s.where(s.isin(valid_set), _MISSING_TOKEN)

            dummies = pd.get_dummies(s, prefix=col)
            dummies.columns = [c.replace(f"{col}_", f"{col}__") for c in dummies.columns]

            # ensure all fitted columns exist (add missing with 0)
            for cat in cats:
                cname = f"{col}__{cat}"
                if cname not in dummies.columns:
                    dummies[cname] = 0

            dummies = dummies[[f"{col}__{cat}" for cat in cats]].astype(int)

            for cname in dummies.columns:
                out[cname] = dummies[cname]

        # Assemble final frame in a stable order: passthrough + OHE
        col_order: List[str] = []
        col_order.extend(self.passthrough_cols_)
        # Add one-hot columns in the fitted order for stability
        col_order.extend([c for c in self.ohe_columns_ if c in out])

        # It's possible that a passthrough column did not make it into 'out' (e.g., missing at transform).
        # Add any remaining keys to the end to avoid KeyErrors, preserving determinism.
        remaining = [c for c in out.keys() if c not in col_order]
        col_order.extend(sorted(remaining))

        X = pd.DataFrame(out, index=df.index)
        X = X.reindex(columns=col_order, fill_value=0)
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ---------- persistence & metadata ----------
    def save(self, path: str) -> None:
        """Persist the fitted encoder to disk."""
        state = {
            "spec": self.spec,
            "target_col": self.target_col,
            "smoothing": self.smoothing,
            "binary_maps_": self.binary_maps_,
            "target_stats_": self.target_stats_,
            "onehot_levels_": {k: list(v) for k, v in self.onehot_levels_.items()},
            "ohe_columns_": self.ohe_columns_,
            "passthrough_cols_": self.passthrough_cols_,
            "exclude_cols": list(self.exclude_cols),
            "fitted_": self.fitted_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "SpecEncoder":
        """Load a previously saved encoder."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            state["spec"],
            target_col=state["target_col"],
            smoothing=state["smoothing"],
            exclude_cols=state.get("exclude_cols", []),
        )
        obj.binary_maps_ = state["binary_maps_"]
        obj.target_stats_ = state["target_stats_"]
        obj.onehot_levels_ = {k: pd.Index(v) for k, v in state["onehot_levels_"].items()}
        obj.ohe_columns_ = state["ohe_columns_"]
        obj.passthrough_cols_ = state["passthrough_cols_"]
        obj.fitted_ = state["fitted_"]
        return obj

    @property
    def feature_names_(self) -> List[str]:
        """Stable output feature names in transform order."""
        if not self.fitted_:
            raise RuntimeError("SpecEncoder is not fitted.")
        return list(self.passthrough_cols_) + list(self.ohe_columns_)

ENC_SPEC = load_spec_from_txt("encoder_suggestions.txt")

encoder = SpecEncoder(
    ENC_SPEC,
    target_col="TARGET",
    exclude_cols=["SK_ID_CURR", "SK_ID_PREV", "community_id"]  # optional
)

# Fit on training data only
encoder.fit(train_df)

# Transform both training and test sets
X_train_encoded = encoder.transform(train_df)
X_test_encoded  = encoder.transform(test_df)

print("Encoded shapes ->")
print("Train:", X_train_encoded.shape)
print("Test:", X_test_encoded.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print(X_train_scaled.mean(axis=0).round(2))
print(X_train_scaled.std(axis=0).round(2))

# Find columns with NaN or zero variance
mask_valid = ~np.isnan(X_train_scaled).any(axis=0)
mask_nonconstant = X_train_scaled.std(axis=0) > 0

valid_columns = mask_valid & mask_nonconstant

# Filter columns
X_train_scaled = X_train_scaled[:, valid_columns]
X_test_scaled = X_test_scaled[:, valid_columns]

print("Filtered shape:", X_train_scaled.shape)

from sklearn.decomposition import PCA

# ---- 1) Fit PCA on scaled training data ----
max_components = min(100, X_train_scaled.shape[1])  # cap to keep charts readable
pca = PCA(n_components=max_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Explained variance ratios
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n95 = int(np.searchsorted(cum, 0.95) + 1)

print(f"Components to reach 95% variance: {n95}/{len(evr)}")

# ---- 2) Cumulative explained variance (with 95% markers) ----
x_axis = np.arange(1, len(cum) + 1)
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=x_axis, y=cum, mode="lines+markers", name="Cumulative EVR"
))
fig_cum.add_hline(
    y=0.95, line_dash="dash",
    annotation_text="95% threshold", annotation_position="bottom right"
)
fig_cum.add_vline(
    x=n95, line_dash="dash",
    annotation_text=f"{n95} comps", annotation_position="top"
)
fig_cum.update_layout(
    title="PCA: Cumulative Explained Variance",
    xaxis_title="Number of Components",
    yaxis_title="Cumulative Explained Variance Ratio",
    yaxis=dict(range=[0, 1.01])
)
fig_cum.show()

# ---- 3) Individual explained variance (top k comps) ----
k = min(30, len(evr))
fig_bar = px.bar(
    x=np.arange(1, k + 1),
    y=evr[:k],
    labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
    title=f"PCA: Explained Variance per Component (Top {k})"
)
fig_bar.update_traces(hovertemplate="PC%{x}: %{y:.4f}")
fig_bar.show()

# ---- 4) Scatter: PC1 vs PC2 colored by TARGET (sampled for speed) ----
sample_n = min(50000, X_train_pca.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(X_train_pca.shape[0], size=sample_n, replace=False)

scatter_df = pd.DataFrame({
    "PC1": X_train_pca[idx, 0],
    "PC2": X_train_pca[idx, 1],
    "TARGET": y_train[idx].astype(int)
})

fig_scatter = px.scatter(
    scatter_df, x="PC1", y="PC2",
    color=scatter_df["TARGET"].map({0: "No Default", 1: "Default"}),
    opacity=0.5,
    title="PCA Projection: PC1 vs PC2 (sampled)",
    labels={"color": "TARGET"}
)
fig_scatter.update_traces(marker=dict(size=4))
fig_scatter.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import plotly.express as px

# ---- Train RandomForest ----
from sklearn.ensemble import RandomForestClassifier

# ---- Train RandomForest ----
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",     # improves recall for minority class
    random_state=42,
    n_jobs=-1
)

# Fit on training data
rf.fit(X_train_scaled, y_train)

print("✅ Random Forest model trained successfully.")
fig.show()

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import plotly.express as px

# ---- Predict probabilities on TRAIN set ----
y_prob_train = rf.predict_proba(X_train_scaled)[:, 1]

# ---- Apply custom threshold ----
threshold = 0.30
y_pred_train = (y_prob_train >= threshold).astype(int)

# ---- Evaluate ----
auc_train = roc_auc_score(y_train, y_prob_train)
print("🧪 Random Forest Model Evaluation on TRAIN Set")
print(f"Threshold used: {threshold}")
print(f"ROC-AUC Score: {auc_train:.4f}\n")

# Classification report
print("📊 Classification Report (Train):")
print(classification_report(y_train, y_pred_train, digits=3))

# ---- Confusion matrix ----
cm_train = confusion_matrix(y_train, y_pred_train)
cm_train_df = pd.DataFrame(cm_train,
                           index=["Actual: No Default", "Actual: Default"],
                           columns=["Predicted: No Default", "Predicted: Default"])

fig = px.imshow(cm_train_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"TRAIN Confusion Matrix (Threshold={threshold}, ROC-AUC={auc_train:.3f})")
fig.update_layout(width=500, height=500)
fig.show()

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import plotly.express as px

# ---- Predict probabilities on TEST set ----
y_prob_test = rf.predict_proba(X_test_scaled)[:, 1]

# ---- Apply custom threshold ----
threshold = 0.30
y_pred_test = (y_prob_test >= threshold).astype(int)

# ---- Evaluate ----
auc_test = roc_auc_score(y_test, y_prob_test)
print("✅ Random Forest Model Evaluation on TEST Set")
print(f"Threshold used: {threshold}")
print(f"ROC-AUC Score: {auc_test:.4f}\n")

# Classification report
print("📊 Classification Report (Test):")
print(classification_report(y_test, y_pred_test, digits=3))

# ---- Confusion matrix ----
cm_test = confusion_matrix(y_test, y_pred_test)
cm_test_df = pd.DataFrame(cm_test,
                          index=["Actual: No Default", "Actual: Default"],
                          columns=["Predicted: No Default", "Predicted: Default"])

fig = px.imshow(cm_test_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"TEST Confusion Matrix (Threshold={threshold}, ROC-AUC={auc_test:.3f})")
fig.update_layout(width=500, height=500)
fig.show()