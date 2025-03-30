import pandas as pd

file_path = r"D:\5.Data Science\Research\Deepshika Rajendran_COMScDS232P-001_Research\Deepshika Rajendran_COMScDS232P-001_Research Code File\Data\Raw\Inbound & Outbound Dataset.xlsx"

# 1. LOAD THE DATA
# Since there is only one sheet in the file, we load it directly.
df = pd.read_excel(file_path)
print("=== SAMPLE ROWS FROM ORIGINAL DATA ===")
print(df.head(10))
print("\n")

# 2. SHOW COLUMN NAMES AND DATAFRAME INFORMATION
print("=== COLUMN NAMES ===")
print(df.columns.tolist())
print("\n")

print("=== DATAFRAME INFO ===")
print(df.info())
print("\n")

# 3. MISSING VALUES PER COLUMN
print("=== MISSING VALUES PER COLUMN ===")
print(df.isna().sum())
print("\n")

# 4. UNIQUE VALUE COUNTS FOR EACH COLUMN
print("=== UNIQUE VALUES PER COLUMN ===")
for col in df.columns:
    unique_count = df[col].nunique(dropna=True)
    print(f"\nColumn: '{col}'")
    print(f"  Unique Values Count: {unique_count}")
print("\n")

# 5. CHECK HOW MANY 'Inbound Message' ARE BLANK
# Here we treat blank as an empty string or NaN.
blank_inbound = (df["Inbound Message"].isna()) | (df["Inbound Message"].astype(str).str.strip() == "")
num_blank_inbound = blank_inbound.sum()
print("=== BLANK 'Inbound Message' COUNT ===")
print(f"Blank Inbound Messages: {num_blank_inbound}")
print("\n")

# 6. SHOW ROWS WHERE 'Inbound Message' IS BLANK BUT THERE IS AN OUTBOUND MESSAGE
# The outbound message is in the "Replied Post" column.
# Filter rows where 'Inbound Message' is blank but 'Replied Post' is not blank
blank_inbound_nonblank_outbound = df[
    blank_inbound & 
    (df["Replied Post"].notna()) & 
    (df["Replied Post"].astype(str).str.strip() != "")
]

# Display results
print("=== ROWS WITH BLANK 'Inbound Message' BUT NON-BLANK 'Replied Post' (sample up to 5) ===")
print(blank_inbound_nonblank_outbound[["Inbound Message", "Replied Post", "Associated Cases"]].head(5))
print("\n")

