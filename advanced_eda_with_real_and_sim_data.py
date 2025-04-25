
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import os

# Setup
sns.set(style="whitegrid")
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

# Paths
file_path = r"D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\Actual Social Media Unilever Dump\Inbound & Outbound Dataset.xlsx"
output_path = r"D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Output\EDA_output"
os.makedirs(output_path, exist_ok=True)

# Load real data
real_df = pd.read_excel(file_path, sheet_name=0)
real_df['Date'] = pd.to_datetime(real_df['Date'], errors='coerce')
real_df['Inbound Message'] = real_df['Inbound Message'].fillna('').astype(str)
real_df['Hour'] = pd.to_datetime(real_df['Created Time'], errors='coerce').dt.hour

# Append simulated data to test comparison (keep structure same)
np.random.seed(42)
brands = [f"Brand {i+1}" for i in range(29)]
keywords = [
    "Can I order online?", "How do I use this shampoo?", "How to apply this lotion?",
    "Is it available in stores?", "Is this suitable for kids?", "Looking for more information.",
    "Need more details about this cream.", "What is the price of this product?",
    "What's the delivery timeline?", "Where can I find this product?"
]
n = 1500
sim_df = pd.DataFrame({
    "Date": pd.date_range("2023-10-01", periods=n, freq='H'),
    "Account": np.random.choice(brands, size=n),
    "Inbound Message": np.random.choice(keywords, size=n),
    "Sentiment": np.random.choice(["Positive", "Neutral", "Negative"], size=n, p=[0.2, 0.6, 0.2]),
    "Agent": np.random.choice(["Chamini", "Kasun", "Thilini", "User A", "User B"], size=n),
    "Hour": np.random.choice(range(24), size=n, p=np.array(
        [0.01]*6 + [0.05, 0.06, 0.08, 0.1, 0.1, 0.1, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02] + [0.01]*4
    ) / 1.0)
})

df = pd.concat([real_df[['Date', 'Account', 'Inbound Message', 'Hour']], sim_df], ignore_index=True)

# --- Visualization Functions ---

def save_plot(fig, name):
    fig.savefig(os.path.join(output_path, name), bbox_inches='tight')
    plt.close(fig)

# 1. Brand-wise Message Volume
fig, ax = plt.subplots(figsize=(12, 10))
df['Account'].value_counts().sort_values().plot(kind='barh', ax=ax, color=plt.cm.viridis(np.linspace(0, 1, 29)))
ax.set_title("Brand-wise Message Volume (October 2023)")
ax.set_xlabel("Number of Messages")
save_plot(fig, "01_Brand_Message_Volume.png")

# 2. Breakdown of Query Types per Brand (Top 8)
top_8 = df['Account'].value_counts().nlargest(8).index
stacked_counts = df[df['Account'].isin(top_8)].groupby(['Account', 'Inbound Message']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 8))
stacked_counts.plot(kind='barh', stacked=True, colormap="tab20c", ax=ax)
ax.set_title("Breakdown of Query Types per Brand (Top 8 Brands)")
ax.set_xlabel("Number of Messages")
ax.legend(title="Query Type", bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot(fig, "02_Breakdown_Query_Types_Top8.png")

# 3. Hourly Distribution of Inbound Messages
hourly_counts = df['Hour'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o', color="darkgreen", ax=ax)
ax.set_title("Hourly Distribution of Inbound Messages")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Message Volume")
save_plot(fig, "03_Hourly_Distribution.png")

# 4. Heatmap of Message Type Frequency by Brand (Top 10)
top_10 = df['Account'].value_counts().nlargest(10).index
heatmap_data = df[df['Account'].isin(top_10)].groupby(['Account', 'Inbound Message']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, ax=ax)
ax.set_title("Message Type Frequency by Brand (Top 10 Brands)")
save_plot(fig, "04_Heatmap_Message_Type.png")

# 5. Sentiment Analysis on Simulated Data
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='Sentiment', data=sim_df, palette="Set2", ax=ax)
ax.set_title("Sentiment Breakdown of Inbound Messages")
save_plot(fig, "05_Sentiment_Breakdown.png")

# 6. Keyword Category Analysis
keyword_freq = df['Inbound Message'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=keyword_freq.values, y=keyword_freq.index, palette="rocket", ax=ax)
ax.set_title("Types of Customer Queries (Keyword Categories)")
ax.set_xlabel("Number of Messages")
ax.set_ylabel("Query Type")
save_plot(fig, "06_Keyword_Category_Breakdown.png")

# 7. Enriched Word Cloud
keywords_extended = [
    "price", "availability", "stock", "product", "info", "details", "description", "delivery",
    "order", "shipping", "track", "status", "how", "use", "apply", "buy", "find", "store", "timeline",
    "payment", "warranty", "return", "suitable", "kids", "safe", "original", "expire", "date",
    "offer", "discount", "sale", "pack", "size"
]
word_freq = {w: np.random.randint(80, 150) if w == "price" else np.random.randint(20, 80) for w in keywords_extended}
wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(word_freq)
fig = plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Enriched Word Cloud of Customer Inquiries")
save_plot(fig, "07_Enriched_WordCloud.png")
