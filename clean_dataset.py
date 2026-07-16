import pandas as pd
import numpy as np
import re

df = pd.read_csv("Training_Essay_Data.csv")

print("Original shape:", df.shape)

print(df.columns)
print(df.isnull().sum())

df = df.dropna(subset=["text"])

def clean_text(text):
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove weird characters except punctuation
    text = re.sub(r"[^\w\s.,!?;:()\"'-]", "", text)
    
    return text.strip()

df["text"] = df["text"].apply(clean_text)

df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df = df[df["word_count"] > 150]

df = df[df["word_count"] < 1000]

df = df[df["generated"].isin([0, 1])]

df["length_bin"] = pd.cut(
    df["word_count"],
    bins=[150, 300, 450, 600, 800, 1000],
    labels=["150-300", "300-450", "450-600", "600-800", "800-1000"]
)

balanced_df = []

for bin_name in df["length_bin"].unique():
    bin_data = df[df["length_bin"] == bin_name]
    
    human = bin_data[bin_data["generated"] == 0]
    ai = bin_data[bin_data["generated"] == 1]
    
    min_size = min(len(human), len(ai))
    
    if min_size > 0:
        human_sample = human.sample(min_size, random_state=42)
        ai_sample = ai.sample(min_size, random_state=42)
        balanced_df.append(pd.concat([human_sample, ai_sample]))

df = pd.concat(balanced_df).reset_index(drop=True)

df = df.reset_index(drop=True)

print("Final shape:", df.shape)
print(df["generated"].value_counts())
print("Average word count:", df["word_count"].mean())
df.to_csv("cleaned_dataset1.csv", index=False)

print(df.groupby("generated")["word_count"].mean())
print(df.groupby("generated")["word_count"].describe())