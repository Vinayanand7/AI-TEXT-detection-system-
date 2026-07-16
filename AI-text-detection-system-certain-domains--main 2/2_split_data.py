import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_dataset1.csv")

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["generated"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["generated"],
    random_state=42
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nTrain distribution:")
print(train_df["generated"].value_counts())

print("\nVal distribution:")
print(val_df["generated"].value_counts())

print("\nTest distribution:")
print(test_df["generated"].value_counts())

print("Saved train/val/test CSVs")