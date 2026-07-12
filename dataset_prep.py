import pandas as pd

df1 = pd.read_csv("dataset\goemotions_1.csv")
df2 = pd.read_csv("dataset\goemotions_2.csv")
df3 = pd.read_csv("dataset\goemotions_3.csv")



# Merge into one dataframe
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Dataset size:", len(df))
print("Columns:", df.columns)

# Step 1: Identify emotion columns (all after 'example_very_unclear')
emotion_columns = df.columns[9:]   # adjust if needed

# Step 2: For each row, find which emotion(s) are marked as 1
def get_emotion(row):
    labels = [col for col in emotion_columns if row[col] == 1]
    if labels:
        return labels[0]   # take the first emotion for simplicity
    else:
        return None

df["labels"] = df.apply(get_emotion, axis=1)

# Step 3: Define mapping from GoEmotions labels → your categories
mapping = {
    "joy": "happy", "excitement": "happy", "amusement": "happy", "love": "happy",
    "sadness": "sad", "disappointment": "sad", "grief": "sad", "remorse": "sad",
    "anger": "angry", "annoyance": "angry", "frustration": "angry",
    "surprise": "surprise", "realization": "surprise", "curiosity": "surprise",
    "fear": "fear", "nervousness": "fear", "embarrassment": "fear",
    "disgust": "neutral", "boredom": "neutral", "confusion": "neutral",
    "neutral": "neutral"
}

# Step 3: Apply mapping
df["emotion"] = df["labels"].map(mapping)

# Drop rows where mapping failed (labels not in your categories)
df = df.dropna(subset=["emotion"])

# Step 4: Keep only text + emotion columns
df_clean = df[["text", "emotion"]]

# Step 5: Save prepared dataset
df_clean.to_csv("goemotions_prepared.csv", index=False)

print("Prepared dataset saved as goemotions_prepared.csv")
print(df_clean.head())
