import pandas as pd

def preprocess():
    # Create CSV files organized by each Shakespeare play
    df = pd.read_csv("data/literature/shakespeare_plays.csv")
    grouped = df.groupby("play_name")
    plays = {play: data for play, data in grouped}
    for play, data in plays.items():
        data.to_csv(f"data/plays/{play}.csv", index=False)