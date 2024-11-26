import pandas as pd


# Read both CSV files
def update_csv(base_path, output_path):
    base_df = pd.read_csv(base_path)
    output_df = pd.read_csv(output_path)

    # Create a merge key combining image_name and class
    base_df["merge_key"] = base_df["image_name"] + "," + base_df["class"]
    output_df["merge_key"] = output_df["image_name"] + "," + output_df["class"]

    # Update the rle values where merge_key matches
    base_df.loc[
        base_df["merge_key"].isin(output_df["merge_key"]), "rle"
    ] = output_df.loc[output_df["merge_key"].isin(base_df["merge_key"]), "rle"].values

    # Remove the temporary merge_key column
    base_df = base_df.drop("merge_key", axis=1)

    # Save the updated dataframe back to base.csv
    base_df.to_csv("base.csv", index=False)


path = "/data/ephemeral/home/mpark/level2-cv-semanticsegmentation-cv-8-lv3/base/"
base_path = path + "base.csv"
output_paths = []
for i in range(19, 29):
    output_paths.append(path + f"outputs/{i}.csv")

for output_path in output_paths:
    update_csv(base_path, output_path)
