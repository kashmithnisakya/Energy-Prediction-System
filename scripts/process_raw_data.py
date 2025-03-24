import pandas as pd
from pathlib import Path

def process_all_csvs():
    raw_folder = Path("../ml/data/raw")
    processed_folder = Path("../ml/data/processed")
    processed_folder.mkdir(parents=True, exist_ok=True)

    for csv_file in raw_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)

            if {'time_index', 'Active Energy', 'kVA'}.issubset(df.columns):
                df_filtered = df[['time_index', 'Active Energy', 'kVA']].copy()
                df_filtered.columns = ['time', 'energy', 'power']

                # Create new filename: remove spaces, replace - with _, convert to lowercase
                new_filename = csv_file.name.replace(" ", "_").replace("-", "_").lower()
                output_file = processed_folder / new_filename
                
                df_filtered.to_csv(output_file, index=False)
                print(f"✅ Processed: {new_filename}")
            else:
                print(f"⚠️ Skipped (missing columns): {csv_file.name}")

        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")

if __name__ == "__main__":
    process_all_csvs()