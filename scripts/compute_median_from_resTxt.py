import numpy as np
import os
import argparse
import shutil, glob
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "infile", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    infile = args.infile
    assert os.path.exists(infile), f"Cannot find {infile}"

    with open(infile, 'r') as file:
        lines = file.readlines()

    # Concatenate the lines and remove the trailing '\n'
    data = ''.join(lines).strip()

    # Split the data into rows based on the line break pattern
    rows = data.split('\\\n')

    # Convert the rows into a list of dictionaries
    scene_data = []
    columns = None

    for row in rows:
        row_data = row.split()
        if not columns:
            columns = row_data
        else:
            scene_data.append(dict(zip(columns, row_data)))

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(scene_data)

    # Convert only numeric columns to numeric data types
    numeric_cols = ['ATE[cm]', 'R_rmse[deg]', 'MPE[%/m]', 'MTE[m]', 'ATE_int[cm]', 'ATE_rpg[cm]', 'R_rpe[deg/s]', 't_rpe[cm/s]', 't_rpe_perc[%]', 'R_rpe[deg/m]']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN (non-numeric) values in numeric_cols
    df.dropna(subset=numeric_cols, inplace=True)

    # Group the data by 'Scene' and compute the median for each metric
    median_metrics = df.groupby('Scene').median()
    print(f"MEDIAN metrics")
    print(median_metrics)
                

    #mean_metrics = df.groupby('Scene').mean()
    #print(f"Mean metrics")
    #print(mean_metrics)
    

