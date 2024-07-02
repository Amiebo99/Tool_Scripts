import pandas as pd
import os
import glob
import argparse

# Function to extract a section from the filename
def extract_section_from_filename(filename):
    # This is severely hard coded but if the foramt is identical every time then we are all good
    section = filename[32:40]
    return section

def main(directory, output_filename):
    # List to store individual DataFrames
    df_list = []

    # Process each CSV file in the directory
    for filepath in glob.glob(os.path.join(directory, '*.DPT')):
        filename = os.path.basename(filepath)
        # print(filename[-18:]) only for debugging
        if filename[-18:] == "_Temperature.0.DPT":
            section = extract_section_from_filename(filename)
        
            # Read the CSV file into a DataFrame (assuming no headers)
            df = pd.read_csv(filepath, header=None)
            
            # Add the extracted section as the third column
            df[2] = section
            
            # Append the DataFrame to the list
            df_list.append(df)

    # Concatenate all DataFrames into one
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the concatenated DataFrame to a new file
    final_df.to_csv(os.path.join(directory, output_filename), index=False, header=False)

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and concatenate CSV files.')
    parser.add_argument('directory', type=str, help='Directory containing the CSV files')
    parser.add_argument('output_filename', type=str, help='Output filename for the concatenated CSV')
    
    args = parser.parse_args()
    
    main(args.directory, args.output_filename)
