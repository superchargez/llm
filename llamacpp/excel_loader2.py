from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import io

app = FastAPI()

# Helper function to load Excel sheet into a Pandas DataFrame
def load_sheet_to_pandas(file: bytes, sheet_name: str = None):
    # Load the workbook and check for available sheets
    workbook = pd.ExcelFile(io.BytesIO(file))
    
    # Use the specified sheet if it exists, otherwise default to the first sheet
    if sheet_name not in workbook.sheet_names:
        sheet_name = workbook.sheet_names[0]
        print(f"Specified sheet '{sheet_name}' does not exist. Using first available sheet: '{sheet_name}'")

    # Load the specified sheet into a DataFrame
    df = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    
    return df

# Main function to identify tables within a sheet
def find_tables(df: pd.DataFrame, empty_threshold=90, window_size=100):
    tables = []
    row_limit, col_limit = df.shape
    
    def is_range_empty(start_row, end_row, start_col, end_col):
        # Extract the sub-dataframe and calculate the percentage of NaNs
        sub_df = df.iloc[start_row:end_row, start_col:end_col]
        empty_percentage = (sub_df.isna().sum().sum() / sub_df.size) * 100
        return empty_percentage > empty_threshold
    
    row, col = 0, 0
    
    while row < row_limit and col < col_limit:
        end_row, end_col = row + window_size, col + window_size
        if end_row >= row_limit: end_row = row_limit
        if end_col >= col_limit: end_col = col_limit
        
        if not is_range_empty(row, end_row, col, end_col):
            # Identify table boundaries
            table_start_row, table_start_col = row, col
            
            # Expand the end row and column until the end of the table is found
            while end_col < col_limit and not is_range_empty(row, end_row, end_col, end_col + window_size):
                end_col += window_size
            while end_row < row_limit and not is_range_empty(end_row, end_row + window_size, col, end_col):
                end_row += window_size
            
            # Append the found table's range
            tables.append((table_start_row, end_row, table_start_col, end_col))
            col = end_col  # Move to the next column block
            
        else:
            col += window_size  # Shift the window to the right
        
        if col >= col_limit:
            row += window_size
            col = 0
    
    return tables

@app.post("/extract-tables/")
async def extract_tables(file: UploadFile = File(...), sheet_name: str = None):
    file_bytes = await file.read()
    df = load_sheet_to_pandas(file_bytes, sheet_name)
    
    tables = find_tables(df)
    print(tables)
    return {"tables": [{"start_row": r[0], "end_row": r[1], "start_col": r[2], "end_col": r[3]} for r in tables]}
