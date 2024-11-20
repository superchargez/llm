from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import io

app = FastAPI()

# Helper function to load all sheets into a dictionary of Pandas DataFrames
def load_all_sheets(file: bytes):
    workbook = pd.ExcelFile(io.BytesIO(file))
    # Load each sheet into a DataFrame, store in a dictionary with sheet names as keys
    sheets = {sheet_name: pd.read_excel(workbook, sheet_name=sheet_name, header=None) for sheet_name in workbook.sheet_names}
    return sheets

# Main function to identify tables within a sheet
def find_tables(df: pd.DataFrame, empty_threshold=90, window_size=100):
    tables = []
    row_limit, col_limit = df.shape
    
    def is_range_empty(start_row, end_row, start_col, end_col):
        sub_df = df.iloc[start_row:end_row, start_col:end_col]
        empty_percentage = (sub_df.isna().sum().sum() / sub_df.size) * 100
        return empty_percentage > empty_threshold
    
    row, col = 0, 0
    
    while row < row_limit and col < col_limit:
        end_row, end_col = row + window_size, col + window_size
        if end_row >= row_limit: end_row = row_limit
        if end_col >= col_limit: end_col = col_limit
        
        if not is_range_empty(row, end_row, col, end_col):
            table_start_row, table_start_col = row, col
            
            while end_col < col_limit and not is_range_empty(row, end_row, end_col, end_col + window_size):
                end_col += window_size
            while end_row < row_limit and not is_range_empty(end_row, end_row + window_size, col, end_col):
                end_row += window_size
            
            tables.append((table_start_row, end_row, table_start_col, end_col))
            col = end_col
            
        else:
            col += window_size
        
        if col >= col_limit:
            row += window_size
            col = 0
    
    return tables

@app.post("/extract-tables/")
async def extract_tables(file: UploadFile = File(...)):
    file_bytes = await file.read()
    sheets = load_all_sheets(file_bytes)
    
    results = {}
    for sheet_name, df in sheets.items():
        tables = find_tables(df)
        results[sheet_name] = [{"start_row": r[0], "end_row": r[1], "start_col": r[2], "end_col": r[3]} for r in tables]
        print(f"Tables in sheet '{sheet_name}':", results[sheet_name])  # Print tables found in each sheet
    
    return {"sheets": results}
