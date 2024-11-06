import os; os.chdir(r"../excel_files")
import pandas as pd
from typing import List, Dict

class TableDetector:
    def __init__(self, min_table_rows=3, min_table_cols=2, max_empty_ratio=0.7):
        """
        Initialize TableDetector with configuration parameters.
        
        Args:
            min_table_rows: Minimum number of rows to consider a region as table
            min_table_cols: Minimum number of columns to consider a region as table
            max_empty_ratio: Maximum ratio of empty cells allowed in a table
        """
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self.max_empty_ratio = max_empty_ratio

    def read_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read Excel or CSV file and return dict of dataframes for each sheet.
        """
        if file_path.endswith('.csv'):
            return {'Sheet1': pd.read_csv(file_path)}
        else:
            return pd.read_excel(file_path, sheet_name=None)

    def is_qa_pair(self, row: pd.Series) -> bool:
        """
        Check if a row contains a question-answer pair.
        """
        # Count non-null cells
        non_null_cells = row.notna().sum()
        
        # Check if row has 2 non-empty cells and contains question-like patterns
        if non_null_cells == 2:
            text = ' '.join(str(x) for x in row if pd.notna(x)).lower()
            question_indicators = ['?', 'what', 'when', 'where', 'why', 'how', 'explain']
            return any(indicator in text for indicator in question_indicators)
        return False

    def get_region_type(self, df_region: pd.DataFrame) -> str:
        """
        Determine if a region is a table, Q&A, or text chunk.
        """
        if df_region.empty:
            return 'empty'

        # Calculate empty ratio
        empty_ratio = df_region.isna().sum().sum() / (df_region.shape[0] * df_region.shape[1])
        
        # Check if region looks like Q&A pairs
        qa_pair_count = sum(self.is_qa_pair(row) for _, row in df_region.iterrows())
        
        if qa_pair_count / df_region.shape[0] > 0.5:
            return 'qa_pairs'
        
        # Check if region looks like a table
        if (df_region.shape[0] >= self.min_table_rows and 
            df_region.shape[1] >= self.min_table_cols and 
            empty_ratio < self.max_empty_ratio):
            return 'table'
            
        return 'text_chunk'

    def find_regions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find and classify different regions in the dataframe.
        
        Returns:
            List of dictionaries containing region information:
            {
                'type': str ('table', 'qa_pairs', 'text_chunk'),
                'start_row': int,
                'end_row': int,
                'start_col': int,
                'end_col': int,
                'data': pd.DataFrame
            }
        """
        regions = []
        current_region = None
        
        # Identify empty rows and columns
        empty_rows = df.isna().all(axis=1)
        
        # Process the dataframe row by row
        current_start = 0
        
        for i in range(len(df) + 1):
            # Check if we're at the end or found an empty row
            if i == len(df) or empty_rows.iloc[i]:
                if current_start < i:
                    # Extract and classify the region
                    region_df = df.iloc[current_start:i]
                    region_type = self.get_region_type(region_df)
                    
                    regions.append({
                        'type': region_type,
                        'start_row': current_start,
                        'end_row': i,
                        'start_col': 0,
                        'end_col': df.shape[1],
                        'data': region_df
                    })
                current_start = i + 1
        
        return regions

    def process_file(self, file_path: str) -> Dict[str, List[Dict]]:
        """
        Process an Excel/CSV file and return detected regions for each sheet.
        """
        sheets = self.read_file(file_path)
        results = {}
        
        for sheet_name, df in sheets.items():
            results[sheet_name] = self.find_regions(df)
            
        return results

def print_regions(regions_by_sheet: Dict[str, List[Dict]]):
    """
    Print detected regions in a readable format.
    """
    for sheet_name, regions in regions_by_sheet.items():
        print(f"\nSheet: {sheet_name}")
        for i, region in enumerate(regions, 1):
            print(f"\nRegion {i}:")
            print(f"Type: {region['type']}")
            print(f"Location: Rows {region['start_row']}-{region['end_row']}, " 
                  f"Cols {region['start_col']}-{region['end_col']}")
            print("\nSample data:")
            print(region['data'].head(3))
            print("..." if len(region['data']) > 3 else "")