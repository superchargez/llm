from typing import Optional, List, Dict, Any, Union
import json
import re
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from jinja2 import Template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime
import random
import glob
from transformers.utils import get_json_schema

app = FastAPI()

def prepare_messages(
    query: str,
    tools: Optional[dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Prepare the system and user messages for the given query and tools.
    
    Args:
        query: The query to be answered.
        tools: The tools available to the user. Defaults to None, in which case if a
            list without content will be passed to the model.
        history: Exchange of messages, including the system_prompt from
            the first query. Defaults to None, the first message in a conversation.
    """
    if tools is None:
        tools = []
    if history:
        messages = history.copy()
        messages.append({"role": "user", "content": query})
    else:
        messages = [
            {"role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
            {"role": "user", "content": query}
        ]
    return messages

def parse_response(text: str) -> str | Dict[str, Any]:
    """Parses a response from the model, returning either the
    parsed list with the tool calls parsed, or the
    model thought or response if couldn't generate one.

    Args:
        text: Response from the model.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    return text

model_name_smollm = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Use BitsAndBytesConfig for quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_smollm,
    quantization_config=nf4_config,
    device_map="auto",
    torch_dtype=torch.float16,  # Use float16 for mixed precision training
    low_cpu_mem_usage=True,  # Enable CPU memory optimization
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name_smollm)

def get_current_time() -> str:
    """Returns the current time in 24-hour format.

    Returns:
        str: Current time in HH:MM:SS format.
    """
    return datetime.now().strftime("%H:%M:%S")

def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)

# Enhanced system prompt for better function generation and Excel handling
system_prompt = Template("""You are an expert AI assistant specialized in data analysis and function composition. You have access to Excel files and can create or use functions to answer questions about them.

When handling Excel-related queries:
1. First determine which Excel file(s) to use
2. Then decide what operations are needed
3. Finally, make the appropriate function calls

You have access to the following tools:
<tools>{{ tools }}</tools>

For any new operations needed, you can suggest function signatures in this format:
<function_suggestion>
{
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
        "param1": {"type": "type1", "description": "param1 description"},
        ...
    },
    "returns": {"type": "return_type", "description": "return description"}
}
</function_suggestion>

Your response MUST follow this format:
<tool_call>[
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
]</tool_call>""")

class ExcelManager:
    def __init__(self, excel_directory: str = "./excel_files"):
        self.excel_directory = Path(excel_directory)
        self.file_cache = {}
        
    def list_excel_files(self) -> List[str]:
        """List all Excel files in the directory."""
        return [str(p) for p in self.excel_directory.glob("*.xlsx")]
    
    def read_excel_file(self, filename: str) -> pd.DataFrame:
        """Read an Excel file and cache it."""
        if filename not in self.file_cache:
            path = self.excel_directory / filename
            if not path.exists():
                raise FileNotFoundError(f"Excel file {filename} not found")
            self.file_cache[filename] = pd.read_excel(path)
        return self.file_cache[filename]
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Get information about an Excel file."""
        df = self.read_excel_file(filename)
        return {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict()
        }
    
    def query_excel(self, filename: str, query: str) -> pd.DataFrame:
        """Query an Excel file using pandas query syntax."""
        df = self.read_excel_file(filename)
        return df.query(query)

class ExcelTools:
    def __init__(self, excel_manager: ExcelManager):
        self.excel_manager = excel_manager
        
    def list_available_files(self) -> List[str]:
        """Lists all available Excel files."""
        return self.excel_manager.list_excel_files()
    
    def get_column_info(self, filename: str) -> Dict[str, Any]:
        """Gets column information for a specific Excel file."""
        return self.excel_manager.get_file_info(filename)
    
    def query_data(self, filename: str, query: str) -> Dict[str, Any]:
        """Queries data from an Excel file."""
        try:
            result = self.excel_manager.query_excel(filename, query)
            return {"result": result.to_dict(orient="records")}
        except Exception as e:
            raise ValueError(f"Query error: {str(e)}")

# Initialize managers and tools
excel_manager = ExcelManager()
excel_tools = ExcelTools(excel_manager)

# Update tools list with Excel-related functions
tools = [
    {
        "name": "list_available_files",
        "description": "Lists all available Excel files",
        "parameters": {},
        "returns": {"type": "list", "description": "List of Excel filenames"}
    },
    {
        "name": "get_column_info",
        "description": "Gets column information for an Excel file",
        "parameters": {
            "filename": {"type": "string", "description": "Name of the Excel file"}
        },
        "returns": {"type": "object", "description": "Column information"}
    },
    {
        "name": "query_data",
        "description": "Queries data from an Excel file",
        "parameters": {
            "filename": {"type": "string", "description": "Name of the Excel file"},
            "query": {"type": "string", "description": "Query string in pandas syntax"}
        },
        "returns": {"type": "object", "description": "Query results"}
    },
    # Your existing tools
    get_json_schema(get_random_number_between),
    get_json_schema(get_current_time)
]

toolbox = {
    "list_available_files": excel_tools.list_available_files,
    "get_column_info": excel_tools.get_column_info,
    "query_data": excel_tools.query_data,
    "get_random_number_between": get_random_number_between,
    "get_current_time": get_current_time
}

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None

@app.post("/query")
async def query_model(request: QueryRequest):
    messages = prepare_messages(request.query, tools=tools, history=request.history)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    tool_calls = parse_response(result)
    
    if isinstance(tool_calls, str):
        # Check if there's a function suggestion
        suggestion_pattern = r"<function_suggestion>(.*?)</function_suggestion>"
        suggestion_match = re.search(suggestion_pattern, tool_calls, re.DOTALL)
        if suggestion_match:
            return {
                "response": "New function suggested",
                "function_suggestion": json.loads(suggestion_match.group(1))
            }
        return {"response": tool_calls}
    
    try:
        tool_responses = []
        for tc in tool_calls:
            func = toolbox.get(tc["name"])
            if func:
                response = func(**tc["arguments"])
                tool_responses.append(response)
        return {"tool_calls": tool_calls, "tool_responses": tool_responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))