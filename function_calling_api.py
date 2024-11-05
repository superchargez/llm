import json
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from jinja2 import Template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import get_json_schema
from datetime import datetime
import random

app = FastAPI()

system_prompt = Template("""You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>{{ tools }}</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>""")

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

tools = [get_json_schema(get_random_number_between), get_json_schema(get_current_time)]

toolbox = {"get_random_number_between": get_random_number_between, "get_current_time": get_current_time}

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None

@app.post("/query")
async def query_model(request: QueryRequest):
    messages = prepare_messages(request.query, tools=tools, history=request.history)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    tool_calls = parse_response(result)
    
    if isinstance(tool_calls, str):
        return {"response": tool_calls}
    
    try:
        tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
        return {"tool_calls": tool_calls, "tool_responses": tool_responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server, use the command: uvicorn your_script_name:app --reload
# curl -X POST "http://localhost:8384/query" -H "Content-Type: application/json" -d '{
#   "query": "What is the current time?"
# }'

