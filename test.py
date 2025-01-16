from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import base64
import re
import json

app = FastAPI()

class UsageMetadata(BaseModel):
    prompt_token_count: int
    total_token_count: int
    candidates_token_count: Optional[int] = None

class ProcessRequest(BaseModel):
    prompt: str

class ProcessResponse(BaseModel):
    response_data: List[Dict[str, Any]]
    usage_metadata: Optional[UsageMetadata] = None

# Load environment variables
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

# Configure API key for Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=API_KEY)

DEFAULT_PROMPT = """
Extract Events and their associated metrics from the following document in JSON format. 
The format should be a list of dictionaries where each dictionary has an 'Event' key and a 'metrics' key. 
The 'metrics' key should hold a list of metrics as dictionaries, each with 'metric_name', 'metric_value', and optionally 'currency'. 
If no currency is found, set it to null. If there are no events or metrics, return an empty list.
Ensure the response is strictly in JSON format.
"""

def parse_json_response(text: str) -> List[Dict[str, Any]]:
    """
    Parse the response text into valid JSON, handling various formats.
    
    Args:
        text (str): Raw response text from Gemini
        
    Returns:
        List[Dict[str, Any]]: Parsed JSON data
    """
    try:
        # Remove any markdown code blocks
        cleaned_text = re.sub(r'```json\s*|\s*```', '', text.strip())
        
        # Replace single quotes with double quotes
        cleaned_text = cleaned_text.replace("'", '"')
        
        # Try to parse the cleaned text
        parsed_data = json.loads(cleaned_text)
        
        # Ensure the result is a list
        if not isinstance(parsed_data, list):
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            else:
                raise ValueError("Response is neither a list nor a dictionary")
        
        return parsed_data
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse JSON response: {str(e)}. Raw response: {text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing response: {str(e)}"
        )

async def process_pdf_with_gemini(pdf_content: bytes, prompt: str) -> dict:
    """
    Process a PDF file using the Gemini API.
    
    Args:
        pdf_content (bytes): Raw PDF file content
        prompt (str): Prompt for the Gemini model
        
    Returns:
        dict: Response from the Gemini API with parsed JSON
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Encode PDF content to base64
        pdf_data = base64.b64encode(pdf_content).decode('utf-8')
        
        # Generate content
        response = model.generate_content([
            {'mime_type': 'application/pdf', 'data': pdf_data},
            prompt
        ])
        
        # Parse the response text into JSON
        parsed_response = parse_json_response(response.text)
        
        # Convert usage metadata to dictionary format
        usage_metadata = None
        if hasattr(response, 'usage_metadata'):
            usage_metadata = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'total_token_count': response.usage_metadata.total_token_count,
                'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', None)
            }
        
        return {
            "response_data": parsed_response,
            "usage_metadata": usage_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/process_pdf", response_model=ProcessResponse)
async def process_pdf(
    file: UploadFile = File(...),
    custom_prompt: str = Form(None)
):
    """
    Endpoint to process PDF files using Gemini AI.
    
    Args:
        file (UploadFile): The PDF file to process
        custom_prompt (str, optional): Custom prompt to use instead of the default
        
    Returns:
        ProcessResponse: The processed response from Gemini with parsed JSON
    """
    # Validate file type
    if not file.content_type == "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    # Use provided prompt or default
    prompt = custom_prompt or DEFAULT_PROMPT
    
    # Read the file content
    content = await file.read()
    
    # Process the PDF
    result = await process_pdf_with_gemini(content, prompt)
    output_filename = f"output_{file.filename}.json"
    with open(output_filename, 'w') as f:
        json.dump(result['response_data'], f, indent=2)
    return ProcessResponse(**result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
