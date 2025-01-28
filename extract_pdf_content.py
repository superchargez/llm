import os
import tempfile
from typing import Dict, Any, List, Optional
import re
import json
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import subprocess
import mimetypes
from hosting_deal_prompt import prompt_pdf  # Import the prompt from your file
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()

# --- Gemini Setup (moved to global scope for reuse) ---
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set GEMINI_API_KEY in your ~/.env file.")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash") # Initialize Gemini model globally

# --- Pydantic Models (moved to global scope for clarity) ---
class UsageMetadata(BaseModel):
    prompt_token_count: int
    total_token_count: int
    candidates_token_count: Optional[int] = None

class ProcessRequest(BaseModel):
    prompt: str

class ProcessResponse(BaseModel):
    response_data: List[Dict[str, Any]]
    usage_metadata: Optional[UsageMetadata] = None


def convert_office_to_pdf(input_file: str) -> str:
    """Converts DOCX/PPTX to PDF using unoconv."""
    mime_type, _ = mimetypes.guess_type(input_file)

    if mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                       'application/vnd.openxmlformats-officedocument.presentationml.presentation']: # docx or pptx
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf_file: # delete=False to return path
                pdf_output_path = temp_pdf_file.name

                # Use unoconv to convert to PDF
                command = ['unoconv', '--format', 'pdf', '--output', pdf_output_path, input_file]
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                if process.returncode != 0:
                    raise Exception(f"Unoconv failed: {process.stderr}")
                return pdf_output_path

        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="unoconv not found. Please ensure it is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail=f"Error converting to PDF using unoconv: {e.stderr}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error converting docx/pptx to PDF: {e}")
    return input_file # Return original path if not docx/pptx


def parse_json_response(text: str) -> List[Dict[str, Any]]:
    """Parses JSON response from Gemini, handling potential format issues."""
    try:
        cleaned_text = re.sub(r'```json\s*|\s*```', '', text.strip())
        cleaned_text = cleaned_text.replace("'", '"')
        parsed_data = json.loads(cleaned_text)
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


async def process_prompt_with_text(file_content: bytes, file_mime_type: str) -> Dict[str, Any]:
    """Processes the file content (PDF or image) with the sports event hosting deal prompt using Gemini."""
    try:
        prompt_content = prompt_pdf.replace("${content}", "Analyze the attached document.") # Generic prompt

        content_parts = []
        if file_mime_type == 'application/pdf':
            content_parts.append({'mime_type': 'application/pdf', 'data': base64.b64encode(file_content).decode('utf-8')})
        elif file_mime_type.startswith('image/'): # Handle direct image uploads as well
            content_parts.append({'mime_type': file_mime_type, 'data': base64.b64encode(file_content).decode('utf-8')})
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for direct processing. Please upload PDF or Image.")
        content_parts.append(prompt_content)


        response = gemini_model.generate_content(content_parts)

        parsed_response = {} # Initialize as empty dict in case of non-JSON response
        try:
            parsed_response = parse_json_response(response.text)
            if parsed_response and isinstance(parsed_response, list) and len(parsed_response) > 0:
                parsed_response = parsed_response[0] # Take the first element if it's a list of dicts
            elif not parsed_response: # Handle empty list case
                parsed_response = {"isHostingDeal": False, "dateOfAnnouncement": None, "places": None, "events": None, "status": None}

        except HTTPException as e: # Catch JSON parsing errors and return raw text for debugging
            print(f"JSON Parsing Error: {e.detail}")
            print(f"Raw Gemini Response: {response.text}")
            parsed_response = {"error": "JSON Parsing Error", "raw_response": response.text}


        usage_metadata = None
        if hasattr(response, 'usage_metadata'):
            usage_metadata = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'total_token_count': response.usage_metadata.total_token_count,
                'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', None)
            }

        final_response = {
            "response_data": parsed_response,
            "usage_metadata": usage_metadata
        }
        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt with Gemini: {e}")



@app.post("/process-sports-deal/")
async def process_sports_deal(file: UploadFile = File(...)):
    """Endpoint to process uploaded files for sports hosting deals."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file: # Keep original filename suffix
            contents = await file.read()
            tmp_file.write(contents)
            temp_file_path = tmp_file.name

        file_mime_type = file.content_type

        # Convert DOCX/PPTX to PDF if necessary
        pdf_file_path = convert_office_to_pdf(temp_file_path)

        # Read file content (now PDF content if conversion happened, or original content if it was already PDF or image)
        with open(pdf_file_path, 'rb') as f:
            file_content = f.read()

        # Determine the mime type to send to Gemini (always PDF after conversion, or original if image/pdf)
        gemini_mime_type = 'application/pdf' if pdf_file_path != temp_file_path else file_mime_type


        prompt_result = await process_prompt_with_text(file_content, gemini_mime_type) # Send PDF content and mime type

        os.unlink(temp_file_path) # Delete original temp file
        if pdf_file_path != temp_file_path: # Delete converted PDF if conversion happened
            os.unlink(pdf_file_path)


        return JSONResponse(content=prompt_result)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)