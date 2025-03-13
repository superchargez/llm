import base64
import tempfile
import mimetypes
import re
import json
from typing import Annotated, Dict, Any, List
from fastapi import FastAPI, Header, HTTPException, File, UploadFile
import google.generativeai as genai
from dotenv import load_dotenv
import os
import subprocess
import httpx
from datetime import datetime
from fastapi.responses import JSONResponse
from urllib.parse import urlparse

app = FastAPI()

# --- Gemini Setup ---
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

# API_KEY = os.getenv("mtp_GEMINI_KEY")  # Or your preferred key
API_KEY = os.getenv("super_chage_GEMINI_KEY")  # Or your preferred key
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables.  Set it in your ~/.env file."
    )

genai.configure(api_key=API_KEY)
# gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # Or your preferred model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
)

def parse_json_response(text: str) -> List[Dict[str, Any]]:
    """Parses JSON response from Gemini, handling potential format issues."""
    try:
        cleaned_text = re.sub(r"`json\s*|\s*`", "", text.strip())
        parsed_data = json.loads(cleaned_text)
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        elif not isinstance(parsed_data, list):
            raise ValueError("Response is neither a list nor a dictionary")
        return parsed_data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"JSON parsing error: {e}. Raw: {text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

def convert_office_to_pdf(input_file: str) -> str:
    """Converts DOCX/PPTX to PDF using soffice."""
    mime_type, _ = mimetypes.guess_type(input_file)
    if mime_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]:
        try:
            temp_dir = tempfile.mkdtemp(dir="/tmp")
            base_filename = os.path.basename(input_file)
            pdf_filename = os.path.splitext(base_filename)[0] + ".pdf"
            pdf_output_path_full = os.path.join(temp_dir, pdf_filename)
            command = [
                "soffice", "--headless", "--convert-to", "pdf",
                "--outdir", temp_dir, input_file
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                raise Exception(f"soffice failed: {process.stderr}")

            if os.path.exists(pdf_output_path_full):
                return pdf_output_path_full
            else:
                raise Exception("PDF not found after conversion.")

        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="soffice not found.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Conversion error: {e}")
    return input_file

# from deals_prompts import unified_modified_prompt  # Ensure this import exists
from deals_prompts import unified_modificaition2  # Ensure this import exists

async def process_sports_deals(file_content: bytes, file_mime_type: str) -> Dict[str, Any]:
    """Processes file content, returns multiple deals."""
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    content_parts = [{"parts": [{"mime_type": file_mime_type, "data": encoded_content}, {"text": unified_modificaition2}]}] # latest file modified
    try:
        # response = model.generate_content(contents=content_parts, generation_config=genai.types.GenerationConfig(max_output_tokens=8192))
        response = model.generate_content(contents=content_parts, generation_config=generation_config)
        parsed_responses = parse_json_response(response.text)
        all_no_deal = all(not resp.get("is_deal", False) for resp in parsed_responses)
        if all_no_deal:
            parsed_responses = [{"is_deal": False}]
        usage_metadata = None
        if hasattr(response, "usage_metadata"):
            usage_metadata = {k: getattr(response.usage_metadata, k) for k in ["prompt_token_count", "total_token_count", "candidates_token_count"] if hasattr(response.usage_metadata, k)}
        return {"response_data": parsed_responses, "usage_metadata": usage_metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

def clean_date(date_str: str) -> str | None:
    """Cleans and validates date strings, handling partial dates."""
    if not date_str:
        return None
    if len(date_str.split("-")) == 2:
        date_str += "-01"
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        return None

def clean_deal_data(data: dict, deal_type: str) -> dict:
    """Cleans deal data, handling None/empty values and specific fields."""
    cleaned = {}
    array_fields = ["sellerOrganizations", "buyerOrganizations", "events", "places", "teams", "athletes", "venues"]
    for field in array_fields:
        if field in data:
            cleaned[field] = [item for item in data[field] if item]
    string_fields = ["type", "valueType", "dealName"]
    for field in string_fields:
        if field in data and data[field]:
            cleaned[field] = data[field]
    boolean_fields = ["isBroadcastDeal", "isSponsorshipDeal", "isHostingDeal"]
    for field in boolean_fields:
        if field in data and data[field] is not None:  # Allow False, but not None
            cleaned[field] = data[field]
    date_fields = ["startDate", "endDate", "dateOfAnnouncement"]
    for field in date_fields:
        if field in data:
            cleaned[field] = clean_date(data.get(field))
    if "currency" in data and data.get("currency"):
        cleaned["currency"] = data["currency"]
    numeric_fields = ["valueAnnualised", "valueTotal"]
    for field in numeric_fields:
        if field in data and data.get(field) is not None:
            cleaned[field] = data[field]
    if deal_type == "hosting" and "status" in data and data.get("status"):
        cleaned["status"] = data["status"]
    return {k: v for k, v in cleaned.items() if v is not None}  # Remove None values

def clean_submission_data(submission_data: dict) -> dict | None:
    """Cleans the full submission, normalizes types, and handles empty deals."""
    if not submission_data or "type" not in submission_data:
        return None
    normalized_type = submission_data["type"].replace("broadcasting", "broadcast")
    submission_data["type"] = normalized_type
    cleaned_deal = clean_deal_data(submission_data.get(normalized_type, {}), normalized_type)
    cleaned_submission = {"type": normalized_type}
    if cleaned_deal:
        cleaned_submission[normalized_type] = cleaned_deal
    return cleaned_submission if len(cleaned_submission) > 1 else None  # Only type: return None


def create_submission_data(deal_type: str, individual_deal_data: Dict) -> Dict | None:
    """Creates the submission data dictionary based on deal type."""
    if deal_type == "broadcasting" and individual_deal_data.get("broadcastingDealDetails") and individual_deal_data.get("is_deal") and individual_deal_data.get("broadcastingDealDetails",{}).get("isBroadcastDeal"):
        return {
            "type": "broadcast",
            "broadcast": individual_deal_data.get("broadcastingDealDetails", {})
        }
    elif deal_type == "sponsorship" and individual_deal_data.get("sponsorshipDealDetails") and individual_deal_data.get("is_deal") and individual_deal_data.get("sponsorshipDealDetails", {}).get("isSponsorshipDeal"):
        sponsorship_details = individual_deal_data.get("sponsorshipDealDetails", {})
        if sponsorship_details.get("type") == "SPONSOR_PARTNER":  # your correction
            sponsorship_details["type"] = "SPONOSOR_PARTNER"
        return {
            "type": "sponsorship",
            "sponsorship": sponsorship_details
        }
    elif deal_type == "hosting" and individual_deal_data.get("hostingDealDetails") and individual_deal_data.get("is_deal") and individual_deal_data.get("hostingDealDetails", {}).get("isHostingDeal"):
        return {
            "type": "hosting",
            "hosting": individual_deal_data.get("hostingDealDetails", {})
        }
    return None


@app.post("/deals/{deal_id}")
async def process_deal(deal_id: str, authorization: Annotated[str | None, Header()] = None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth")
    bearer_token = authorization.split(" ")[1]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://quantum.mtptest.co.uk/api/ai/deals/{deal_id}", headers={"Authorization": f"Bearer {bearer_token}"})
            response.raise_for_status()
            deal_data = response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Fetch error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")

    if not deal_data or "data" not in deal_data or "attachments" not in deal_data["data"] or not deal_data["data"]["attachments"]["default"]:
        raise HTTPException(status_code=404, detail="Deal/attachment not found")

    file_url = deal_data["data"]["attachments"]["default"][0]["url"]
    file_mime_type = deal_data["data"]["attachments"]["default"][0]["mime"]
    # Correctly extract filename from URL
    file_name = os.path.basename(urlparse(file_url).path)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Download error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Download request error: {e}")

    # --- File Handling and Conversion ---
    try:
        # Use NamedTemporaryFile with the original file name and appropriate suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(file_content)
            temp_file_path = tmp_file.name  # Get the path to the temporary file

        # Convert office documents to PDF if necessary
        if file_mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.presentationml.presentation"):
            pdf_file_path = convert_office_to_pdf(temp_file_path)  # Pass the temp file path
            with open(pdf_file_path, "rb") as f:
                file_content = f.read()
            file_mime_type = "application/pdf" # Update mime type after conversion.
            os.unlink(temp_file_path)  # Delete original temp file
            temp_file_path = pdf_file_path # Update temp file path


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File handling error: {e}")


    prompt_result = await process_sports_deals(file_content, file_mime_type)
    deals_dir = "../output/deals"
    deal_id_dir = os.path.join(deals_dir, deal_id)
    os.makedirs(deal_id_dir, exist_ok=True)
    deals_log_path = os.path.join(deal_id_dir, "deals_content.json")
    try:
        with open(deals_log_path, "w", encoding="utf8") as deals_log_file:
            json.dump({"deal_id": deal_id, "gemini_response": prompt_result}, deals_log_file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to {deals_log_path}: {e}")

    if prompt_result and prompt_result.get("response_data"):
        for i, individual_deal_data in enumerate(prompt_result["response_data"]):
            deal_types = individual_deal_data.get("deal_types", [])
            deal_type_counts = {}  # Track deal types within this individual_deal_data
            for deal_type in deal_types:
                submission_data = create_submission_data(deal_type, individual_deal_data)
                if not submission_data:
                    continue

                cleaned_data = clean_submission_data(submission_data)
                if cleaned_data is None:
                    print(f"Cleaning resulted in None for deal {i}, type {deal_type}.")
                    continue

                # File naming and indexing
                if deal_type not in deal_type_counts:
                    deal_type_counts[deal_type] = 0
                deal_index = deal_type_counts[deal_type]
                deal_type_counts[deal_type] += 1

                cleaned_log_path = os.path.join(deal_id_dir, f"deal_content_to_database_{i}_{deal_type}_{deal_index}.json")
                try:
                    with open(cleaned_log_path, "w", encoding="utf8") as log_file:
                        json.dump({"deal_id": deal_id, "response_index": i, "deal_type": deal_type, "deal_index": deal_index, "submission_data": cleaned_data}, log_file, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error writing to {cleaned_log_path}: {e}")
                print(f"Submission Payload (deal {i}, type {deal_type}, index {deal_index}) after cleaning: {cleaned_data}")

                try:
                    async with httpx.AsyncClient() as client:
                        post_response = await client.post(f"https://quantum.mtptest.co.uk/api/ai/deals/{deal_id}", headers={"Authorization": f"Bearer {bearer_token}", "Accept": "*/*", "Content-Type": "application/json"}, json=cleaned_data)
                        post_response.raise_for_status()
                        print(f"Deal {i}, type {deal_type}, index {deal_index} submitted successfully.")
                        success_file = os.path.join(deal_id_dir, f"submitted_deal_{i}_{deal_type}_{deal_index}.json")
                        with open(success_file, "w", encoding="utf8") as sf:
                            json.dump({"deal_id": deal_id, "response_index": i, "deal_type": deal_type, "deal_index": deal_index, "submitted_payload": cleaned_data}, sf, indent=2, ensure_ascii=False)
                except httpx.HTTPStatusError as e:
                    print(f"Error submitting deal {i}, type {deal_type}, index {deal_index}: {e.response.status_code} - {e.response.text}")
                except httpx.RequestError as e:
                    print(f"Request error submitting deal {i}, type {deal_type}, index {deal_index}: {e}")
                except Exception as e:
                    print(f"Unexpected error submitting deal {i}, type {deal_type}, index {deal_index}: {e}")
                # Clean up: Delete the temp file (either the original or the converted PDF)
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)


    return prompt_result

@app.get("/")
def health():
    return "API is working at 8084"

@app.post("/process-sports-deal/")
async def process_sports_deal(file: UploadFile = File(...)):
    """Endpoint for sports deals (broadcasting, sponsorship, hosting)."""
    try:
        original_filename = file.filename or "uploaded_file"
        file_extension = os.path.splitext(original_filename)[1]

        if not file_extension:
            file_extension = mimetypes.guess_extension(file.content_type) or ""
            if file_extension:
                original_filename += file_extension

        if file_extension and not file_extension.startswith("."):
            file_extension = "." + file_extension

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            temp_file_path = tmp_file.name

        file_mime_type = file.content_type

        pdf_file_path = convert_office_to_pdf(temp_file_path)

        with open(pdf_file_path, "rb") as f:
            file_content = f.read()

        gemini_mime_type = (
            "application/pdf" if pdf_file_path != temp_file_path else file_mime_type
        )

        prompt_result = await process_sports_deals(file_content, gemini_mime_type)
        
        os.unlink(temp_file_path)  # Always unlink the initial temp file
        if pdf_file_path != temp_file_path:
            os.unlink(pdf_file_path)

        return JSONResponse(content=prompt_result)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Internal server error: {str(e)}"}
        )