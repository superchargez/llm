from pathlib import Path
import os, re
import logging, traceback
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import mimetypes
import requests, json
import time
from prompt_utils import prompt_gen # copy it
from celery import Celery, states
from typing import List, Dict
import glob
from openai import OpenAI
from prompt_utils import json_prompt, prompt_gen
import aiohttp, asyncio
import redis
import datetime
from functools import lru_cache
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Load environment variables
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

# Configure API key for Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment")

genai.configure(api_key=API_KEY)

# Define generation configuration
generation_config = {
    "temperature": 0.0,  # Reduce randomness for accuracy
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are an assistant that extracts text, tables, charts, or content accurately from images provided. Always include the page number or context of the content in your extraction. If you find data in any language other than english then output as english."
)

improved_prompt = prompt_gen()

def ensure_directory_exists(directory: str):
    """Create the directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image(image_path: str, output_dir: str, page_number: Optional[int] = None):
    """Process a single image and extract content."""
    try:
        image = Image.open(image_path)
        extracted_content = f"Processed image content from page {page_number}"

        # Save markdown file
        page_md_path = os.path.join(output_dir, f"page_{page_number}.md")
        with open(page_md_path, "w") as md_file:
            md_file.write(extracted_content)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        logger.error(traceback.format_exc())
        raise

def convert_and_process_file(file_path: str, output_dir: str):
    """Convert non-image files (PDF, PPTX, DOCX) to images and process."""
    temp_pdf_path = os.path.join(output_dir, "temp_output.pdf")
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)

        # Conversion logic (e.g., for DOCX or PPTX to PDF)
        if file_path.lower().endswith(".pdf"):
            images = pdf_to_images(file_path, output_dir)
        else:
            convert_with_gotenberg(file_path, temp_pdf_path)
            images = pdf_to_images(temp_pdf_path, output_dir)
        os.remove(temp_pdf_path)

        # Process images
        for idx, image_path in enumerate(images, start=1):
            process_image(image_path, output_dir, page_number=idx)
    except Exception as e:
        logger.error(f"Error converting and processing file {file_path}: {e}")
        logger.error(traceback.format_exc())
        raise

def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Convert PDF pages to images."""
    from pymupdf import open as open_pdf

    image_paths = []
    try:
        pdf = open_pdf(pdf_path)
        for page_num, page in enumerate(pdf):
            image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix = page.get_pixmap(dpi=300)
            pix.save(image_path)
            image_paths.append(image_path)
        pdf.close()
        return image_paths
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        logger.error(traceback.format_exc())
        raise

def convert_with_gotenberg(file_path: str, output_pdf_path: str):
    """Convert file to PDF using Gotenberg API."""
    try:
        url = "http://localhost:3000/forms/libreoffice/convert"
        with open(file_path, 'rb') as file:
            response = requests.post(
                url,
                files={"files": file}
            )
            response.raise_for_status()

            with open(output_pdf_path, "wb") as output_file:
                output_file.write(response.content)
    except requests.RequestException as e:
        logger.error(f"Gotenberg API conversion failed: {e}")
        logger.error(traceback.format_exc())
        raise

# Processing limits and tracking
MAX_IMAGE_SIZE_MB = 2
PAUSE_THRESHOLD_MB = 20
MAX_IMAGES_BEFORE_PAUSE = 50
PAUSE_DURATION = 120  # seconds

# Global processing state
processing_state = {
    "total_processed_mb": 0,
    "processed_images_count": 0,
    "last_pause_time": 0
}

def reset_processing_state():
    """Reset the processing state after pause"""
    processing_state["total_processed_mb"] = 0
    processing_state["processed_images_count"] = 0
    processing_state["last_pause_time"] = time.time()

def check_and_handle_processing_limits(file_size_mb: float) -> bool:
    """
    Check if processing should pause based on limits.
    Returns True if processing can continue, False if file should be skipped.
    """
    current_time = time.time()
    
    # Skip if individual file is too large
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        logger.info(f"Skipping file: size {file_size_mb}MB exceeds {MAX_IMAGE_SIZE_MB}MB limit")
        return False
    
    if (processing_state["total_processed_mb"] >= PAUSE_THRESHOLD_MB or 
        processing_state["processed_images_count"] >= MAX_IMAGES_BEFORE_PAUSE):
        
        time_since_last_pause = current_time - processing_state["last_pause_time"]
        
        if time_since_last_pause < PAUSE_DURATION:
            remaining_pause = PAUSE_DURATION - time_since_last_pause
            logger.info(f"Pausing for {remaining_pause:.2f} seconds...")
            time.sleep(remaining_pause)
        
        # Reset state after pause
        reset_processing_state()
    
    # Update processing state
    processing_state["total_processed_mb"] += file_size_mb
    processing_state["processed_images_count"] += 1
    
    return True

def extract_image_content(image_path: str, page_number: Optional[int] = None) -> str:
    """
    Process an image if it meets size requirements and handles pausing when needed.
    """
    try:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)  # Convert to MB
        
        if not check_and_handle_processing_limits(file_size_mb):
            return f"> Page {page_number}: Image skipped (exceeds size limit)"
        
        logger.info(f"Processing image: {image_path} ({file_size_mb:.2f}MB)")
        image = Image.open(image_path)
        
        # Include page number in prompt if available
        context_prompt = improved_prompt
        if page_number is not None:
            context_prompt += f"\nThis is from page {page_number}"
        
        response = model.generate_content([context_prompt, image])
        extracted_text = response.text.strip()
        
        if not extracted_text or extracted_text.lower() == 'no content':
            return f"> Page {page_number}: No meaningful content found"
            
        return f"# > Page {page_number}\n{extracted_text}"
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Page {page_number}: Error extracting content"

##### Extracting and mapping JSON content ####
# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_EXPIRATION = 300  # 5 min

# API Configuration
BASE_URL = "https://quantum.mtptest.co.uk/api"
TIMEOUT_SECONDS = 30
SIMILARITY_THRESHOLD = 0.4

# In-memory cache for responses
response_cache = {}
source_emoji = {
    "redis": "🧱",
    "lru_cache": "💾",
    "api": "🌐"
}
def init_redis() -> redis.Redis:
    """Initialize Redis connection"""
    try:
        return redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
    except redis.ConnectionError as e:
        logger.warning(f"Failed to connect to Redis: {e}. Continuing without Redis caching.")
        return None

@lru_cache(maxsize=1000)
def get_cached_response(cache_key: str) -> Optional[dict]:
    """Get cached response from memory"""
    return response_cache.get(cache_key)

async def fetch_with_timeout(session: aiohttp.ClientSession, url: str, params: dict) -> dict:
    """Fetch data from API with timeout and improved error handling"""
    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    try:
        async with session.get(url, params=params, timeout=TIMEOUT_SECONDS) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"API request failed with status {response.status}. Response: {error_text}")
                return {"data": [], "error": f"HTTP {response.status}"}
    except asyncio.TimeoutError:
        logger.error(f"Request timed out for URL: {url}")
        return {"data": [], "error": "timeout"}
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error for URL {url}: {str(e)}")
        return {"data": [], "error": f"connection_error: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from {url}: {str(e)}")
        return {"data": [], "error": "invalid_json"}
    except Exception as e:
        logger.error(f"Unexpected error fetching data from {url}: {str(e)}")
        return {"data": [], "error": f"unexpected_error: {str(e)}"}

async def search_similarity(
    session: aiohttp.ClientSession,
    redis_client: redis.Redis,
    query: str,
    entity_type: str,
    taxonomy: str = None
) -> dict:
    """Search for similar events or metrics with caching"""
    cache_key = f"{entity_type}:{query}"

    # Check local in-memory cache first (lru_cache)
    cached = get_cached_response(cache_key)
    if cached:
        logger.info(f"{source_emoji['lru_cache']} Searching for '{query}' in {entity_type}: Found in LRU Cache")
        return cached

    # Try Redis cache next
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            response = json.loads(cached)
            response_cache[cache_key] = response  # Update local cache
            logger.info(f"{source_emoji['redis']} Searching for '{query}' in {entity_type}: Found in Redis")
            return response

    # Prepare API request
    params = {
        "q": query,
        "compact": "true",
        "limit": 1,
        "select": "name",
        "threshold": SIMILARITY_THRESHOLD
    }

    if taxonomy:
        params["taxonomy"] = taxonomy

    url = f"{BASE_URL}/embeddings/{entity_type}"
    if entity_type == "Event":
        url += "/name"

    logger.info(f"{source_emoji['api']} Searching for '{query}' in {entity_type}: Querying external API")
    result = await fetch_with_timeout(session, url, params)

    # Cache the result in both Redis and local cache
    if redis_client and result.get("data"):
        redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
    response_cache[cache_key] = result

    return result

class DateParser:
    """A simplified and robust date parser that handles various date formats."""

    MONTH_NAMES = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    @classmethod
    def _get_month_number(cls, month_str: str) -> int:
        """Convert month name to number."""
        return cls.MONTH_NAMES.get(month_str[:3].lower(), 0)

    @classmethod
    def parse_date(cls, date_str: str) -> Optional[str]:
        """Parse a date string and return it in ISO format (YYYY-MM-DD)."""
        if not date_str:
            return None

        try:
            date_str = re.sub(r'[\u2013\u2014]', '-', date_str.strip().lower())

            # Patterns and handling logic
            patterns = [
                (r'(\d{4})-(\d{2})-(\d{2})', lambda y, m, d: (int(y), int(m), int(d))),
                (r'(\d{2})/(\d{2})/(\d{4})', lambda d, m, y: (int(y), int(m), int(d))),
                (r'(\d{2})\s+([a-z]+)\s+(\d{4})', lambda d, m, y: (int(y), cls._get_month_number(m), int(d))),
                (r'([a-z]+)\s+(\d{4})', lambda m, y: (int(y), cls._get_month_number(m), 1))
            ]

            for pattern, extractor in patterns:
                match = re.match(pattern, date_str)
                if match:
                    year, month, day = extractor(*match.groups())
                    return datetime.datetime(year, month, day).strftime('%Y-%m-%d')

            logger.warning(f"Could not parse date string: {date_str}")
            return None

        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
            return None

async def process_metric(
    session: aiohttp.ClientSession,
    redis_client: redis.Redis,
    metric: dict,
    event_name: str,
    event_date: str
) -> Optional[dict]:
    """Process a single metric and create mapping for both event and metric"""
    # Search for similar event
    event_search = await search_similarity(
        session,
        redis_client,
        event_name,
        "Event"
    )
    
    # Get mapped event data from search results
    mapped_event = event_search["data"][0] if event_search.get("data") else {
        "name": None,
        "id": None,
        "_distance": 0
    }

    # Search for similar metric
    metric_search = await search_similarity(
        session,
        redis_client,
        metric["name"],
        "Metric",
        "event"
    )
    
    # Get mapped metric data from search results
    mapped_metric = metric_search["data"][0] if metric_search.get("data") else {
        "name": None,
        "id": None,
        "_distance": 0
    }
    
    return {
        "taxonomy": {
            "type": "Event",
            "name": event_name
        },
        "mappedTaxonomy": {
            "type": "Event",
            "name": mapped_event.get("name"),
            "id": mapped_event.get("id"),
            "similarity": mapped_event.get("_distance", 0)
        },
        "metric": metric["name"],
        "mappedMetric": {
            "name": mapped_metric.get("name"),
            "id": mapped_metric.get("id"),
            "similarity": mapped_metric.get("_distance", 0)
        },
        "valueType": "Estimated",
        "valueMin": metric.get("value"),
        "valueMax": None,
        "currency": metric.get("currency"),
        "date": DateParser.parse_date(event_date)
    }

def transform_mappings(page_mappings: list) -> list:
    """
    Transform page mappings into the desired flat structure.
    
    Args:
        page_mappings (list): List of page mappings with nested events and metrics
        
    Returns:
        list: Flattened list of mappings in the desired format
    """
    transformed_mappings = []
    
    for event in page_mappings:
        for metric in event.get("metrics", []):
            transformed_mapping = {
                "taxonomy": metric["taxonomy"],
                "mappedTaxonomy": metric["mappedTaxonomy"],
                "metric": metric["metric"],
                "mappedMetric": metric["mappedMetric"],
                "valueType": metric["valueType"],
                "valueMin": metric["valueMin"],
                "valueMax": metric["valueMax"],
                "currency": metric["currency"],
                "date": metric["date"]
            }
            transformed_mappings.append(transformed_mapping)
    
    return transformed_mappings

def transform_and_filter_mappings(transformed_mappings):
    """
    Filter out events from transformed mappings where either mappedTaxonomy.name 
    or mappedMetric.name is null.
    
    Args:
        transformed_mappings (list): List of dictionaries containing event mappings
        
    Returns:
        list: Filtered list of mappings excluding entries with null names
    """
    filtered_mappings = [
        mapping for mapping in transformed_mappings
        if (mapping.get('mappedTaxonomy', {}).get('name') is not None and 
            mapping.get('mappedMetric', {}).get('name') is not None)
    ]
    
    return filtered_mappings

from pydantic import BaseModel
class MetricsResponse(BaseModel):
    success: bool
    status_code: int
    message: str

class APIResponse(BaseModel):
    success: bool
    job_id: str
    results: List[MetricsResponse] | MetricsResponse | Dict[str, str]

async def post_mapping_file(file_path: Path, job_id: str) -> List[MetricsResponse] | MetricsResponse:
    """Process the metrics file and post to database"""
    try:
        metrics_content = json.loads(file_path.read_text())
        
        if not metrics_content:
            wrapped_content = {"extraction": []}
            response = requests.post(
                f"https://quantum.mtptest.co.uk/api/ai/data/{job_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(wrapped_content)
            )
            return MetricsResponse(
                success=response.status_code == 200,
                status_code=response.status_code,
                message=response.content.decode()
            )

        # Process in chunks
        chunk_size = 30
        chunks = [metrics_content[i:i + chunk_size] for i in range(0, len(metrics_content), chunk_size)]
        
        responses = []
        for chunk in chunks:
            wrapped_content = {"extraction": chunk}
            response = requests.post(
                f"https://quantum.mtptest.co.uk/api/ai/data/{job_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(wrapped_content)
            )
            responses.append(
                MetricsResponse(
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    message=response.content.decode()
                )
            )
        
        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from enum import Enum
class ProcessingStage(str, Enum):
    INITIALIZING = "initializing"
    IMAGE_EXTRACTION = "image_extraction"
    CONTENT_PROCESSING = "content_processing"
    METRICS_EXTRACTION = "metrics_extraction"
    MAPPING_GENERATION = "mapping_generation"
    FINAL_PROCESSING = "final_processing"
    COMPLETED = "completed"
    FAILED = "failed"

@celery_app.task(bind=True)
def process_file_task(self, file_path: str, output_dir: str, base_filename: str):
    """Celery task to process the uploaded file with status and progress tracking."""
    try:
        self.update_state(state='PROGRESS',
                         meta={'stage': 'INITIALIZING',
                               'progress': 0,
                               'current_operation': "Initializing task",
                               'details': None})

        ensure_directory_exists(output_dir)
        content_pages = []

        reset_processing_state()

        mime_type, _ = mimetypes.guess_type(file_path)

        self.update_state(state='PROGRESS',
                         meta={'stage': 'IMAGE_EXTRACTION',
                               'progress': 10,
                               'current_operation': "Converting document to images",
                               'details': {'mime_type': mime_type}})

        if mime_type and mime_type.startswith("image/"):
            extracted_content = extract_image_content(file_path, page_number=1)
            content_pages.append({"page": 1, "content": extracted_content})

            self.update_state(state='PROGRESS',
                            meta={'stage': 'CONTENT_PROCESSING',
                                  'progress': 30,
                                  'current_operation': "Processing single image",
                                  'details': {'current_page': 1, 'total_pages': 1}})

            page_md_path = os.path.join(output_dir, "page_1.md")
            with open(page_md_path, "w") as md_file:
                md_file.write(extracted_content)
        else:
            temp_pdf_path = os.path.join(output_dir, "temp_output.pdf")
            if mime_type == "application/pdf":
                temp_pdf_path = file_path

            image_paths = pdf_to_images(temp_pdf_path, output_dir)
            total_pages = len(image_paths)

            for idx, image_path in enumerate(image_paths, start=1):
                self.update_state(state='PROGRESS',
                                meta={'stage': 'CONTENT_PROCESSING',
                                      'progress': 30 + (20 * idx // total_pages),
                                      'current_operation': "Processing document pages",
                                      'details': {'current_page': idx, 'total_pages': total_pages}})

                page_content = extract_image_content(image_path, page_number=idx)
                content_pages.append({"page": idx, "content": page_content})

                page_md_path = os.path.join(output_dir, f"page_{idx}.md")
                with open(page_md_path, "w") as md_file:
                    md_file.write(page_content)

        # Save combined Markdown save to combined md
        combined_md_output_path = os.path.join(output_dir, f"{base_filename}_extraction.md")
        with open(combined_md_output_path, "w") as md_file:
            for page in content_pages:
                md_file.write(f"{page['content']}\n\n")

        self.update_state(state='PROGRESS',
                         meta={'stage': 'METRICS_EXTRACTION',
                               'progress': 50,
                               'current_operation': "Extracting metrics from content",
                               'details': None})

        pages = read_md_files(output_dir)
        all_results = []
        for idx, page in enumerate(pages, 1):
            self.update_state(state='PROGRESS',
                            meta={'stage': 'METRICS_EXTRACTION',
                                  'progress': 50 + (20 * idx // total_pages),
                                  'current_operation': "Processing metrics",
                                  'details': {'current_page': idx, 'total_pages': total_pages}})

            page_results = extract_metrics(
                content=page['content'],
                # client=client, # openai's client depricated
                model=model,
                prompt=json_prompt,
                page_num=page['page_num']
            )

            output_filename = page['filename'].replace('.md', '.json')
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(page_results, f, indent=2, ensure_ascii=False)
            all_results.append(page_results)

        combined_json_file = os.path.join(output_dir, 'combined.json')
        with open(combined_json_file, 'w', encoding='utf-8') as f:
            json.dump({"pages": all_results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully processed combined JSON and saved to {combined_json_file}")
        total_pages = len(pages)
        self.update_state(state='PROGRESS',
                         meta={'stage': 'MAPPING_GENERATION',
                               'progress': 70,
                               'current_operation': "Generating mappings",
                               'details': None})

        mappings = []

        async def process_file():
            redis_client = init_redis()
            async with aiohttp.ClientSession() as session:
                for idx, page in enumerate(all_results, start=1):
                    self.update_state(state='PROGRESS',
                                    meta={'stage': 'MAPPING_GENERATION',
                                          'progress': 70 + (20 * idx // len(all_results)),
                                          'current_operation': "Processing mappings",
                                          'details': {'current_page': idx, 'total_pages': len(all_results)}})
                    page_mappings = {
                        "page": idx,
                        "events": []
                    }

                    for event in page.get("events", []):
                        event_name = event.get("name", "")
                        event_date = event.get("date", "")

                        tasks = [
                            process_metric(session, redis_client, metric, event_name, event_date)
                            for metric in event.get("metrics", [])
                        ]

                        results = await asyncio.gather(*tasks)
                        event_mappings = {
                            "event_name": event_name,
                            "event_date": event_date,
                            "metrics": [r for r in results if r]
                        }
                        page_mappings["events"].append(event_mappings)

                    # Save mappings to individual page JSON files
                    output_filename = os.path.join(output_dir, f"mapping_page_{idx}.json")
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(page_mappings, f, indent=2)

                    mappings.extend(page_mappings["events"])

        asyncio.run(process_file())

        self.update_state(state='PROGRESS',
                         meta={'stage': 'FINAL_PROCESSING',
                               'progress': 90,
                               'current_operation': "Finalizing processing",
                               'details': None})

        transformed_mappings = transform_mappings(mappings)

        combined_mapping_file = os.path.join(output_dir, 'combined_mapping.json')
        with open(combined_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_mappings, f, indent=2)

        logger.info(f"Successfully processed combined JSON and saved mappings to {output_dir}")

        filtered_mappings = transform_and_filter_mappings(transformed_mappings)
        filtered_mapping_file = os.path.join(output_dir, 'transformed_mapping.json')
        with open(filtered_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_mappings, f, indent=2)
        job_id = base_filename
        filtered_mapping_file = Path(filtered_mapping_file)
        result = asyncio.run(post_mapping_file(filtered_mapping_file, job_id))
        logger.info(f"Posted combined_mapping.json to database: {result}")


        self.update_state(state='SUCCESS',
                         meta={'stage': 'COMPLETED',
                               'progress': 100,
                               'current_operation': "Processing completed",
                               'details': {'output_dir': output_dir}})

        return {"status": "success", "output_dir": output_dir}

    except Exception as e:
        logger.error(f"Task error: {str(e)}")
        self.update_state(state='FAILURE',
                         meta={'stage': 'FAILED',
                               'progress': 0,
                               'current_operation': "Processing failed",
                               'details': {'error': str(e)}})
        raise

#### Extracting Json content from MD files ####
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)
prompt = prompt_gen(json_prompt)

def read_md_files(output_dir: str) -> List[Dict[str, str]]:
    """Read all MD files in the directory and return their contents."""
    pages = []
    # Get all .md files
    for md_file in glob.glob(os.path.join(output_dir, "*.md")):
        base_name = os.path.basename(md_file)
        if '_extract.md' in base_name:
            continue
            
        # Extract page number from filename
        page_match = re.search(r'page_(\d+)', base_name)
        if page_match:
            page_num = page_match.group(1)
            with open(md_file, 'r', encoding='utf-8') as f:
                pages.append({
                    'filename': base_name,
                    'page_num': page_num,
                    'content': f.read().strip()
                })
    
    return sorted(pages, key=lambda x: int(x['page_num']))

# This is important Openai's API don't delete
# def extract_metrics(content: str, client: OpenAI, prompt: str, page_num: int) -> Dict:
#     """Extract metrics and their associated events using LLM."""
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a sports event analyst. Extract event details and metrics from the provided content."},
#                 {"role": "user", "content": f"{prompt}\n\nContent:\n{content}"}
#             ], temperature=0.0,
#             response_format={"type": "json_object"}
#         )
        
#         summary_raw = response.choices[0].message.content.strip()
#         summary_json = re.sub(r'```json|```', '', summary_raw).strip()
        
#         parsed_response = json.loads(summary_json)
#         if 'page_number' not in parsed_response:
#             parsed_response['page_number'] = page_num
            
#         # Validate response structure
#         if 'events' not in parsed_response:
#             logger.warning(f"No events found in response for page {page_num}")
#             parsed_response['events'] = []
            
#         return parsed_response
        
#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse JSON response for page {page_num}: {summary_raw}")
#         logger.error(f"JSON Error: {str(e)}")
#         logger.error(traceback.format_exc())
#         return {"page_number": page_num, "events": []}
#     except Exception as e:
#         logger.error(f"Unexpected error processing page {page_num}: {str(e)}")
#         logger.error(traceback.format_exc())
#         return {"page_number": page_num, "events": []}

def extract_metrics(content: str, model: genai.GenerativeModel, prompt: str, page_num: int) -> Dict:
    """Extract metrics and their associated events using Gemini generative mode."""
    try:
        response = model.generate_content(
            contents = f"{prompt}\n\nContent:\n{content}",
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192
            )
        )


        summary_raw = response.text.strip()
        summary_json = re.sub(r'```json|```', '', summary_raw).strip()
        summary_json = summary_json.replace("'", '"')
        parsed_response = json.loads(summary_json)
        return parsed_response
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for page {page_num}: {summary_raw}")
        logger.error(f"JSON Error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"page_number": page_num, "events": []}

#### Main API Endpoints ####
@app.post("/data/{id}")
async def extract_content(id: str):
    """Endpoint to download and process a file by ID."""
    try:
        # First get file details from the API
        details_url = f"http://quantum.mtptest.co.uk/api/ai/data/{id}"
        headers = {
            "Authorization": "Bearer b790736a5a75be213c4d752d51efe9cb7d05041a"
        }

        # Create output directory using ID
        output_dir = os.path.join("../output", id)
        ensure_directory_exists(output_dir)

        try:
            # Fetch file details first
            details_response = requests.get(details_url, headers=headers, timeout=20)
            details_response.raise_for_status()
            
            # Parse the response
            data = details_response.json().get("data")
            if not data or "attachments" not in data:
                return JSONResponse(
                    content={"error": "No attachments found for the given ID"}, 
                    status_code=400
                )

            # Get file details from response
            attachment = data["attachments"]["default"][0]
            file_url = attachment["url"]
            mime_type = attachment["mime"]

            if not file_url:
                return JSONResponse(
                    content={"error": "File URL not found in response"}, 
                    status_code=400
                )

            # Download the actual file
            file_response = requests.get(file_url, headers=headers, timeout=20)
            file_response.raise_for_status()

        except requests.exceptions.Timeout:
            return JSONResponse(
                content={"error": "Request timed out while fetching file details"}, 
                status_code=504
            )
        except requests.exceptions.RequestException as e:
            return JSONResponse(
                content={"error": f"Failed to fetch file: {str(e)}"}, 
                status_code=500
            )

        # Determine file extension based on mime type
        extension = mimetypes.guess_extension(mime_type) or '.bin'
        
        # Save downloaded file
        file_path = os.path.join(output_dir, f"{id}{extension}")
        with open(file_path, "wb") as f:
            f.write(file_response.content)
        
        # Log successful file save
        logger.info(f"Successfully downloaded and saved file with ID {id} to {file_path}")
        
        # Handle different file types
        if mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # docx
                        'application/vnd.openxmlformats-officedocument.presentationml.presentation']:  # pptx
            # Convert to PDF first
            temp_pdf_path = os.path.join(output_dir, "temp_output.pdf")
            try:
                convert_with_gotenberg(file_path, temp_pdf_path)
                logger.info(f"Successfully converted file with ID {id} to PDF")
                
                # Start Celery task with the PDF path
                task = process_file_task.apply_async(args=[temp_pdf_path, output_dir, id])
                return {"message": "Processing started.", "task_id": task.id, "output_directory": output_dir}
                
            except Exception as e:
                logger.error(f"Error converting file to PDF: {str(e)}")
                return JSONResponse(
                    content={"error": f"Failed to convert file to PDF: {str(e)}"}, 
                    status_code=500
                )
                
        elif mime_type == 'application/pdf':
            # Process PDF directly
            task = process_file_task.apply_async(args=[file_path, output_dir, id])
            return {"message": "Processing started.", "task_id": task.id, "output_directory": output_dir}
            
        elif mime_type and mime_type.startswith('image/'):
            # Process image directly
            task = process_file_task.apply_async(args=[file_path, output_dir, id])
            return {"message": "Processing started.", "task_id": task.id, "output_directory": output_dir}
            
        else:
            return JSONResponse(
                content={"error": f"Unsupported file type: {mime_type}"}, 
                status_code=400
            )

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up downloaded file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file: {str(cleanup_error)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Enhanced status endpoint that provides detailed progress information"""
    try:
        from rich import print
        task = process_file_task.AsyncResult(task_id)
        print(f"Task state from Redis: {task.state}")  # Crucial logging
        print(f"Task info from Redis: {task.info}")
        if task.state == states.PENDING:
            return {
                "state": "PENDING",
                "stage": "pending",
                "progress": 0,
                "current_operation": "Task is pending execution",
                "details": None
            }
            
        elif task.state == states.FAILURE:
            return {
                "state": "FAILURE",
                "stage": ProcessingStage.FAILED,
                "progress": 0,
                "current_operation": "Task failed",
                "details": {
                    "error": str(task.info.get('error', "Unknown error occurred"))
                }
            }
            
        elif task.state == 'PROGRESS':
            return {
                "state": "PROGRESS",
                "stage": task.info.get('stage', ProcessingStage.INITIALIZING),
                "progress": task.info.get('progress', 0),
                "current_operation": task.info.get('current_operation', "Processing"),
                "details": task.info.get('details', {})
            }
            
        elif task.state == states.SUCCESS:
            return {
                "state": "SUCCESS",
                "stage": ProcessingStage.COMPLETED,
                "progress": 100,
                "current_operation": "Task completed successfully",
                "details": task.info
            }
            
        return {
            "state": task.state,
            "stage": "unknown",
            "progress": 0,
            "current_operation": "Unknown state",
            "details": None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving task status: {str(e)}"
        )

@app.get("/display-raw/{filename}", response_class=HTMLResponse)
async def display_raw_md(filename: str):
    from jinja2 import Template
    try:
        output_dir = os.path.join("../output", filename)
        combined_md_output_path = os.path.join(output_dir, f"{filename}_extraction.md")
        processing_flag_path = os.path.join(output_dir, ".processing")
        error_flag_path = os.path.join(output_dir, ".error")

        # Check if the output directory exists
        if not os.path.exists(output_dir):
            return HTMLResponse(
                content="<h1>Process Not Started</h1><p>The extraction process has not been initiated for this file.</p>",
                status_code=404
            )

        # Check for error flag
        if os.path.exists(error_flag_path):
            with open(error_flag_path, 'r') as error_file:
                error_message = error_file.read()
            return HTMLResponse(
                content=f"<h1>Processing Error</h1><p>The extraction process encountered an error: {error_message}</p>",
                status_code=500
            )

        # Check for processing flag
        if os.path.exists(processing_flag_path):
            return HTMLResponse(
                content="""
                    <h1>Processing in Progress</h1>
                    <p>Your file is currently being processed. Please check back in a few moments.</p>
                    <script>
                        setTimeout(function() {
                            window.location.reload();
                        }, 5000);  // Refresh every 5 seconds
                    </script>
                """,
                status_code=202
            )

        # Check if the output file exists
        if not os.path.exists(combined_md_output_path):
            return HTMLResponse(
                content="<h1>Processing Not Complete</h1><p>The extraction process hasn't generated the output file yet.</p>",
                status_code=404
            )

        # If we get here, the file exists and we can display it
        with open(combined_md_output_path, "r") as md_file:
            extracted_content = md_file.read()

        html_content = markdown.markdown(extracted_content, extensions=["tables", "fenced_code"])

        template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Markdown Content</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:hover { background-color: #f1f1f1; }
            </style>
        </head>
        <body>
            {{ html_content | safe }}
        </body>
        </html>
        """
        rendered_html = Template(template).render(html_content=html_content)
        return HTMLResponse(content=rendered_html)

    except Exception as e:
        logger.error(f"Display error: {e}")
        return HTMLResponse(
            content=f"<h1>Unexpected Error</h1><p>An error occurred while displaying the content: {e}</p>",
            status_code=500
        )

@app.get("/display-json/{id}", response_class=HTMLResponse)
async def display_content_content(id: str):
    from jinja2 import Template
    try:
        output_dir = os.path.join("../output", id)
        combined_json_output_path = os.path.join(output_dir, f"combined_mapping.json")
        processing_flag_path = os.path.join(output_dir, ".processing")
        error_flag_path = os.path.join(output_dir, ".error")

        # Check if the output directory exists
        if not os.path.exists(output_dir):
            return HTMLResponse(
                content="<h1>Process Not Started</h1><p>The extraction process has not been initiated for this file.</p>",
                status_code=404
            )

        # Check for error flag
        if os.path.exists(error_flag_path):
            with open(error_flag_path, 'r') as error_file:
                error_message = error_file.read()
            return HTMLResponse(
                content=f"<h1>Processing Error</h1><p>The extraction process encountered an error: {error_message}</p>",
                status_code=500
            )

        # Check for processing flag
        if os.path.exists(processing_flag_path):
            return HTMLResponse(
                content="""
                    <h1>Processing in Progress</h1>
                    <p>Your file is currently being processed. Please check back in a few moments.</p>
                    <script>
                        setTimeout(function() {
                            window.location.reload();
                        }, 5000);  // Refresh every 5 seconds
                    </script>
                """,
                status_code=202
            )

        # Check if the output file exists
        if not os.path.exists(combined_json_output_path):
            return HTMLResponse(
                content="<h1>Processing Not Complete</h1><p>The extraction process hasn't generated the output file yet.</p>",
                status_code=404
            )

        with open(combined_json_output_path, "r") as md_file:
            extracted_content = md_file.read()

        return JSONResponse(content=json.loads(extracted_content))

    except Exception as e:
        logger.error(f"Display error: {e}")
        return HTMLResponse(
            content=f"<h1>Unexpected Error</h1><p>An error occurred while displaying the content: {e}</p>",
            status_code=500
        )
