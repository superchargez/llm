import json
import aiohttp
import asyncio
from datetime import datetime
# import iso4217parse
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for API responses
event_cache: Dict[str, Any] = {}
metric_cache: Dict[str, Any] = {}

# Base URL for API calls
BASE_URL = "https://quantum.mtptest.co.uk/api/embeddings"

async def fetch_similar_event(session: aiohttp.ClientSession, event_name: str) -> dict:
    """Fetch similar event from API with caching."""
    if event_name in event_cache:
        logger.info(f"Cache hit for event: {event_name}")
        return event_cache[event_name]
    
    url = f"{BASE_URL}/Event/name"
    params = {
        'q': event_name,
        'compact': 'true',
        'select': 'name',
        'limit': 1,
        'threshold': 0.4
    }
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('data'):
                    result = data['data'][0]
                    event_cache[event_name] = result
                    return result
            return None
    except Exception as e:
        logger.error(f"Error fetching event {event_name}: {str(e)}")
        return None

async def fetch_similar_metric(session: aiohttp.ClientSession, metric_name: str) -> dict:
    """Fetch similar metric from API with caching."""
    if metric_name in metric_cache:
        logger.info(f"Cache hit for metric: {metric_name}")
        return metric_cache[metric_name]
    
    url = f"{BASE_URL}/Metric"
    params = {
        'q': metric_name,
        'compact': 'true',
        'select': 'name',
        'taxonomy': 'event',
        'limit': 1,
        'threshold': 0.4
    }
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('data'):
                    result = data['data'][0]
                    metric_cache[metric_name] = result
                    return result
            return None
    except Exception as e:
        logger.error(f"Error fetching metric {metric_name}: {str(e)}")
        return None

async def process_metric(session: aiohttp.ClientSession, event_data: dict, metric_name: str, metric_data: dict) -> dict:
    """Process a single metric and create the required output format."""
    # Get similar event and metric from API
    similar_event = await fetch_similar_event(session, event_data['event_name'])
    similar_metric = await fetch_similar_metric(session, metric_name)
    
    # Convert currency code to ISO format
    currency_code = metric_data.get('currency')
    # if currency_code:
    #     try:
    #         currency = iso4217parse.parse(currency_code)
    #         currency_code = currency.code
    #     except:
    #         currency_code = metric_data.get('currency')

    # Format date
    date_str = event_data.get('event_date')
    try:
        # Assuming date format is YYYY/YY or similar
        date_obj = datetime.strptime(date_str.split('/')[0], '%Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')
    except:
        formatted_date = datetime.now().strftime('%Y-%m-%d')

    return {
        "taxonomy": {
            "type": "Event",
            "name": event_data['event_name']
        },
        "mappedTaxonomy": {
            "type": "Event",
            "name": similar_event['name'] if similar_event else None,
            "id": similar_event['id'] if similar_event else None,
            "similarity": similar_event['_distance'] if similar_event else None
        },
        "metric": metric_name,
        "mappedMetric": {
            "name": similar_metric['name'] if similar_metric else None,
            "id": similar_metric['id'] if similar_metric else None,
            "similarity": similar_metric['_distance'] if similar_metric else None
        },
        "valueType": "Estimated",
        "valueMin": metric_data['value'],
        "valueMax": None,
        "currency": currency_code,
        "date": formatted_date
    }

async def process_json_file(input_file: str, output_file: str):
    """Main function to process the JSON file."""
    try:
        # Read input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)

        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []

            # Process each slide
            for slide_data in data['slides'].values():
                for event in slide_data:
                    # Process each metric in the event
                    for metric_name, metric_data in event['metrics'].items():
                        task = process_metric(session, event, metric_name, metric_data)
                        tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out any errors
            results = [r for r in results if isinstance(r, dict)]

        # Write results to output file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processing complete. Results written to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")

# Run the async function
if __name__ == "__main__":
    input_file = "combined_slides_metrics.json"
    input_file = "../extracted_content/everything/slide/markdowns/combined_slides_metrics.json"
    output_file = "../extracted_content/everything/slide/markdowns/processed_metrics.json"

    asyncio.run(process_json_file(input_file, output_file))