prompt = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

1. **Event-Based Organization**:
   - Treat each year or distinct timeframe (e.g., "2008/09", "2009/10") as a separate event.
   - For each event, include:
     - A descriptive `event_name` that combines the main context (e.g., "German Bundesliga", "English Premier League") with the specific year or range of years (e.g., "2008/09", "2017").
       - **Do not include any metric names (e.g., "Net Debt") in the `event_name`. Metrics belong in the `metrics` key.**
     - An `event_date` field corresponding to the year or range of years.

2. **Metrics Grouping**:
   - For each event, group all relevant metrics under a `metrics` key.
   - Each metric should include:
     - A descriptive key (e.g., "Match Revenue from Ticket Sales", "Advertising Revenue from Sponsors").
     - The numeric `value`.
     - The `currency` key using the ISO 3-letter code (e.g., EUR for euros). (Optional)
  - NOTE: It is important to skip currency field/key if it not a currency or does not make sense to represent a value by currency symbol, for example in case of number of people or percent of some other value.

3. **Enhanced Descriptive Metric Names**:
   - Ensure that all metric names are unambiguous and self-explanatory.
   - Avoid generic or vague names such as "Other" or "Advertising". Instead:
     - Add contextual information to clarify the metric’s meaning (e.g., "Other Costs for Operations", "Advertising Revenue from Sponsors").
     - When possible, include details from the input (e.g., category names, associated segments, or specific activities) to make the names meaningful.
   - Example:
     - Instead of "Youth, amateurs, academies", use "Costs for Youth, Amateur, and Academy Development".
     - Instead of "Other", use "Other Costs for Match Operations" or "Other Revenue from Ancillary Services".

4. **Comprehensive Extraction**:
   - Extract metrics directly from both the slide text and the spreadsheet data, without inferring or calculating values not explicitly mentioned.
   - Include all rows and columns of provided data, unless explicitly excluded in the input.

5. **Table of Contents and Non-Metric Slides**:
   - If the input content represents a table of contents, an agenda, or contains no explicit metrics, **exclude it from processing and output generation**.
   - Identify these slides based on their lack of detailed data, their generic structure, or headings such as "Table of Contents", "Agenda", or "Overview".

6. **Output Format**:
   - The output should be a JSON array of events, where each event is a separate JSON object.
   - Example:
[
  {
    "event_name": "English Premier League 2017",
    "event_date": "2017",
    "metrics": {
      "Net Debt of Premier League Clubs": {
        "value": 1954,
        "currency": "GBP"
      },
      "Net Debt of Championship Clubs": {
        "value": 1812,
        "currency": "GBP"
      }
    }
  }
]

7. Validation: 
Ensure metric names are consistent with their category (e.g., revenue, costs, operations).
Verify that each metric name aligns with the broader context provided in the input.

8. No Inference or Aggregation: Do not infer metrics or calculate totals. Only extract metrics as provided in the input.

9. Adaptability:If the input mentions costs, revenues, or specific activities, integrate these details into the metric names to avoid ambiguity.

Example: If the input includes "Advertising", clarify whether it refers to "Advertising Costs" or "Advertising Revenue" based on context.

"""
prompt_092 = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

1. **Event-Based Organization**:
   - Treat each year or distinct timeframe (e.g., "2008/09", "2009/10") as a separate event.
   - For each event, include:
     - A descriptive `event_name` that combines the main context (e.g., "German Bundesliga") and the specific year or range of years (e.g., "2008/09").
     - An `event_date` field corresponding to the year or range of years.

2. **Metrics Grouping**:
   - For each event, group all relevant metrics under a `metrics` key.
   - Each metric should include:
     - A descriptive key (e.g., "Match Revenue from Ticket Sales", "Advertising Revenue from Sponsors").
     - The numeric `value`.
     - The `currency` key using the ISO 3-letter code (e.g., EUR for euros).

3. **Enhanced Descriptive Metric Names**:
   - Ensure that all metric names are unambiguous and self-explanatory.
   - Avoid generic or vague names such as "Other" or "Advertising". Instead:
     - Add contextual information to clarify the metric’s meaning (e.g., "Other Costs for Operations", "Advertising Revenue from Sponsors").
     - When possible, include details from the input (e.g., category names, associated segments, or specific activities) to make the names meaningful.
   - Example:
     - Instead of "Youth, amateurs, academies", use "Costs for Youth, Amateur, and Academy Development".
     - Instead of "Other", use "Other Costs for Match Operations" or "Other Revenue from Ancillary Services".

4. **Comprehensive Extraction**:
   - Extract metrics directly from both the slide text and the spreadsheet data, without inferring or calculating values not explicitly mentioned.
   - Include all rows and columns of provided data, unless explicitly excluded in the input.

5. **Output Format**:
   - The output should be a JSON array of events, where each event is a separate JSON object.
   - Example:

```json
[
  {
    "event_name": "German Bundesliga 2008/09",
    "event_date": "2008/09",
    "metrics": {
      "Match Revenue from Ticket Sales": {
        "value": 363.4,
        "currency": "EUR"
      },
      "Advertising Revenue from Sponsors": {
        "value": 488.77,
        "currency": "EUR"
      },
      "Other Revenue from Ancillary Services": {
        "value": 164.24,
        "currency": "EUR"
      }
    }
  }
]

6. Validation: Ensure metric names are consistent with their category (e.g., revenue, costs, operations).
Verify that each metric name aligns with the broader context provided in the input.

7. No Inference or Aggregation: Do not infer metrics or calculate totals. Only extract metrics as provided in the input.

8. Adaptability: If the input mentions costs, revenues, or specific activities, integrate these details into the metric names to avoid ambiguity.

Example: If the input includes "Advertising", clarify whether it refers to "Advertising Costs" or "Advertising Revenue" based on context. """

prompt_093 = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

1. **Event-Based Organization**:
   - Treat each year or distinct timeframe (e.g., "2008/09", "2009/10") as a separate event.
   - For each event, include:
     - A descriptive `event_name` that combines the main context (e.g., "German Bundesliga") and the specific year or range of years (e.g., "2008/09").
     - An `event_date` field corresponding to the year or range of years.

2. **Metrics Grouping**:
   - For each event, group all relevant metrics (e.g., revenue, advertising, transfers) under a `metrics` key.
   - Each metric should include:
     - A descriptive key (e.g., "Match Revenue", "Advertising Revenue").
     - The numeric `value`.
     - The `currency` key using the ISO 3-letter code (e.g., EUR for euros).

3. **Comprehensive Extraction**:
   - Extract metrics directly from both the slide text and the spreadsheet data, without inferring or calculating values not explicitly mentioned.
   - Include all rows and columns of provided data, unless explicitly excluded in the input.

4. **Output Format**:
   - The output should be a JSON array of events, where each event is a separate JSON object.
   - Example:
   [
  {
    "event_name": "German Bundesliga 2008/09",
    "event_date": "2008/09",
    "metrics": {
      "Match Revenue": {
        "value": 363.4,
        "currency": "EUR"
      },
      "Advertising Revenue": {
        "value": 488.77,
        "currency": "EUR"
      }
    }
  },
  {
    "event_name": "German Bundesliga 2009/10",
    "event_date": "2009/10",
    "metrics": {
      "Match Revenue": {
        "value": 379.29,
        "currency": "EUR"
      },
      "Advertising Revenue": {
        "value": 511.89,
        "currency": "EUR"
      }
    }
  }
]

5. Naming Guidelines:

event_name should combine the league or competition name and the specific year or range (e.g., "German Bundesliga 2008/09").
Metric names should be concise but descriptive (e.g., "Match Revenue", "Advertising Revenue").

6. No Inference or Aggregation:Do not infer metrics or calculate totals. Only extract metrics as provided in the input.

7. Validation: Ensure that the output JSON structure is valid and correctly nested.

If the input text or spreadsheet includes multiple contexts, ensure each context is treated as a separate event, and metrics are grouped accordingly. """

prompt_094_metrics = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

Contextual Information: Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.
Hierarchical Organization: Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.
Explicit Descriptions: For each metric, include:
A descriptive key (metric_name) to provide clear identification of the metric (e.g., "English Premier League Broadcasting Revenue 2020/21" or "Total Revenue of the English Premier League 2011/12 to 2022/23").
The value as a numeric entry.
For monetary values, include the currency key with the ISO 3-letter code (e.g., GBP for British pounds).
Comprehensive Output: Ensure that the output includes all relevant metrics for all countries, regions, or categories mentioned in the input. If a spreadsheet or table is provided, include metrics for all rows and columns, unless explicitly excluded by the input.
No Inference or Calculation: Do not infer or calculate any metrics that are not explicitly mentioned in the text or spreadsheet data. Only extract metrics that are directly provided in the input.
Direct Extraction: Extract metrics directly from the spreadsheet data, including all rows and columns. Do not summarize or aggregate data unless explicitly stated in the input.
Output Format:

{ "event": { "event_name": "...", "event_date": "...", "metrics": { "[metric_name]": { "value": ..., // write full number if it is a thousand then write 1000. if it is million then write 1000000 and so on instead of writing 1 million or 1 billion. "currency": "...", }, ... } } }

Specifically for this prompt, please ensure that the metric names are descriptive and include the following information:
- The name of the league or competition (e.g., "English Premier League")
- The type of revenue (e.g., "Broadcasting", "Commercial", "Matchday")
- The year or range of years (e.g., "2020/21", "2011/12 to 2022/23")

For example, a metric name could be "English Premier League Broadcasting Revenue 2020/21" or "English Premier League Total Revenue 2011/12 to 2022/23".

Note: If a metric is not explicitly mentioned in the text or spreadsheet data, do not include it in the output. If a spreadsheet or table is provided, only extract metrics that are directly provided in the data, without summarizing or aggregating the data unless explicitly stated in the input.
"""

prompt_095 = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

Contextual Information: Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.
Hierarchical Organization: Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.
Explicit Descriptions: For each metric, include:
A descriptive key (metric_name) to provide clear identification of the metric (e.g., Total Revenue Premier League 2020/2021 or Total Revenue of the Big Five soccer leagues in Europe 2011/12 to 2022/23).
The value as a numeric entry.
For monetary values, include the currency key with the ISO 3-letter code (e.g., EUR for euros).
Comprehensive Output: Ensure that the output includes all relevant metrics for all countries, regions, or categories mentioned in the input. If a spreadsheet or table is provided, include metrics for all rows and columns, unless explicitly excluded by the input.
No Inference or Calculation: Do not infer or calculate any metrics that are not explicitly mentioned in the text or spreadsheet data. Only extract metrics that are directly provided in the input.
Direct Extraction: Extract metrics directly from the spreadsheet data, including all rows and columns. Do not summarize or aggregate data unless explicitly stated in the input.
Output Format:

{ "event": { "event_name": "...", "event_date": "...", "metrics": { "[metric_name]": { "value": ..., // write full number if it is a thousand then write 1000. if it is million then write 1000000 and so on instead of writing 1 million or 1 billion. "currency": "...", }, ... } } }

Note: If a metric is not explicitly mentioned in the text or spreadsheet data, do not include it in the output. If a spreadsheet or table is provided, only extract metrics that are directly provided in the data, without summarizing or aggregating the data unless explicitly stated in the input.
"""

# "For non-monetary values, include a unit key describing the unit of measurement (e.g., people or percent)."

prompt_096 = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

Contextual Information: Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.

Hierarchical Organization: Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.

Explicit Descriptions: For each metric, include:

A descriptive key (metric_name) to provide clear identification of the metric (e.g., Total Revenue Premier League 2020/2021 or Total Revenue of the Big Five soccer leagues in Europe 2011/12 to 2022/23).
The value as a numeric entry.
For monetary values, include the currency key with the ISO 3-letter code (e.g., EUR for euros).

If a person's or a country's or an organization's achievements are mentioned and it makes sense to put them as metrics then do so.
Metric Naming Convention: Use a descriptive and consistent naming convention for metrics. For example, use the format Total [Metric Name] [Category/Region] [Year/Range] (e.g., Total Revenue Premier League 2020/2021 or Total Revenue of the Big Five soccer leagues in Europe 2011/12 to 2022/23).

Comprehensive Output: Ensure that the output includes all relevant metrics for all countries, regions, or categories mentioned in the input. If a spreadsheet or table is provided, include metrics for all rows and columns, unless explicitly excluded by the input.

Exclusion Criteria: Do not infer metrics or include information that is not explicitly mentioned in the slide content or spreadsheet. If no relevant metrics are present, omit the slide in the output.

Important: If you encounter table of contents or meta information about pptx file or slide, in that case skip slide.
Note: Also skip metrics that have null value. And skip slide from output if it does not contain relevant information or expected output is going to be empty list of metrics, or there are no events mentioned.
Output Format:

{
  "event": {
    "event_name": "...",
    "event_date": "...",
    "metrics": {
      "[metric_name]": {
        "value": ..., // write full number if it is a thousand then write 1000. if it is million then write 1000000 and so on instead of writing 1 million or 1 billion.
        "currency": "...",
      },
      ...
    }
  }
}
"""

prompt_097 = """
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

Contextual Information: Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.

Hierarchical Organization: Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.

Explicit Descriptions: For each metric, include:

A descriptive key (metric_name) to provide clear identification of the metric (e.g., Total Revenue Premier League 2020/2021 or Forecast Total Revenue Premier League 2022/2023).
The value as a numeric entry.
For monetary values, include the currency key with the ISO 3-letter code (e.g., EUR for euros).
For non-monetary values, include a unit key describing the unit of measurement (e.g., people or percent).
If a person's or a country's or an organization's achievements are mentioned and it makes sense to put them as metrics then do so.
Metric Naming Convention: Use a descriptive and consistent naming convention for metrics. For example, use the format Total [Metric Name] [Category/Region] [Year/Range] (e.g., Total Revenue Premier League 2020/2021 or Total Revenue of the Big Five soccer leagues in Europe 2011/12 to 2022/23).

Exclusion Criteria: Do not infer metrics or include information that is not explicitly mentioned in the slide content or spreadsheet. If no relevant metrics are present, omit the slide in the output.

Important: If you encounter table of contents or meta information about pptx file or slide, in that case skip slide.
Note: Also skip metrics that have null value. And skip slide from output if it does not contain relevant information or expected output is going to be empty list of metrics, or there are no events mentioned.
Output Format:

{
  "event": {
    "event_name": "...",
    "event_date": "...",
    "metrics": {
      "[metric_name]": {
        "value": ...,
        "currency": "...", // if value is a currency otherwise don't mention it
      },
      ...
    }
  }
}
"""

prompt_098 = """Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

1. **Contextual Information:** Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.

2. **Hierarchical Organization:** Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.

3. **Explicit Descriptions:** For each metric, include:
   - A descriptive key (`metric_name`) to provide clear identification of the metric (e.g., `revenue_2019_20` or `forecast_revenue_2022_23`).
   - The `value` as a numeric entry.
   - For monetary values, include the `currency` key with the ISO 3-letter code (e.g., `EUR` for euros).
   - For non-monetary values, include a `unit` key describing the unit of measurement (e.g., `billion_euros`, `percent`, `people`).
   - If a person's or a country's or an organization's achievements are mentioned and it makes sense to put them as metrics then do so.

4. **Exclusion Criteria:** Do not infer metrics or include information that is not explicitly mentioned in the slide content or spreadsheet. If no relevant metrics are present, omit the slide in the output.
    - Important: If you encounter table of contents or meta information about pptx file or slide, in that case skip slide.
    - Note: Also skip metrics that have null value. And skip slide from output if it does not contain relevant information or expected output is going to be empty list of metrics, or there are no events mentioned.

5. **Output Format:** 
   ```json
   {
     "event": {
       "event_name": "...",
       "event_date": "...",
       "metrics": {
         "metric_key": {
           "metric_name": "...",
           "value": ...,
           "currency": "...",
         },
         ...
       }
     }
   }
"""

prompt_099 = """ 
Please extract all relevant metrics from the provided slide content, including economic, social, environmental, and other metrics explicitly mentioned in the text and/or spreadsheet data. Use the following guidelines:

1. **Contextual Information:** Link each metric explicitly to its broader context as described in the slide text. Ensure that metrics extracted from the spreadsheet are understood and interpreted within the slide's context.

2. **Hierarchical Organization:** Structure the output in a hierarchical JSON format that nests metrics under descriptive and meaningful keys derived from the slide content (e.g., event name, region, or category). If a metric is tied to a specific year or range of years, include this in the JSON structure.

3. **Explicit Descriptions:** For each metric, include:
   - A descriptive key (`metric_name`) to provide clear identification of the metric (e.g., `revenue_2019_20` or `forecast_revenue_2022_23`).
   - The `value` as a numeric entry.
   - For monetary values, include the `currency` key with the ISO 3-letter code (e.g., `EUR` for euros).
   - For non-monetary values, include a `unit` key describing the unit of measurement (e.g., `billion_euros`, `percent`, `people`).

4. **Supplementary Information:** If a metric is derived from the spreadsheet and does not make sense without context, add a `source_context` key to reference relevant text from the slide that provides meaning to the metric.

5. **Exclusion Criteria:** Do not infer metrics or include information that is not explicitly mentioned in the slide content or spreadsheet. If no relevant metrics are present, omit the slide in the output.

6. **Output Format:** 
   ```json
   {
     "event": {
       "event_name": "...",
       "event_date": "...",
       "metrics": {
         "metric_key": {
           "metric_name": "...",
           "value": ...,
           "currency": "...",
           "unit": "...",
           "source_context": "..."
         },
         ...
       }
     }
   }

"""

prompt_001 = """ Please extract all relevant metrics from the provided data, including economic, social, environmental, and other metrics that can impact the local, national, or global community. Please only extract metrics that are explicitly mentioned in the text, and do not generate any new information or make any assumptions.

Please output the metrics in JSON format, with each metric nested under a descriptive key. If the metric represents a monetary value, please output it as a numeric value without currency symbols or units (e.g., 67500000 instead of 67.5 million). Please include a 'currency' key with a 3-letter ISO code (e.g., 'GBP', 'USD') if the metric represents a monetary value.

Non-monetary metrics should be outputted in the same format, with a descriptive key for the metric (e.g., 'items_sold' with a numeric value) and a 'unit' key that describes the unit of measurement (e.g., 'pairs', 'people', etc.).

Please note not all slides are relevant for the purpose as some contain only table of content of description of pptx file itself. If you don't find anything relevant i.e. no metrics in a slide then skip the slide with nothing in it.

If a metric is not mentioned in the text, do not include it in the output. Please provide the output in the following format:

{
  "event": {
    "event_name": "...",
    "event_date": "...",
    "metrics": {
      "...": {
        "...": "...",
        "...": "..."
      }
    }
  }
}

Please be as accurate and precise as possible, and only include information that is explicitly mentioned in the text.
"""