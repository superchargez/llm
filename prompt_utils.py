def prompt_gen(prompt:str=None) -> str:
    prompt = improved_prompt
    return prompt

md_prompt_best4summary = """
### Primary Directive
Extract sporting event metrics while maintaining accurate event attribution. Focus on quantifiable data and metrics directly related to events.

### Event Attribution Rules
1. EXPLICIT EVENT: If page contains clear event name (e.g., "FIFA World Cup 2026")
   - Use this as primary event
   - Override previous context
2. CONTEXTUAL EVENT: If page contains metrics but no clear event name
   - Check if metrics align with previous event (within 3 pages)
   - Look for transitional phrases ("In the same tournament", "During this championship")
   - Maintain previous event if strong correlation exists
3. NEW EVENT DETECTION:
   - Look for date changes, location changes, or participant changes
   - Compare with previous summaries for continuity breaks
   - Mark clear transitions: "New Event Detected: [Reason for transition]"

### Context Maintenance
- Track last explicit event mention (page number and name)
- Maintain rolling 3-page context window
- Score context confidence:
   HIGH: Direct event mention or clear continuation
   MEDIUM: Implicit continuation with matching participants/venue
   LOW: Only temporal/theme alignment

### Event Resolution (Priority Order)
1. EXPLICIT EVENT: Extract proper event name if present (e.g., "FIFA World Cup 2026")
2. CONTEXTUAL EVENT: Use event from provided context if metrics match
3. FILE EVENT: Use filename-derived event if appropriate
4. MULTI-EVENT: Handle multiple events with clear metric associations

### Data Extraction Focus
Extract ALL quantifiable data including:
- Performance: scores, rankings, times, distances
- Participation: participant counts, teams, countries
- Logistics: dates, locations, attendance
- Organization: owners, governing bodies
- Categories: sport types, divisions

### Required Output Structure

### Event Attribution
- Primary Event: [Name]
- Related Events: [If multiple present]

### [Event Name] Metrics
[All metrics in table format where possible]

### Summary
#### Current Page
- Event(s): [Names]
- Key Metrics: [Brief summary]
- Context Changes: [Event focus changes]

#### Previous Pages Summary
[Maintained from context]


### Data Presentation
1. Use Markdown tables for structured data
2. Maintain visual hierarchy with headings
3. Include blank lines between sections
4. Format complex data clearly:

   ### [Event] Statistics
   | Metric | Value |
   |--------|--------|
   | Data   | Value |

### Context Requirements
- Always include: `Context: {previous_summary}`
- Event transitions must be explicit
- Mark uncertain associations: `[Uncertain Event] | [Metric]`
- Maintain previous event context until clear switch

### Exclusions
- Descriptive text without metrics
- Non-event related data
- Generic content without clear event association

### No Relevant Data Response
If no event-related metrics found: "No relevant event metrics found on this page. Previous context maintained: [Event Name]"

### IMPORTANT NOTE:
- Output response should be markdown formatted.
"""

md_prompt_02 = """
Extract factual information directly presented in this image that is explicitly related to sporting events, focusing specifically on quantifiable metrics and data *directly related to those events* (e.g., scores, times, attendance, participating teams, not general news articles about sports). Prioritize extracting quantifiable data (numbers, dates, etc.). If space is limited, favor quantifiable data over descriptive text or other non-numerical information. Focus solely on what is directly visible. Do not infer, interpret, or add any external context. Maintain the visual hierarchy and grouping of information as presented in the image. Exclude any information that is not directly related to the sporting event's metrics or data, such as source notes, image credits, generic placeholders, table of contents, purely descriptive text without data, images without associated data, generic section titles that are not proper names of sporting events, reports, evaluations, critiques, glossaries, abbreviations, or similar documents that do not contain event-specific metrics.

**Contextual Information (Provided Separately):** You will be provided with a summary of previously processed pages. This summary may contain the name of the main sporting event and other relevant information extracted from those pages. Use this context to help identify the event associated with the data on the current page, especially if the event name is missing or unclear on the current page. The context will be provided in the following format: `Context: {summary of previous pages}`.

Specifically:

### Event Identification:

*   If the image contains the *proper name* of a sporting event (e.g., "FIFA World Cup 2026," "Super Bowl LVII," "Wimbledon Championships 2023"), extract it. A proper name is the official, unique title of a specific event (e.g., 'Boston Marathon 2024'), not a generic term like 'Marathon Results' or a document title like 'Event Report'.
*   If metrics *directly related to the event* are present on the same page, include the event name as a top-level heading: `### Event | [Event Name]`.
*   If no *proper* event name is present on the current page, but the context contains an event name, use the event name from the context as the top-level heading: `### Event | [Event Name from Context]`.
*   If neither a *proper* event name is present on the current page nor in the provided context, but metrics *clearly tied to a sporting event* are, use a generic heading like `### Sporting Event Metrics`.
*   If only the proper event name is present without associated metrics, output only `### Event | [Event Name]` and stop processing the page.

### Data Extraction Focus:

*   Extract *all* quantifiable data and metrics directly related to the sporting event. This includes, but is not limited to:
    *   **Performance Metrics:** Scores, rankings, times, distances, wins/losses, etc.
    *   **Participation Metrics:** Number of participants, teams, countries, etc.
    *   **Logistical Metrics:** Dates, locations, attendance, revenue, budget, etc.
    *   **Organizational Metrics:** Owner, organizer, governing body, frequency (e.g., annual, bi-annual), etc.
    *   **Categorical Metrics:** Sport type (e.g., football, basketball, skiing), event type (e.g., championship, tournament, race), etc.
*   If data is presented in a way that implies a metric (e.g., "Organized by: [Name of Organization]"), extract it even if it's not a numerical value. If descriptive text is essential for understanding a metric (e.g., 'Average Speed: 150 mph'), include the concise description along with the metric.

### Titles and Headings:

*   Extract all titles, headings, and subheadings that are directly related to the sporting event's metrics.
*   Represent hierarchy using Markdown headings (`#`, `##`, `###`, etc.).
*   **Insert a blank line between sections or headings** to improve readability. Avoid merging headings into a continuous block without spacing.

### Tables and Data:

*   For tabular data, **add a blank line before and after each table** to ensure clear separation from other sections. Use Markdown headings above tables to indicate their context. For tables with complex structures (merged cells, multi-level headers), represent the structure as accurately as possible using Markdown. If a perfect Markdown representation is not feasible, prioritize clear data extraction over perfect formatting.
*   Output example for a table:
### Statistics

#### Economic & Tourism
|Column 1| Column 2|
|Row 1 Data| Value|


### Graphical Elements:

*   If the image contains charts, graphs, maps, or other graphical elements with associated labels or data points directly related to a sporting event, extract the data and represent it in a Markdown table with appropriate headings.
*   Precede any graphical data representation with a heading that reflects the content's purpose.

### Logos and Images:

*   Only if a logo or image is directly tied to a specific metric or data point of a sporting event, briefly describe it using square brackets, e.g., [Olympic Rings logo associated with revenue figures]. Exclude generic logos, watermarks, irrelevant visuals, or images without associated data.

### Maintain Visual Grouping:

*   Use headings and subheadings to reflect the logical grouping of information as visually presented in the image. Avoid inline formatting like bold text (**) for section names, using Markdown headings instead.

### Output Format:

*   Please try to put all data in tabular format if possible. If not possible then use lists or paragraphs to represent the data but only in markdown language not in HTML style.

### Conciseness:

*   Present the extracted information concisely in Markdown format. Minimize unnecessary spaces and words to optimize rendering and readability. However, don't sacrifice readability for spacing, i.e., don't put two headings together; keep them separate so that parsing markdown to HTML is possible and a nice output is displayed.

### Exclusion Criteria:

*   Exclude table of contents, purely descriptive text without quantifiable data, images of people or events without associated metrics, any other non-data-driven content, generic section titles or phrases that are not proper names of sporting events, reports, evaluations, critiques, glossaries, abbreviations, or similar documents that do not contain event-specific metrics.

### If No Relevant Information:

*   If no quantifiable information directly related to a sporting event is visible, output: No relevant information found.

### Handling Uncertainties:

*   If the data is unclear or ambiguous in the image, use a question mark or a brief note in parentheses (e.g., 'Score: 24 (?)' or 'Date: June (possibly)').

### Summary Generation:

*   After extracting the information from the current page, create a concise summary of the key information, including the event name (if present), key metrics, and any other relevant details. This summary should be suitable for inclusion in the context provided for subsequent pages. The summary should be in the format: `Event: [Event Name], Key Metrics: [List of key metrics], Other Details: [Other relevant information]`. If no relevant information is found on the page, the summary should be: `No relevant information found on this page.`

### Important Note:
*   You must output response in markdown style, rather than simple text format, so that extracting data is easier.
"""

md_prompt = """
Extract factual information directly presented in this image that is explicitly related to sporting events, focusing specifically on quantifiable metrics and data *directly related to those events*. Focus solely on what is directly visible. Do not infer, interpret, or add any external context. Maintain the visual hierarchy and grouping of information as presented in the image. Exclude any information that is not directly related to the sporting event's metrics or data, such as source notes, image credits, generic placeholders like "Your Headline" or "Your Notes", table of contents, purely descriptive text without data, images without associated data, generic section titles that are not event names (e.g., "Event Overview," "Introduction," "Summary"), *reports, evaluations, critiques, glossaries, abbreviations, or similar documents or lists that do not contain event-specific metrics*.

Specifically:

### Event Identification:
- If the image contains the *proper name* of a sporting event (e.g., "FIFA World Cup 2026," "Super Bowl LVII," "Wimbledon Championships 2023"), extract it. A proper name should be a specific, recognizable title of a sporting event, not a generic description or document title. If metrics *directly related to the event* are present on the same page, include the event name as a top-level heading: `### Event | [Event Name]`. If no such metrics are present, but the *proper* event name is, output only `### Event | [Event Name]` and stop processing the page.
- If no *proper* event name is present, but metrics *clearly tied to a sporting event* are, use a generic heading like `### Sporting Event Metrics`.

### Data Extraction Focus:
- Extract *all* quantifiable data and metrics directly related to the sporting event. This includes, but is not limited to:
    - **Performance Metrics:** Scores, rankings, times, distances, wins/losses, etc.
    - **Participation Metrics:** Number of participants, teams, countries, etc.
    - **Logistical Metrics:** Dates, locations, attendance, revenue, budget, etc.
    - **Organizational Metrics:** Owner, organizer, governing body, frequency (e.g., annual, bi-annual), etc.
    - **Categorical Metrics:** Sport type (e.g., football, basketball, skiing), event type (e.g., championship, tournament, race), etc.
- If data is presented in a way that implies a metric (e.g., "Organized by: [Name of Organization]"), extract it even if it's not a numerical value.

### Titles and Headings:
- Extract all titles, headings, and subheadings that are directly related to the sporting event's metrics.
- Represent hierarchy using Markdown headings (`#`, `##`, `###`, etc.).
- **Insert a blank line between sections or headings** to improve readability.
- Avoid merging headings into a continuous block without spacing.

### Tables and Data:
- For tabular data, **add a blank line before and after each table** to ensure clear separation from other sections.
- Use Markdown headings above tables to indicate their context. For example:

### Statistics

#### Economic & Tourism
|Column 1| Column 2|
|Row 1 Data| Value|

- For data presented in non-tabular formats (e.g., lists, paragraphs), reproduce it faithfully using Markdown lists or paragraphs, ensuring clarity.

### Graphical Elements:
 - If the image contains charts, graphs, maps, or other graphical elements with associated labels or data points directly related to a sporting event, extract the data and represent it in a Markdown table with appropriate headings.
 - Precede any graphical data representation with a heading that reflects the content's purpose.

### Logos and Images:
 - Only if a logo or image is directly tied to a specific metric or data point of a sporting event, briefly describe it using square brackets, e.g., [Olympic Rings logo associated with revenue figures].
 - Exclude generic logos, watermarks, irrelevant visuals, or images without associated data.

### Maintain Visual Grouping:
 - Use headings and subheadings to reflect the logical grouping of information as visually presented in the image.
 - Avoid inline formatting like bold text (**) for section names, using Markdown headings instead.

### Output Format:
 - Please try to put all data in tabular format if possible. If not possible then use lists or paragraphs to represent the data but only in markdown language not in HTML style.

### Conciseness:
 - Present the extracted information concisely in Markdown format. Minimize unnecessary spaces and words to optimize rendering and readability. However, don't sacrifice readability for spacing, i.e. don't put two headings together keep them separate so that parsing markdown to html is possible and a nice output is displayed.

### Exclusion Criteria:
 - Exclude table of contents, purely descriptive text without quantifiable data, images of people or events without associated metrics, any other non-data-driven content, generic section titles or phrases that are not proper names of sporting events (e.g., "Event Overview," "Introduction," "Summary," "Background," "Key Findings," "Methodology"), reports, evaluations, critiques, glossaries, abbreviations, or similar documents or lists that do not contain event-specific metrics.

### If No Relevant Information or Only Event Name:
 - If no quantifiable information directly related to a sporting event is visible, or only the proper event name is present without associated metrics, output: ### Event | [Event Name] (if a proper event name exists) or No relevant information found. (if no proper event name exists). Do not process the page further.
 - If something is part of background of image i.e. not a part of main content or lacks focus then exclude it from the output.

### Important Note:
 - If something can be considered a metric or data point, even if it is not a numerical value, please include it in the output. For example, "Host City: [City Name]" or "Winner: [Team Name]" or Competing Nations/Teams and if there is geographical or other demographic distinghing element present in image that should be included as they are directly related to the event.

"""

# improved_prompt = """ # best yet
# Extract factual information directly presented in this image that is explicitly related to sporting events, prefer quantifiable metrics over quantifiable and data *directly related to those events*. Focus solely on what is directly visible. Do not infer, interpret, or add any external context. Maintain the visual hierarchy and grouping of information as presented in the image. Exclude any information that is not directly related to the sporting event's metrics or data, such as source notes, image credits, generic placeholders like "Your Headline" or "Your Notes", table of contents, purely descriptive text without data, images without associated data, generic section titles that are not event names (e.g., "Event Overview," "Introduction," "Summary"), *reports, evaluations, critiques, glossaries, abbreviations, or similar documents or lists that do not contain event-specific metrics*.

# Specifically:

# ### Event Identification:
# - If the image contains the *proper name* of a sporting event (e.g., "FIFA World Cup 2026," "Super Bowl LVII," "Wimbledon Championships 2023"), extract it. A proper name should be a specific, recognizable title of a sporting event, not a generic description or document title. If metrics *directly related to the event* are present on the same page, include the event name as a top-level heading: `### Event | [Event Name]`. If no such metrics are present, but the *proper* event name is, output only `### Event | [Event Name]` and stop processing the page.
# - If no *proper* event name is present, but metrics *clearly tied to a sporting event* are, use a generic heading like `### Sporting Event Metrics`.

# ### Event Name and Information:
# - If any date is mentioned in page, then think carefully about it whether it can be part of event name or not, sometimes it is for example Evaluation report of FIFA World Cuip and on page it is mentioned that visitors arrived on 1 june 2022 then it is most likely FIFA 2022 World Cup.
# - Event specefic meta information, like organizer; frequency; location and sport/s,  are also metrics that will be used later, so please extract them as well.
# - Please extract all relevant information of the event as they maybe metrics or data points, for example if it is mentioned that event is held in Paris then it is a metric, Location: Paris.

# ### Titles and Headings:
# - Extract all titles, headings, and subheadings that are directly related to the sporting event's metrics.
# - Represent hierarchy using Markdown headings (`#`, `##`, `###`, etc.).
# - **Insert a blank line between sections or headings** to improve readability.
# - Avoid merging headings into a continuous block without spacing.

# ### Tables and Data:
# - For tabular data, **add a blank line before and after each table** to ensure clear separation from other sections.
# - Use Markdown headings above tables to indicate their context. For example:

# ### Statistics

# #### Economic & Tourism
# |Column 1| Column 2|
# |Row 1 Data| Value|

#  - For data presented in non-tabular formats (e.g., lists, paragraphs), reproduce it faithfully using Markdown lists or paragraphs, ensuring clarity.

# ### Graphical Elements:
#  - If the image contains charts, graphs, maps, or other graphical elements with associated labels or data points directly related to a sporting event, extract the data and represent it in a Markdown table with appropriate headings.
#  - Precede any graphical data representation with a heading that reflects the content's purpose.

# ### Logos and Images:
#  - Only if a logo or image is directly tied to a specific metric or data point of a sporting event, briefly describe it using square brackets, e.g., [Olympic Rings logo associated with revenue figures].
#  - Exclude generic logos, watermarks, irrelevant visuals, or images without associated data.

# ### Maintain Visual Grouping:
#  - Use headings and subheadings to reflect the logical grouping of information as visually presented in the image.
#  - Avoid inline formatting like bold text (**) for section names, using Markdown headings instead.

# ### Conciseness:
#  - Present the extracted information concisely in Markdown format. Minimize unnecessary spaces and words to optimize rendering and readability. However, don't sacrifice readability for spacing, i.e. don't put two headings together keep them separate so that parsing markdown to html is possible and a nice output is displayed.

# ### Specific Inclusions (Focus on Key Metrics):
#  - GSI Ranking/Event Rating/Similar Rankings: Extract rankings, scores, points, or any other quantifiable metrics associated with event performance or standing.
#  - Competing Nations/Participants: Extract the number or list of competing nations, athletes, or participants. Include breakdowns by gender (Men/Women) if available.
#  - Past/Future Events: Extract dates, locations, or other identifying information for past or future iterations of the event.

# ### Exclusion Criteria:
#  - Exclude table of contents, purely descriptive text without quantifiable data, images of people or events without associated metrics, any other non-data-driven content, generic section titles or phrases that are not proper names of sporting events (e.g., "Event Overview," "Introduction," "Summary," "Background," "Key Findings," "Methodology"), reports, evaluations, critiques, glossaries, abbreviations, or similar documents or lists that do not contain event-specific metrics.

# ### If No Relevant Information or Only Event Name:
#  - If no quantifiable or qualitative information directly related to a sporting event is visible, or only the proper event name is present without associated metrics, output: ### Event | [Event Name] (if a proper event name exists) or No relevant information found. (if no proper event name exists). Do not process the page further.
#  - If something is part of background of image i.e. not a part of main content or lacks focus then exclude it from the output.

#  """

# improved_prompt_01 = """ # sometimes missed bed nights as a heading and put that as a metric
# Extract factual information directly presented in this image that is explicitly related to sporting events, prioritizing quantifiable metrics. Focus solely on what is directly visible. Do not infer, interpret, or add external context. Maintain the visual hierarchy and grouping of information. Exclude information not directly related to the sporting event's metrics or data, such as source notes, image credits, generic placeholders, purely descriptive text without data, images without associated data, generic section titles (e.g., "Overview," "Introduction"), reports, evaluations, critiques, glossaries, abbreviations, or similar documents without event-specific metrics.

# Specifically:

# ### Event Identification:
# - Extract the *proper name* of any sporting event (e.g., "FIFA World Cup 2026"). A proper name is a specific, recognizable title, not a generic description.
# - If metrics *directly related to the event* are present, use the event name as a top-level heading: `### Event | [Event Name]`.
# - If only the *proper* event name is present without metrics, output only `### Event | [Event Name]` and stop processing.
# - If no *proper* event name is present but metrics *clearly tied to a sporting event* exist, use `### Sporting Event Metrics`.

# ### Event Meta Information and Metrics:
# - Extract event-specific meta information: organizer, frequency, location, dates, and sport(s). Consider dates within the context of event names (e.g., "FIFA 2022 World Cup"). Treat these as key metrics.
# - Extract all other relevant event information as metrics or data points (e.g., "Location: Paris").

# ### Titles and Headings:
# - Extract all titles, headings, and subheadings directly related to the sporting event's metrics, maintaining hierarchy using Markdown headings (`#`, `##`, `###`, etc.).
# - **Insert a blank line between sections/headings.**

# ### Tables and Data:
# - **Add a blank line before and after each table.**
# - Use Markdown headings above tables for context. Example:

# ### Statistics

# #### Economic & Tourism
# |Column 1| Column 2|
# |---|---|
# |Row 1 Data| Value|

# - Reproduce non-tabular data (lists, paragraphs) using Markdown lists or paragraphs.

# ### Graphical Elements:
# - Extract data from charts, graphs, maps, etc., and represent it in Markdown tables with headings.
# - Precede graphical data with a descriptive heading.

# ### Logos and Images:
# - Briefly describe logos/images *only* if directly tied to a specific metric (e.g., [Olympic Rings logo associated with revenue figures]).
# - Exclude generic logos, watermarks, irrelevant visuals, or images without associated data.

# ### Maintain Visual Grouping:
# - Use headings/subheadings to reflect the visual grouping of information.
# - Avoid inline formatting like bold text for section names; use Markdown headings.

# ### Conciseness:
# - Present information concisely in Markdown. Maintain readability.

# ### Specific Inclusions (Key Metrics):
# - Rankings (GSI Ranking, Event Rating, etc.): Extract rankings, scores, points, etc.
# - Participants: Extract the number/list of competing nations, athletes, etc., including gender breakdowns.
# - Past/Future Events: Extract dates, locations, etc., of past/future iterations.

# ### Exclusion Criteria:
# - Exclude table of contents, purely descriptive text without data, images of people/events without metrics, non-data-driven content, generic section titles not proper event names, reports, evaluations, critiques, glossaries, abbreviations, or similar documents without event-specific metrics. Exclude background information not part of the main content.

# ### If No Relevant Information or Only Event Name:
# - If no quantifiable/qualitative information directly related to a sporting event is visible, or only the proper event name is present, output: `### Event | [Event Name]` (if a proper name exists) or `No relevant information found.` (if no proper name exists) and stop processing.
# """

improved_prompt = """
Extract factual information directly presented in this image that is explicitly related to sporting events, prioritizing quantifiable metrics. Focus solely on what is directly visible. Do not infer, interpret, or add external context. Maintain the visual hierarchy and grouping of information. Exclude information not directly related to the sporting event's metrics or data, such as source notes, image credits, generic placeholders, purely descriptive text without data, images without associated data, generic section titles (e.g., "Overview," "Introduction"), reports, evaluations, critiques, glossaries, abbreviations, or similar documents without event-specific metrics.

Specifically:

### Event Identification:
- Extract the *proper name* of any sporting event (e.g., "FIFA World Cup 2026"). A proper name is a specific, recognizable title, not a generic description.
- If metrics *directly related to the event* are present, use the event name as a top-level heading: `### Event | [Event Name]`.
- If only the *proper* event name is present without metrics, output only `### Event | [Event Name]` and stop processing.
- If no *proper* event name is present but metrics *clearly tied to a sporting event* exist, use `### Sporting Event Metrics`.

### Event Meta Information and Metrics:
- Extract event-specific meta information: organizer, frequency, location, dates, and sport(s). Consider dates within the context of event names (e.g., "FIFA 2022 World Cup"). Treat these as key metrics.
- Extract all other relevant event information as metrics or data points (e.g., "Location: Paris").

### Titles and Headings:
- Extract all titles, headings, and subheadings directly related to the sporting event's metrics, maintaining hierarchy using Markdown headings (`#`, `##`, `###`, etc.).
- **Insert a blank line between sections/headings.**

### Tables and Data:
- **Add a blank line before and after each table.**
- Use Markdown headings above tables for context. Example:

### Statistics

#### Economic & Tourism
|Column 1| Column 2|
|---|---|
|Row 1 Data| Value|

- Reproduce non-tabular data (lists, paragraphs) using Markdown lists or paragraphs.

### Graphical Elements:
- Extract data from charts, graphs, maps, etc., and represent it in Markdown tables with headings.
- Precede graphical data with a descriptive heading.

### Logos and Images:
- Briefly describe logos/images *only* if directly tied to a specific metric (e.g., [Olympic Rings logo associated with revenue figures]).
- Exclude generic logos, watermarks, irrelevant visuals, or images without associated data.

### Maintain Visual Grouping:
- Use headings/subheadings to reflect the visual grouping of information.
- Avoid inline formatting like bold text for section names; use Markdown headings.

### Conciseness:
- Present information concisely in Markdown. Maintain readability.

### Specific Inclusions (Key Metrics):
- Rankings (GSI Ranking, Event Rating, etc.): Extract rankings, scores, points, etc.
- Participants: Extract the number/list of competing nations, athletes, etc., including gender breakdowns.
- Past/Future Events: Extract dates, locations, etc., of past/future iterations.

### Exclusion Criteria:
- Exclude table of contents, purely descriptive text without data, images of people/events without metrics, non-data-driven content, generic section titles not proper event names, reports, evaluations, critiques, glossaries, abbreviations, or similar documents without event-specific metrics. Exclude background information not part of the main content.

### If No Relevant Information or Only Event Name:
- If no quantifiable/qualitative information directly related to a sporting event is visible, or only the proper event name is present, output: `### Event | [Event Name]` (if a proper name exists) or `No relevant information found.` (if no proper name exists) and stop processing.

### Please NOTE:
 - If something appears to be different, as it maybe has different font size, highlighting in image or is underlined i.e. somehow highlighted or focused as compared to other metircs under the same table then, and does not have a value associated with it then think about it whether it is better to put it as a seprate heading and information below it as a seprate table if so then make such value a name of table instead of using it as a metric.
 - For example "Bed Nights By Visitors" is usually highlighted by putting an underline under it. For example:

| Bed nights by visitor type |  |
| Spectators | - |
| Athletes | 2,700* |
| Team Officials | 1,300* |

And since Spectators or Athletes on their own don't make sense as a metric so Bed nights by visitor type should be a seprate heading (however it is very important to first look at image and see if such a section is in anyway highlighted or not). If it is decided that it makes sense to make it a separate heading then output should be something like:

##### Bed nights by visitor type (put another # with it to make it understandable that it is actually part of above larger section/table)
| Visitor Type | Value |
|---|---|
| Spectators | 2,700* |
| Athletes | 1,300* |
"""

# json_prompt = """Please extract the following information in JSON format:
#     {
#         "event_name": "<event name with year>",
#         "event_date": "<event date or year>",
#         "metrics": {
#             "<metric_name>": {
#                 "value": <numeric_value>,
#                 "currency": "<currency_code or null>"

#             }
#         }
#     }"""

# json_prompt = """Extract event information and metrics from the content below. Return in the following JSON format:
#     {
#         "page_number": <page_number>,
#         "events": [
#             {
#                 "name": "<event name>",
#                 "date": "<event date>",
#                 "metrics": [
#                     {
#                         "name": "<metric_name>",
#                         "value": <numeric_value>,
#                         "currency": "<currency_code or null>"

#                     }
#                 ]
#             }
#         ]
#     }

#     For non-currency values, use null for currency. Format numbers as plain numbers without grouping.
#     If you find multiple events on the same page, include them as separate objects in the events array.
#     """

# json_prompt = '''Extract event information and metrics from the content below. Return in the following JSON format:
# {
#     "page_number": <page_number>,
#     "events": [
#         {
#             "name": "<event name>",
#             "date": "<event date>",
#             "metrics": [
#                 {
#                     "name": "<metric_name>",
#                     "value": <numeric_value>,
#                     "currency": "<currency_code or null>"
#                 }
#             ]
#         }
#     ]
# }

# Guidelines for extraction:

# 1. Date formatting:
#    - Convert all dates to ISO format YYYY-MM-DD
#    - For date ranges, use the start date
#    - If year is missing in event name but can be determined from date or content, append it to event name

# 2. Numeric values:
#    - Convert all values to their full number without abbreviations
#    - K = multiply by 1,000 (e.g., 50K → 50000)
#    - M = multiply by 1,000,000 (e.g., $40M → 40000000000)
#    - B = multiply by 1,000,000,000
#    - Remove currency symbols and commas before conversion
#    - Output numbers as plain integers without underscores or separators

# 3. Currency:
#    - Use standard 3-letter currency codes (e.g., USD, EUR)
#    - If currency symbol present but no explicit code, use most likely currency code
#    - Set to null if no currency indicated

# 4. Multiple events:
#    - Create separate objects in events array
#    - Maintain all metrics associated with respective event

# Extract all relevant information while preserving the exact structure of the JSON schema.
# '''

json_prompt = """
Extract event information and metrics from the content below. Return in the following JSON format:
{
    "events": [
        {
            "name": "<event name>",
            "date": "<event date>",
            "metrics": [
                {
                    "name": "<metric_name>",
                    "value": <numeric_value>,
                    "currency": "<currency_code or null>"
                }
            ]
        }
    ]
}

Guidelines for extraction:

1. Date formatting:
    - Convert all dates to ISO format YYYY-MM-DD
    - For date ranges, use the start date
    - If year is missing in event name but can be determined from date or content, append it to event name

2. Numeric values:
    - Convert all values to their full number without abbreviations
    - K = multiply by 1,000 (e.g., 50K → 50000)
    - M = multiply by 1,000,000 (e.g., $40M → 40000000)
    - B = multiply by 1,000,000,000
    - Remove currency symbols and commas before conversion
    - Output numbers as plain integers without underscores or separators

3. Currency:
    - Use standard 3-letter currency codes (e.g., USD, EUR)
    - If currency symbol present but no explicit code, use most likely currency code
    - Set to null if no currency indicated

4. Multiple events:
    - Create separate objects in events array
    - Maintain all metrics associated with respective event

5. Handling tables and lists with multiple values:
    - Extract data from tables and lists as individual metrics.
    - If a table cell contains multiple values separated by "/", treat each value as a separate metric.
    - **Metric Naming Logic (Priority Order):**
        1.  **Local Context (Same Table):** If other metrics within the *same table* provide a clear context (e.g., "Athletes - Total"), use that context to create more descriptive names. For example, "Men / Women | 168 / 77" under a table containing "Athletes - Total" should result in "Men Athletes" and "Women Athletes".
        2.  **Table/Section Heading:** If the local context within the table is insufficient, use the table or section heading to create metric names. For example, if there was no "Athletes - Total" metric, "Men / Women | 168 / 77" under the "Sporting" heading the resulting metrics would be "Sporting Men" and "Sporting Women".
        3. Simple Splitting: If neither of the above apply use heading name and metric mentioned under it and split it in two/three (however many are they) so that each metric is mentioned separately.
    - For example, a table under "Competing Nations By Continent" should create metrics like "Competing Nations North America", "Competing Nations Europe", etc.

6. Missing Values:
    - If a metric has no numeric value (e.g., "-"), do not include it in the output.

7. Ensure all metrics are extracted:
    - Extract all metrics present in the input, including rankings, ratings, and breakdowns. Do not omit any data.

Extract all relevant information while preserving the exact structure of the JSON schema.
"""

if __name__ == "__main__":
    print(prompt_gen(md_prompt))
    print(prompt_gen(json_prompt))