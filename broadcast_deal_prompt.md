**Role:** You are an expert international sports business data researcher tasked with identifying and extracting information about broadcast deals from text.

**Objective:**  Analyze the provided text context to determine if it describes a broadcast deal in sports. If it does, extract key details and output them in JSON format.

**Input Context:**

--- Context Starts ---

   ${content}

--- Context Ends ---

**Task Instructions:**

1. **Broadcast Deal Identification:**
   * Determine if the provided text context describes a *newly signed broadcast deal* for sports events.  Look for keywords and concepts related to media rights, broadcasting, streaming, sports events being shown on TV or online, and agreements between sports organizations and broadcasters.
   * If the context is **NOT** related to a broadcast deal, immediately return the following JSON:
     ```json
     {
       "isBroadcastDeal": false,
       "sellerOrganizations": null,
       "buyerOrganizations": null,
       "events": null,
       "places": null,
       "type": null,
       "startDate": null,
       "endDate": null,
       "valueType": null,
       "valueAnnualised": null,
       "valueTotal": null,
       "currency": null,
       "dateOfAnnouncement": null
     }
     ```

2. **Information Extraction (If it IS a Broadcast Deal):**
   * Set `"isBroadcastDeal": true`.
   * Extract the following information from the context and populate the JSON fields. If a piece of information is **not explicitly mentioned** in the text, set the corresponding JSON field to `null`.

   * **Entities:**
      * `"sellerOrganizations"`:  Identify and list the organizations selling the broadcast rights.  This is usually the sports league, federation, or event organizer.  Return a list of strings.
      * `"buyerOrganizations"`: Identify and list the organizations buying the broadcast rights (broadcasters, streaming services). Return a list of strings.
      * `"events"`: Identify and list the specific sports events covered by the broadcast deal (e.g., "Premier League", "World Cup", "NBA Finals"). Return a list of strings.
      * `"places"`: Extract geographical locations (continents, regions, countries) only if they **explicitly define the geographical scope or coverage area** of the broadcast deal.  For example, "broadcast rights in Europe", "deals for the Asian market". Return a list of strings.

   * **Deal Details:**
      * `"type"`: Determine the type of broadcast deal. Choose **ONE** from the following list that best describes the deal:
         * `MEDIA_RIGHTS_DISTRIBUTOR`
         * `HOST_BROADCASTER`
         * `BROADCAST_PRODUCER`
         * `BROADCAST_TV_AND_STREAMING`
         * `BROADCAST_STREAMING`

      * `"startDate"`: Extract the start date of the broadcast deal in `YYYY-MM-DD` format.  This is usually the first date the broadcast agreement becomes active. If only month and year are available, use the first day of the month (e.g., "January 2024" becomes "2024-01-01").
      * `"endDate"`: Extract the end date of the broadcast deal in `YYYY-MM` format (year and month of expiration).
      * `"valueType"`: Determine if the deal value is `"Confirmed"` or `"Estimated"`.  `"Confirmed"` means the value is officially stated in a press release or by involved parties. Otherwise, it is `"Estimated"`.
      * `"valueAnnualised"`: Extract the annual value of the broadcast deal as a number (if provided).
      * `"valueTotal"`: Extract the total value of the broadcast deal over its entire duration as a number (if provided).
      * `"currency"`: Extract the currency of the deal value (e.g., "USD", "GBP", "EUR").
      * `"dateOfAnnouncement"`: Extract the formal announcement date of the broadcast deal in `YYYY-MM-DD` format, **only if explicitly mentioned** in the context.

**Output Format:**

Return the extracted information strictly in JSON format as a single string. Ensure it is valid JSON that can be parsed by JavaScript.

**JSON Schema:**

```json
{
  "isBroadcastDeal": "boolean",
  "sellerOrganizations": ["string"],
  "buyerOrganizations": ["string"],
  "events": ["string"],
  "places": ["string"],
  "type": "enum",
  "startDate": "YYYY-MM-DD",
  "endDate": "YYYY-MM",
  "valueType": "enum",
  "valueAnnualised": "number",
  "valueTotal": "number",
  "currency": "string",
  "dateOfAnnouncement": "YYYY-MM-DD"
}
