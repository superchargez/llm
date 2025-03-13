
unified_modificaition2 ="""
Your task is to analyze the attached document and identify if it discusses any of the following types of sports deals: **broadcasting deals**, **sponsorship deals**, and **hosting deals**. If deals are identified, you will also extract key information about each deal type. A document may contain multiple deals of the same type as well as a mixture of different types.

**CRITICAL RULES FOR DEAL IDENTIFICATION:**

1. **Only Include NEW/ACTIVE Deals:**
   * Only include deals that are newly signed, recently announced, or actively being negotiated
   * EXCLUDE any historical deals, expired deals, or past arrangements
   * If a document mentions a date in the past and discusses deals from that time period, those deals should be excluded

2. **Strict Deal Verification:**
   * Only include deals that are EXPLICITLY mentioned as deals, agreements, contracts, or formal arrangements
   * General information about hosting, broadcasting, or sponsorship activities should NOT be considered as deals unless specifically described as such
   * The mere mention of a sport being broadcast or a venue hosting an event is NOT sufficient to qualify as a deal

3. ** Hosting Deal Status Accuracy:**
   * For hosting deals, carefully distinguish between:
     - "Event Awarded": Only when explicitly stated that the host has been officially selected
     - "Bid Confirmed": When a formal bid has been submitted
     - "Interested Party": When a location expresses interest or is considering hosting, but hasn't formally bid

**Deal Type Definitions and Identification Guidance:**

* **Broadcasting Deal:**
    * MUST involve a formal agreement between specific parties for media rights
    * MUST have economic value (free-to-air broadcasts do NOT qualify as deal)
    * Keywords and phrases that indicate a valid deal:
      - "signed a broadcasting agreement"
      - "awarded media rights"
      - "secured broadcasting rights"
      - "reached a deal for television coverage"

* **Sponsorship Deal:**
    * MUST involve specific financial or in-kind support
    * MUST be a formal agreement between named parties
    * Keywords and phrases that indicate a valid deal:
      - "signed as official sponsor"
      - "announced partnership worth"
      - "agreed sponsorship terms"
      - "became official partner"

* **Hosting Deal:**
    * MUST involve specific events and locations
    * MUST be about future or current (as described/mentioned in document) events (not past ones)
    * Status MUST be carefully determined based on explicit language:
      - "awarded the rights to host" → Event Awarded
      - "submitted official bid" → Bid Confirmed
      - "expressing interest" or "considering hosting" → Interested Party

**Additional Validation Rules:**

1. Multiple Deals Check:
   * Scan the entire document thoroughly for all deals
   * Each deal should be evaluated independently
   * Multiple deals of the same type should be included separately in the appropriate section

2. Value Verification:
   * Only include monetary values that are explicitly associated with the deal
   * For broadcasting and sponsorship deals, there MUST be a financial component
   * If multiple values are mentioned, use the most recent or explicitly stated figure

3. Date Verification:
   * Always verify if dates mentioned are for new/upcoming deals
   * If a date is mentioned in past tense or as historical reference, the deal should be excluded
   * For deals without specific days, use the first day of the mentioned month

Example of Correct Classification:
Text: "Germany is looking forward to host new Bundesliga which is expected to make 100M Euros more than it previously did for Germany. It has been reported in news on 7 September 2007 that Germany is probably going to host the coming football league."
Correct Classification: This should be classified as a hosting deal with status "Interested Party" because:
- Germany is expressing interest but hasn't formally bid
- No official award or bid has been confirmed
- The mention of revenue is a projection, not a deal value

Accurate JSON is Mandatory: Ensure your response is always valid JSON and follows the specified structure precisely.

Example JSON Output Scenarios:
1. No deal found:
{"is_deal": false}

2. Sponsorship deal only:
{
  "is_deal": true,
  "deal_types": ["sponsorship"],
  "broadcastingDealDetails": null,
  "sponsorshipDealDetails": {
    "isSponsorshipDeal": true,
    "sellerOrganizations": ["Organization A"],
    "buyerOrganizations": ["Company B"],
    "teams": ["Team X"],
    "events": [],
    "athletes": [],
    "venues": [],
    "places": [],
    "type": "SHIRT_SPONSOR",
    "startDate": "2024-01-01",
    "endDate": "2025-12-01",
    "valueType": "Confirmed",
    "valueAnnualised": 1000000,
    "valueTotal": 2000000,
    "currency": "USD",
    "dateOfAnnouncement": "2023-12-15"
  },
  "hostingDealDetails": null
}

3. Broadcasting and Hosting deals:
{
  "is_deal": true,
  "deal_types": ["broadcasting", "hosting"],
  "broadcastingDealDetails": {
    "isBroadcastDeal": true,
    "sellerOrganizations": ["League Z"],
    "buyerOrganizations": ["Network Y"],
    "events": ["Championship Series"],
    "places": ["Europe"],
    "type": "BROADCAST_TV_AND_STREAMING",
    "startDate": "2025-07-01",
    "endDate": "2028-06-01",
    "valueType": "Estimated",
    "valueAnnualised": 5000000,
    "valueTotal": 15000000,
    "currency": "EUR",
    "dateOfAnnouncement": "2024-05-20"
  },
  "sponsorshipDealDetails": null,
  "hostingDealDetails": {
    "isHostingDeal": true,
    "dateOfAnnouncement": "2024-03-10",
    "places": ["City W", "Country V"],
    "events": ["Global Games 2030"],
    "status": "Event Awarded"
  }
}

4. Broadcasting, Sponsorship, and Hosting deals:
{
  "is_deal": true,
  "deal_types": ["broadcasting", "sponsorship", "hosting"],
  "broadcastingDealDetails": { /* ... broadcasting details ... */ },
  "sponsorshipDealDetails": { /* ... sponsorship details ... */ },
  "hostingDealDetails": { /* ... hosting details ... */ }
}
"""

step_1_prompt = """

"""