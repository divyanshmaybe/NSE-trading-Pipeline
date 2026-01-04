# -*- coding: utf-8 -*-
"""
Simplified NSE Web Scraper for WhatsApp Notifications

This module scrapes NSE corporate filings announcements and processes them
for trading signals without complex infrastructure dependencies.
"""

import os
import time
import json
import requests
from datetime import datetime
from typing import List, Dict
from nse_sentiment import process_filing


# Relevant file types
RELEVANT_FILE_TYPES = {
    "Outcome of Board Meeting": {"positive": True, "negative": True},
    "Press Release": {"positive": True, "negative": False},
    "Appointment": {"positive": True, "negative": True},
    "Acquisition": {"positive": True, "negative": True},
    "Updates": {"positive": True, "negative": True},
    "Action(s) initiated or orders passed": {"positive": True, "negative": True},
    "Investor Presentation": {"positive": True, "negative": True},
    "Sale or Disposal": {"positive": True, "negative": True},
    "Bagging/Receiving of Orders/Contracts": {"positive": True, "negative": True},
    "Change in Director(s)": {"positive": True, "negative": True},
}

NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"


def get_session_headers() -> Dict[str, str]:
    """Get session headers for NSE website"""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        "Origin": "https://www.nseindia.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache",
    }


def fetch_nse_announcements() -> List[Dict]:
    """Fetch announcements from NSE API"""
    try:
        session = requests.Session()
        headers = get_session_headers()

        # Establish session
        print("[NSE] Establishing NSE session...")
        session.get(NSE_BASE_URL, headers=headers, timeout=20)
        time.sleep(1)
        session.get(f"{NSE_BASE_URL}/companies-listing/corporate-filings-announcements", headers=headers, timeout=20)
        time.sleep(1)

        # Fetch announcements
        print("[NSE] Fetching announcements from API...")
        response = session.get(NSE_API_URL, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            print(f"[WARN] Unexpected API response format")
            return []

        print(f"[NSE] Fetched {len(data)} announcements")

        # Convert to our format
        announcements = []
        for item in data:
            try:
                announcement = {
                    'symbol': item.get('symbol', ''),
                    'desc': item.get('desc', ''),
                    'dt': item.get('dt', ''),
                    'attchmntFile': item.get('attchmntFile', ''),
                    'sm_name': item.get('sm_name', ''),
                    'sm_isin': item.get('sm_isin', ''),
                    'an_dt': item.get('an_dt', ''),
                    'sort_date': item.get('sort_date', ''),
                    'seq_id': item.get('seq_id', ''),
                    'attchmntText': item.get('attchmntText', ''),
                    'fileSize': item.get('fileSize', ''),
                }
                announcements.append(announcement)
            except Exception as e:
                print(f"[WARN] Failed to process announcement: {e}")
                continue

        return announcements

    except Exception as e:
        print(f"[ERROR] Failed to fetch NSE announcements: {e}")
        return []


def is_relevant_announcement(announcement: Dict) -> bool:
    """Check if announcement is relevant for processing"""
    desc = announcement.get('desc', '').lower()
    pdf_url = announcement.get('attchmntFile', '')

    # Must have PDF attachment
    if not pdf_url or not pdf_url.strip():
        return False

    # Check if description contains relevant keywords
    relevant_keywords = [
        'board meeting', 'press release', 'appointment', 'acquisition',
        'financial results', 'quarterly results', 'annual report',
        'investor presentation', 'change in director', 'dividend',
        'bonus', 'rights issue', 'merger', 'amalgamation', 'outcome',
        'corporate action', 'insider trading', 'shareholding'
    ]

    return any(keyword in desc for keyword in relevant_keywords)


def process_announcements(announcements: List[Dict]) -> List[Dict]:
    """Process announcements and generate signals"""
    signals = []

    for ann in announcements:
        try:
            if not is_relevant_announcement(ann):
                continue

            print(f"[PROCESS] Processing {ann.get('symbol', '')}: {ann.get('desc', '')[:50]}...")

            # Generate signal using sentiment analysis
            signal = process_filing(ann)

            if signal:
                signals.append(signal)
                print(f"[SUCCESS] Generated signal for {signal['symbol']}: {signal['signal']}")
            else:
                print(f"[SKIP] No actionable signal for {ann.get('symbol', '')}")

        except Exception as e:
            print(f"[ERROR] Failed to process announcement: {e}")
            continue

    return signals


def main():
    """Main function to scrape and process NSE announcements"""
    print("[START] NSE Announcement Scraper")

    # Fetch announcements
    announcements = fetch_nse_announcements()

    if not announcements:
        print("[END] No announcements fetched")
        return

    # Filter and process relevant announcements
    relevant_announcements = [ann for ann in announcements if is_relevant_announcement(ann)]
    print(f"[INFO] Found {len(relevant_announcements)} relevant announcements out of {len(announcements)}")

    # Process and generate signals
    signals = process_announcements(relevant_announcements)

    print(f"[END] Generated {len(signals)} trading signals")

    # In a real implementation, you would send these signals to WhatsApp
    # For now, just print them
    for signal in signals:
        print(f"SIGNAL: {signal}")


if __name__ == "__main__":
    main()
