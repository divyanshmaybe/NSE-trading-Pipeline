import os
import time
import json
import gzip
import requests
from datetime import datetime
from typing import List, Dict
from nse_sentiment import process_filing

RELEVANT_FILE_TYPES = {
    "Outcome of Board Meeting",
    "Press Release",
    "Appointment",
    "Acquisition",
    "Updates",
    "Action(s) initiated or orders passed",
    "Investor Presentation",
    "Sale or Disposal",
    "Bagging/Receiving of Orders/Contracts",
    "Change in Director(s)",
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
    try:
        session = requests.Session()
        headers = get_session_headers()
        session.get(NSE_BASE_URL, headers=headers, timeout=20)
        time.sleep(1)
        session.get(f"{NSE_BASE_URL}/companies-listing/corporate-filings-announcements", headers=headers, timeout=20)
        time.sleep(1)

        headers_no_accept_encoding = headers.copy()
        headers_no_accept_encoding['Accept-Encoding'] = 'identity'
        response = session.get(NSE_API_URL, headers=headers_no_accept_encoding, timeout=30)
        response.raise_for_status()

        content_encoding = response.headers.get('Content-Encoding', 'none')
        if content_encoding == 'gzip' or response.content.startswith(b'\x1f\x8b'):
            try:
                decompressed = gzip.decompress(response.content)
                json_text = decompressed.decode('utf-8')
            except Exception:
                return []
        elif content_encoding == 'br':
            try:
                try:
                    import brotli as _brotli
                    decompressed = _brotli.decompress(response.content)
                except ImportError:
                    try:
                        import brotlicffi as _brotlicffi
                        decompressed = _brotlicffi.decompress(response.content)
                    except ImportError:
                        try:
                            json_text = response.text
                        except Exception:
                            return []
                json_text = decompressed.decode('utf-8')
            except Exception:
                return []
        else:
            try:
                json_text = response.content.decode('utf-8')
            except UnicodeDecodeError:
                return []

        if not json_text.strip():
            return []

        data = json.loads(json_text)
        if not isinstance(data, list):
            return []

        from datetime import datetime, timedelta
        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)

        announcements = []
        for item in data:
            try:
                an_dt_str = item.get('an_dt', '')
                if an_dt_str:
                    try:
                        an_dt = datetime.strptime(an_dt_str, '%d-%b-%Y %H:%M:%S')
                        if an_dt < five_minutes_ago:
                            continue
                    except ValueError:
                        pass

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
            except Exception:
                continue

        return announcements

    except Exception:
        return []


def is_relevant_announcement(announcement: Dict) -> bool:
    desc = announcement.get('desc', '').strip().lower()
    pdf_url = announcement.get('attchmntFile', '')
    if not pdf_url or not pdf_url.strip():
        return False
    return desc in {k.lower() for k in RELEVANT_FILE_TYPES}


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

    # In a real implementation, you would send these signals through the configured notifier (Telegram)
    # For now, just print them
    for signal in signals:
        print(f"SIGNAL: {signal}")


if __name__ == "__main__":
    main()
