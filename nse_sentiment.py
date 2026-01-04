# -*- coding: utf-8 -*-
"""
Simplified NSE Filings Sentiment Agent for WhatsApp Notifications

This script processes NSE corporate filings, extracts text from PDFs,
fetches stock data, and generates trading signals using LLM analysis.
Sends actionable signals via WhatsApp notifications.
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader
import pandas as pd
import base64

# Load environment variables
load_dotenv()

# Configuration
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

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Validate API key
if not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY not found in environment variables!")
    print("[WARN] Please set it in .env file or export GEMINI_API_KEY=your_key")
else:
    print(f"[INFO] GEMINI_API_KEY loaded (length: {len(GEMINI_API_KEY)})")


def map_filing_type(desc: str) -> str:
    """Map announcement description to a relevant filing type"""
    if not desc:
        return ""

    desc_lower = desc.lower()

    # Direct keyword matching
    for filing_type in RELEVANT_FILE_TYPES.keys():
        if filing_type.lower() in desc_lower:
            return filing_type

    # Fuzzy matching
    if "board" in desc_lower and "meeting" in desc_lower:
        return "Outcome of Board Meeting"
    elif "press" in desc_lower or "release" in desc_lower:
        return "Press Release"
    elif "appoint" in desc_lower or "resignation" in desc_lower:
        return "Appointment"
    elif "acqui" in desc_lower or "merger" in desc_lower:
        return "Acquisition"
    elif "update" in desc_lower:
        return "Updates"
    elif "order" in desc_lower or "action" in desc_lower:
        return "Action(s) initiated or orders passed"
    elif "presentation" in desc_lower or "investor" in desc_lower:
        return "Investor Presentation"
    elif "sale" in desc_lower or "disposal" in desc_lower or "divestment" in desc_lower:
        return "Sale or Disposal"
    elif "contract" in desc_lower or "bagging" in desc_lower:
        return "Bagging/Receiving of Orders/Contracts"
    elif "director" in desc_lower and "change" in desc_lower:
        return "Change in Director(s)"

    return ""


def should_use_positive_impact(filing_type: str) -> bool:
    """Check if positive impact should be fetched"""
    if not filing_type or filing_type not in RELEVANT_FILE_TYPES:
        return False
    return RELEVANT_FILE_TYPES[filing_type].get("positive", False)


def should_use_negative_impact(filing_type: str) -> bool:
    """Check if negative impact should be fetched"""
    if not filing_type or filing_type not in RELEVANT_FILE_TYPES:
        return False
    return RELEVANT_FILE_TYPES[filing_type].get("negative", False)


def download_pdf(url: str, filename: str) -> str:
    """Download PDF and return local filename"""
    try:
        if not url or not url.strip():
            return ""

        docs_dir = Path(__file__).parent / "docs"
        docs_dir.mkdir(exist_ok=True)

        unique_suffix = str(hash(filename + str(time.time())))[:8]
        temp_filename = f"{filename}.{unique_suffix}.pdf"
        path = docs_dir / temp_filename

        if not url.startswith('http'):
            url = f"https://www.nseindia.com{url}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        return temp_filename
    except Exception as e:
        print(f"[ERROR] Failed to download PDF {filename}: {e}")
        return ""


def fetch_stock_data(symbol: str, filing_time: str) -> str:
    """Fetch current stock price"""
    try:
        # Try to get price from Yahoo Finance or similar free API
        # For now, return a placeholder with current timestamp
        return f"Current price: Market data unavailable, timestamp: {filing_time}"
    except Exception as exc:
        print(f"[WARN] Price fetch failed for {symbol}: {exc}")
        return f"Current price: N/A, timestamp: {filing_time}"


def get_pos_impact(file_type: str, use_positive: bool, static_data_path: str = "staticdata.csv") -> str:
    """Get positive impact scenario"""
    if not use_positive:
        return "not applicable for this filing type"

    try:
        if not os.path.exists(static_data_path):
            return "not much specific"

        staticdf = pd.read_csv(static_data_path)
        match = staticdf[staticdf["file type"].str.lower() == file_type.lower()]

        if not match.empty:
            return str(match["positive impct "].values[0])
        else:
            return "not much specific"
    except Exception as e:
        print(f"Error reading static data: {e}")
        return "not much specific"


def get_neg_impact(file_type: str, use_negative: bool, static_data_path: str = "staticdata.csv") -> str:
    """Get negative impact scenario"""
    if not use_negative:
        return "not applicable for this filing type"

    try:
        if not os.path.exists(static_data_path):
            return "not much specific"

        staticdf = pd.read_csv(static_data_path)
        match = staticdf[staticdf["file type"].str.lower() == file_type.lower()]

        if not match.empty:
            return str(match["negtive impct"].values[0])
        else:
            return "not much specific"
    except Exception as e:
        print(f"Error reading static data: {e}")
        return "not much specific"


def generate_trading_signal(pos_impact: str, neg_impact: str, stocktechdata: str, pdf_filename: str = "") -> dict:
    """
    Generate trading signal using Gemini LLM
    """
    try:
        if not GEMINI_API_KEY:
            return {"final_signal": 0, "Confidence": 0, "explanation": "API key not configured"}

        # Build prompt
        prompt = f"""
Analyze this NSE corporate filing and generate a trading signal.

Stock Data: {stocktechdata}
Positive Impact Scenario: {pos_impact}
Negative Impact Scenario: {neg_impact}

Based on the filing content, determine:
1. Trading signal: 1 (BUY), -1 (SELL), or 0 (HOLD)
2. Confidence level (0.0 to 1.0)
3. Detailed explanation

Return ONLY JSON in this format:
{{
    "final_signal": 1,
    "Confidence": 0.85,
    "explanation": "Detailed analysis here..."
}}
"""

        # Setup LLM
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            api_key=GEMINI_API_KEY
        )

        # Prepare message
        message_parts = [{"type": "text", "text": prompt}]

        # Attach PDF if available
        pdf_path = None
        if pdf_filename:
            docs_dir = Path(__file__).parent / "docs"
            pdf_path = docs_dir / pdf_filename
            if pdf_path.exists():
                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
                    message_parts.append({
                        "type": "file",
                        "mime_type": "application/pdf",
                        "base64": pdf_b64
                    })
                    print(f"[INFO] PDF attached: {pdf_filename}")
                except Exception as e:
                    print(f"[WARN] Failed to attach PDF: {e}")

        # Get response
        human_msg = HumanMessage(content=message_parts)
        response = model.invoke([human_msg])

        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)

        # Handle list format
        if isinstance(content, list):
            content = '\n'.join([block.get('text', '') if isinstance(block, dict) else str(block) for block in content])

        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            result = json.loads(content.strip())

        # Cleanup PDF
        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
                print(f"[CLEANUP] Deleted PDF: {pdf_filename}")
            except:
                pass

        return result

    except Exception as e:
        print(f"[ERROR] Signal generation failed: {e}")

        # Cleanup PDF on error
        if pdf_filename:
            docs_dir = Path(__file__).parent / "docs"
            pdf_path = docs_dir / pdf_filename
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except:
                    pass

        return {"final_signal": 0, "Confidence": 0, "explanation": f"Error: {str(e)}"}


def process_filing(announcement: dict) -> dict:
    """
    Process a single NSE filing and generate trading signal

    Args:
        announcement: Dict with filing data

    Returns:
        Signal dict or None if not actionable
    """
    try:
        symbol = announcement.get('symbol', '')
        desc = announcement.get('desc', '')
        pdf_url = announcement.get('attchmntFile', '')
        filing_time = announcement.get('sort_date', announcement.get('dt', ''))

        print(f"[PROCESS] Analyzing {symbol}: {desc[:60]}...")

        # Map filing type
        filing_type = map_filing_type(desc)
        if not filing_type:
            print(f"[SKIP] Not a relevant filing type")
            return None

        print(f"[INFO] Filing type: {filing_type}")

        # Download PDF
        if not pdf_url:
            print(f"[SKIP] No PDF URL")
            return None

        filename = pdf_url.split("/")[-1]
        pdf_filename = download_pdf(pdf_url, filename)
        if not pdf_filename:
            print(f"[SKIP] PDF download failed")
            return None

        # Get impact scenarios
        use_positive = should_use_positive_impact(filing_type)
        use_negative = should_use_negative_impact(filing_type)

        pos_impact = get_pos_impact(filing_type, use_positive)
        neg_impact = get_neg_impact(filing_type, use_negative)

        # Get stock data
        stock_data = fetch_stock_data(symbol, filing_time)

        # Generate signal
        print(f"[LLM] Generating signal for {symbol}...")
        llm_result = generate_trading_signal(pos_impact, neg_impact, stock_data, pdf_filename)

        signal = llm_result.get('final_signal', 0)
        confidence = llm_result.get('Confidence', 0)
        explanation = llm_result.get('explanation', '')

        print(f"[RESULT] {symbol}: Signal={signal}, Confidence={confidence}")

        # Only return actionable signals
        if signal in (1, -1):
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'explanation': explanation,
                'filing_time': filing_time,
                'subject_of_announcement': filing_type,
                'reference_price': None,  # Could extract from stock_data if available
            }

        return None

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return None


# Test function
def test_signal_generation():
    """Test the signal generation with sample data"""
    test_announcement = {
        'symbol': 'RELIANCE',
        'desc': 'Outcome of Board Meeting',
        'attchmntFile': 'https://www.nseindia.com/api/corporate-announcements/sample.pdf',
        'sort_date': '2024-01-15 10:30:00'
    }

    result = process_filing(test_announcement)
    if result:
        print(f"[TEST] Generated signal: {result}")
    else:
        print("[TEST] No actionable signal generated")


if __name__ == "__main__":
    # Run test
    test_signal_generation()
