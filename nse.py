# -*- coding: utf-8 -*-
"""
NSE Trading Signal Monitor with WhatsApp Alerts

This script integrates the NSE scraper and signal generator to:
1. Poll NSE announcements every minute
2. Generate trading signals using LLM
3. Send actionable signals (BUY/SELL) to WhatsApp via Twilio

Usage:
    python nse_whatsapp_monitor.py

Environment Variables Required:
    GEMINI_API_KEY: Google Gemini API key for LLM
    TWILIO_ACCOUNT_SID: Twilio account SID
    TWILIO_AUTH_TOKEN: Twilio auth token
    TWILIO_WHATSAPP_FROM: Twilio WhatsApp number (e.g., whatsapp:+14155238886)
    TWILIO_WHATSAPP_TO: Your WhatsApp number (e.g., whatsapp:+919876543210)
    DEMO_MODE: Set to 'true' to run 24/7 (default: false for market hours only)
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nse_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import Twilio
try:
    from twilio.rest import Client
except ImportError:
    logger.error("Twilio not installed. Run: pip install twilio")
    sys.exit(1)

# Add project paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import your existing modules
try:
    from nse_web_scraper import fetch_nse_announcements, is_relevant_announcement, process_announcements
except ImportError as e:
    logger.error(f"Failed to import NSE modules: {e}")
    logger.error("Make sure nse_web_scraper.py and nse_sentiment.py are in the same directory")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

POLL_INTERVAL = 60  # Poll every 1 minute
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")  # e.g., whatsapp:+14155238886
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")  # e.g., whatsapp:+919876543210

# Market hours (IST)
MARKET_OPEN = (9, 15)  # 9:15 AM
MARKET_CLOSE = (15, 30)  # 3:30 PM

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ============================================================================
# TWILIO WHATSAPP CLIENT
# ============================================================================

class WhatsAppNotifier:
    """Send WhatsApp messages via Twilio"""
    
    def __init__(self):
        self.validate_config()
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info(f"✅ WhatsApp notifier initialized: {TWILIO_WHATSAPP_FROM} -> {TWILIO_WHATSAPP_TO}")
    
    def validate_config(self):
        """Validate Twilio configuration"""
        missing = []
        if not TWILIO_ACCOUNT_SID:
            missing.append("TWILIO_ACCOUNT_SID")
        if not TWILIO_AUTH_TOKEN:
            missing.append("TWILIO_AUTH_TOKEN")
        if not TWILIO_WHATSAPP_FROM:
            missing.append("TWILIO_WHATSAPP_FROM")
        if not TWILIO_WHATSAPP_TO:
            missing.append("TWILIO_WHATSAPP_TO")
        
        if missing:
            raise ValueError(f"Missing Twilio config: {', '.join(missing)}")
        
        # Validate WhatsApp number format
        if not TWILIO_WHATSAPP_FROM.startswith("whatsapp:"):
            raise ValueError("TWILIO_WHATSAPP_FROM must start with 'whatsapp:' (e.g., whatsapp:+14155238886)")
        if not TWILIO_WHATSAPP_TO.startswith("whatsapp:"):
            raise ValueError("TWILIO_WHATSAPP_TO must start with 'whatsapp:' (e.g., whatsapp:+919876543210)")
    
    def send_signal(self, signal_data: Dict) -> bool:
        """
        Send trading signal to WhatsApp
        
        Args:
            signal_data: Dict with keys: symbol, signal, confidence, explanation, 
                        filing_time, subject_of_announcement, reference_price
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)
            explanation = signal_data.get('explanation', 'No explanation')
            filing_time = signal_data.get('filing_time', 'Unknown')
            subject = signal_data.get('subject_of_announcement', 'Unknown')
            price = signal_data.get('reference_price')
            
            # Format signal emoji
            signal_emoji = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
            
            # Format confidence as percentage
            confidence_pct = f"{confidence * 100:.1f}%"
            
            # Format price
            price_str = f"₹{price:.2f}" if price else "N/A"
            
            # Build message
            message = f"""
🚨 *NSE TRADING SIGNAL* 🚨

*Symbol:* {symbol}
*Action:* {signal_emoji}
*Confidence:* {confidence_pct}
*Current Price:* {price_str}

*Filing Type:* {subject}
*Time:* {filing_time}

*Analysis:*
{explanation[:500]}{"..." if len(explanation) > 500 else ""}

---
⚠️ This is automated analysis. Trade at your own risk.
""".strip()
            
            # Send via Twilio
            logger.info(f"📱 Sending WhatsApp message for {symbol} ({signal_emoji})...")
            msg = self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=message,
                to=TWILIO_WHATSAPP_TO
            )
            
            logger.info(f"✅ WhatsApp sent successfully: SID={msg.sid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send WhatsApp: {e}")
            return False
    
    def send_startup_message(self):
        """Send startup notification"""
        try:
            mode = "DEMO MODE (24/7)" if DEMO_MODE else "MARKET HOURS ONLY"
            message = f"""
🤖 *NSE Signal Monitor Started*

Mode: {mode}
Polling: Every {POLL_INTERVAL} seconds
Market Hours: {MARKET_OPEN[0]}:{MARKET_OPEN[1]:02d} - {MARKET_CLOSE[0]}:{MARKET_CLOSE[1]:02d} IST

Ready to monitor NSE filings! 🚀
""".strip()
            
            self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=message,
                to=TWILIO_WHATSAPP_TO
            )
            logger.info("✅ Startup notification sent")
        except Exception as e:
            logger.warning(f"⚠️ Could not send startup notification: {e}")
    
    def send_error_alert(self, error_msg: str):
        """Send error alert"""
        try:
            message = f"""
⚠️ *NSE Monitor Error*

{error_msg}

Monitor may need attention.
""".strip()
            
            self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=message,
                to=TWILIO_WHATSAPP_TO
            )
            logger.info("✅ Error alert sent")
        except Exception as e:
            logger.warning(f"⚠️ Could not send error alert: {e}")


# ============================================================================
# MARKET HOURS CHECK
# ============================================================================

def is_market_open() -> bool:
    """Check if NSE is currently open"""
    if DEMO_MODE:
        return True
    
    from datetime import datetime, time as dt_time, timezone, timedelta
    
    # Use UTC offset for IST (UTC+5:30) - works on all Python versions
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    current_time = now.time()
    
    # Weekend check
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    market_open_time = dt_time(MARKET_OPEN[0], MARKET_OPEN[1])
    market_close_time = dt_time(MARKET_CLOSE[0], MARKET_CLOSE[1])
    
    return market_open_time <= current_time <= market_close_time


# ============================================================================
# MAIN MONITOR
# ============================================================================

class NSESignalMonitor:
    """Main monitor that polls NSE and sends WhatsApp alerts"""

    def __init__(self):
        self.whatsapp = WhatsAppNotifier()
        self.seen_announcements = set()
        self.load_seen_announcements()
        logger.info("✅ NSE Signal Monitor initialized")

    def load_seen_announcements(self):
        """Load previously seen announcement IDs"""
        try:
            if os.path.exists("processed_announcements.json"):
                with open("processed_announcements.json", 'r') as f:
                    data = json.load(f)
                    self.seen_announcements = set(data.get('processed_ids', []))
                    logger.info(f"📚 Loaded {len(self.seen_announcements)} seen announcements")
        except Exception as e:
            logger.warning(f"⚠️ Could not load seen announcements: {e}")

    def save_seen_announcements(self):
        """Save seen announcement IDs"""
        try:
            data = {
                'processed_ids': list(self.seen_announcements),
                'last_updated': datetime.now().isoformat()
            }
            with open("processed_announcements.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Could not save seen announcements: {e}")

    def poll_nse(self) -> List[Dict]:
        """Poll NSE for new announcements"""
        try:
            # Fetch all announcements
            announcements = fetch_nse_announcements()

            # Filter to only new announcements
            new_announcements = []
            for ann in announcements:
                seq_id = ann.get('seq_id', '')
                if seq_id and seq_id not in self.seen_announcements:
                    new_announcements.append(ann)
                    self.seen_announcements.add(seq_id)

            if new_announcements:
                logger.info(f"✅ Found {len(new_announcements)} new announcements")

            return new_announcements

        except Exception as e:
            logger.error(f"❌ Error polling NSE: {e}", exc_info=True)
            return []
    
    def run(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting NSE Signal Monitor...")
        logger.info(f"📊 DEMO_MODE: {DEMO_MODE}")
        logger.info(f"⏱️ Polling interval: {POLL_INTERVAL} seconds")
        
        # Send startup notification
        self.whatsapp.send_startup_message()
        
        error_count = 0
        max_errors = 5
        
        while True:
            try:
                # Check market hours
                if not is_market_open():
                    logger.info("🔴 Market closed - waiting...")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                logger.info(f"🔍 Polling NSE at {datetime.now().strftime('%H:%M:%S')}")
                
                # Poll for new announcements
                announcements = self.poll_nse()
                
                if not announcements:
                    logger.info("📭 No new announcements")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                # Process announcements and get signals
                signals = process_announcements(announcements)

                # Send WhatsApp notifications for each signal
                for signal_data in signals:
                    try:
                        logger.info(f"📱 Sending WhatsApp for {signal_data['symbol']}...")
                        success = self.whatsapp.send_signal(signal_data)

                        if success:
                            logger.info(f"✅ Signal sent for {signal_data['symbol']}")
                        else:
                            logger.warning(f"⚠️ Failed to send signal for {signal_data['symbol']}")

                    except Exception as e:
                        logger.error(f"❌ Error sending WhatsApp for {signal_data.get('symbol', 'unknown')}: {e}", exc_info=True)
                        continue
                
                # Save seen announcements
                self.save_seen_announcements()
                
                # Reset error count on success
                error_count = 0
                
                # Wait before next poll
                logger.info(f"⏳ Waiting {POLL_INTERVAL} seconds before next poll...")
                time.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitor stopped by user")
                self.save_seen_announcements()
                break
            
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error in monitoring loop ({error_count}/{max_errors}): {e}", exc_info=True)
                
                if error_count >= max_errors:
                    logger.critical("🚨 Too many errors - sending alert and stopping")
                    self.whatsapp.send_error_alert(f"Monitor crashed after {max_errors} errors: {e}")
                    break
                
                # Exponential backoff on errors
                wait_time = min(300, POLL_INTERVAL * (2 ** error_count))
                logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    # Validate API keys
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not set in environment")
        sys.exit(1)
    
    # Create and run monitor
    monitor = NSESignalMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
