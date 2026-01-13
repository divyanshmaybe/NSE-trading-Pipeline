# -*- coding: utf-8 -*-
"""
NSE Trading Signal Monitor (Telegram by default)

Integrates NSE scraper and signal generator to poll announcements, generate signals, and send notifications.

Environment Variables:
    GEMINI_API_KEY: Google Gemini API key for LLM
    TELEGRAM_BOT_TOKEN: Telegram bot token for notifications (required)
    TELEGRAM_CHAT_IDS: Comma-separated Telegram chat IDs to send messages to
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

import io as _io
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('nse_monitor.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler(sys.stdout)
try:
    stream_handler.setStream(_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True))
except Exception:
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            stream_handler = logging.StreamHandler(sys.stdout)
    except Exception:
        pass

stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Load environment variables
load_dotenv()

# Import Telegram notifier (preferred)
try:
    from telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Telegram notifier not available: {e}")
    TelegramNotifier = None
    TELEGRAM_AVAILABLE = False

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


# Market hours (IST)
MARKET_OPEN = (9, 15)  # 9:15 AM
MARKET_CLOSE = (15, 30)  # 3:30 PM

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# NOTE: Legacy Twilio/WhatsApp support has been removed.
# Telegram is the only supported notifier now.




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
    """Main monitor that polls NSE and sends notifications"""

    def __init__(self, dry_run: bool = False):
        # Initialize Telegram notifier (required for notifications)
        if TELEGRAM_AVAILABLE:
            try:
                self.notifier = TelegramNotifier(dry_run=dry_run)
                self.notification_type = "Telegram"
            except Exception as e:
                logger.warning(f"Telegram notifier failed: {e}")
                if not dry_run:
                    # If not in dry-run and Telegram init fails, surface the error
                    raise
                self.notifier = None
                self.notification_type = "None (dry-run)"
        else:
            # Telegram must be configured; without it notifications are disabled
            logger.warning("Telegram notifier is not available; notifications will be disabled")
            self.notifier = None
            self.notification_type = "None"

        self.seen_announcements = set()
        self.load_seen_announcements()
        logger.info(f"✅ NSE Signal Monitor initialized (notifications: {self.notification_type})")

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
        if self.notifier:
            self.notifier.send_startup_message()
        
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

                # Debug: Show what announcements were found
                logger.info(f"📋 Processing {len(announcements)} new announcements:")
                for i, ann in enumerate(announcements[:5], 1):  # Show first 5
                    logger.info(f"   {i}. {ann.get('symbol', 'N/A')}: {ann.get('desc', 'N/A')[:60]}...")
                if len(announcements) > 5:
                    logger.info(f"   ... and {len(announcements) - 5} more")
                
                # Process announcements and get signals
                signals = process_announcements(announcements)

                # Send notifications for each signal
                for signal_data in signals:
                    try:
                        logger.info(f"📱 Sending notification for {signal_data['symbol']}...")
                        success = self.notifier.send_signal(signal_data) if self.notifier else False

                        if success:
                            logger.info(f"✅ Signal sent for {signal_data['symbol']}")
                        else:
                            logger.warning(f"⚠️ Failed to send signal for {signal_data['symbol']}")

                    except Exception as e:
                        logger.error(f"❌ Error sending notification for {signal_data.get('symbol', 'unknown')}: {e}", exc_info=True)
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
                    if self.notifier:
                        self.notifier.send_error_alert(f"Monitor crashed after {max_errors} errors: {e}")
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
    import argparse

    parser = argparse.ArgumentParser(description='NSE Signal Monitor')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode (no external notifications/LLM calls)')
    args = parser.parse_args()

    dry_run = args.dry_run or os.getenv('DRY_RUN', '').lower() in {'1', 'true', 'yes'}

    # Validate API keys (if not in dry-run)
    if not dry_run and not GEMINI_API_KEY and not os.getenv("GEMINI_API_KEYS"):
        logger.error("❌ GEMINI_API_KEY or GEMINI_API_KEYS not set in environment")
        sys.exit(1)

    # Create and run monitor
    monitor = NSESignalMonitor(dry_run=dry_run)
    monitor.run()


if __name__ == "__main__":
    main()
