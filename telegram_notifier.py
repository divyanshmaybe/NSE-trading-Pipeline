import os
import requests
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    """Send notifications via Telegram Bot API"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = bool(dry_run) or os.getenv('DRY_RUN', '').lower() in {'1', 'true', 'yes'}

        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
        self.chat_ids = [cid.strip() for cid in os.getenv('TELEGRAM_CHAT_IDS', '').split(',') if cid.strip()]

        if self.dry_run:
            print("[DRY-RUN] Telegram notifier running in dry-run mode (no messages will be sent)")
            self.bot_token = self.bot_token or "dummy_token"  # Allow dry-run without real token
            self.chat_ids = self.chat_ids or ["dummy_chat"]
            return

        # Validate configuration
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment variables")

        if not self.chat_ids:
            raise ValueError("TELEGRAM_CHAT_IDS not set in environment variables")

        # Test connection (don't fail, just warn)
        if not self._test_connection():
            print("[WARN] Telegram bot connection test failed. Check bot token and chat IDs.")
            print("[INFO] Telegram notifications will be skipped until configured properly.")

        print(f"✅ Telegram notifier initialized: {len(self.chat_ids)} chat(s)")

    def _test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            return response.status_code == 200 and response.json().get('ok')
        except Exception:
            return False

    def send_message(self, message: str) -> bool:
        """Send message to all configured chats"""
        if self.dry_run:
            print(f"[DRY-RUN] Would send Telegram message to {len(self.chat_ids)} chats:")
            print(f"  Message: {message[:100]}...")
            return True

        success_count = 0

        for chat_id in self.chat_ids:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }

                response = requests.post(url, data=data, timeout=15)

                if response.status_code == 200 and response.json().get('ok'):
                    print(f"✅ Telegram sent to chat {chat_id}")
                    success_count += 1
                else:
                    error = response.json().get('description', 'Unknown error')
                    print(f"❌ Telegram failed to chat {chat_id}: {error}")

            except Exception as e:
                print(f"❌ Telegram exception for chat {chat_id}: {e}")

        return success_count > 0

    def send_signal(self, signal_data: Dict) -> bool:
        """
        Send trading signal to Telegram
        """
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)
            explanation = signal_data.get('explanation', 'No explanation')
            filing_time = signal_data.get('filing_time', 'Unknown')
            subject = signal_data.get('subject_of_announcement', 'Unknown')

            # Format signal emoji
            signal_emoji = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"

            # Format confidence as percentage
            confidence_pct = f"{confidence * 100:.1f}%"

            # Build message
            message = f"""🚨 *NSE TRADING SIGNAL* 🚨

*Symbol:* `{symbol}`
*Action:* {signal_emoji}
*Confidence:* {confidence_pct}

*Filing Type:* {subject}
*Time:* {filing_time}

*Analysis:*
{explanation[:800]}{"..." if len(explanation) > 800 else ""}

---
⚠️ This is automated analysis. Trade at your own risk."""

            return self.send_message(message)

        except Exception as e:
            print(f"❌ Failed to send Telegram signal: {e}")
            return False

    def send_startup_message(self):
        """Send startup notification"""
        try:
            from nse import DEMO_MODE, POLL_INTERVAL, MARKET_OPEN, MARKET_CLOSE

            mode = "DEMO MODE (24/7)" if DEMO_MODE else "MARKET HOURS ONLY"
            startup_message = f"""🤖 *NSE Signal Monitor Started*

Mode: {mode}
Polling: Every {POLL_INTERVAL} seconds
Market Hours: {MARKET_OPEN[0]}:{MARKET_OPEN[1]:02d} - {MARKET_CLOSE[0]}:{MARKET_CLOSE[1]:02d} IST

Ready to monitor NSE filings! 🚀"""

            test_message = f"""✅ *System Test - NSE Monitor Active*

Time: {__import__('datetime').datetime.now().strftime('%H:%M:%S IST')}
Status: Online and monitoring
Next poll: Every {POLL_INTERVAL} seconds

This confirms Telegram notifications are working! 📱"""

            # Send startup message
            startup_success = self.send_message(startup_message)

            # Send test message after a short delay
            if not self.dry_run:
                __import__('time').sleep(2)
            test_success = self.send_message(test_message)

            if startup_success and test_success:
                print("✅ Startup and test messages sent successfully")
            elif startup_success:
                print("✅ Startup message sent, test message failed")

        except Exception as e:
            print(f"❌ Could not send startup/test messages: {e}")

    def send_error_alert(self, error_msg: str):
        """Send error alert"""
        try:
            message = f"""⚠️ *NSE Monitor Error*

{error_msg}

Monitor may need attention."""
            self.send_message(message)
        except Exception as e:
            print(f"❌ Could not send error alert: {e}")
