import os
import logging
from nse_web_scraper import fetch_nse_announcements, is_relevant_announcement
from nse_sentiment import process_filing
from telegram_notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('one-shot')

SEND_CAP = int(os.getenv('SEND_CAP', '0'))


def main():
    # Initialize Telegram notifier (required for notifications)
    try:
        notifier = TelegramNotifier(dry_run=False)
        logger.info("Using Telegram notifier")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram notifier: {e}")
        return

    logger.info("Fetching announcements from NSE...")
    anns = fetch_nse_announcements()
    if not anns:
        logger.info("No announcements fetched. Exiting.")
        return

    # Filter relevant announcements
    relevant = [a for a in anns if is_relevant_announcement(a)]
    logger.info(f"Found {len(relevant)} relevant announcements")

    # Process up to SEND_CAP announcements (0 => no cap)
    to_process = relevant
    if SEND_CAP > 0:
        to_process = relevant[:SEND_CAP]
        logger.info(f"SEND_CAP set: processing first {SEND_CAP} announcements")

    signals = []
    for ann in to_process:
        logger.info(f"Processing {ann.get('symbol', 'N/A')}: {ann.get('desc', '')[:80]}...")
        try:
            sig = process_filing(ann)
            if sig:
                signals.append(sig)
                logger.info(f"Generated signal for {sig['symbol']} -> {sig['signal']}")
            else:
                logger.info(f"No actionable signal for {ann.get('symbol')} (skipping)")
        except Exception as e:
            logger.error(f"Error processing filing: {e}")

    if not signals:
        logger.info("No actionable signals generated. Exiting.")
        return

    logger.info(f"Sending {len(signals)} messages (live)")
    sent = 0
    for sig in signals:
        try:
            ok = notifier.send_signal(sig)
            if ok:
                sent += 1
                logger.info(f"Sent signal for {sig['symbol']}")
            else:
                logger.warning(f"Failed to send signal for {sig['symbol']}")
        except Exception as e:
            logger.error(f"Exception sending signal for {sig.get('symbol', 'N/A')}: {e}")

    logger.info(f"Done. Successfully sent {sent}/{len(signals)} messages")


if __name__ == '__main__':
    main()
