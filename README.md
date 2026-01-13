# NSE Filings Automation: Motivation & Problem Statement

NSE filings are official financial reports, disclosures, and corporate documents submitted by companies listed on the National Stock Exchange (NSE). These filings provide crucial information about a company’s financial performance, management decisions, operations, and corporate actions, all of which are made public to ensure transparency and compliance with SEBI regulations.

Major financial sources such as CNBC, Economic Times, and large financial research firms regularly analyze these filings to draw insights and publish conclusions. When they identify impactful information that could influence a company’s market value, they release related news articles or headlines. As soon as this information is published, it often triggers significant stock price movements within a short time window.

## Problem Identified

The process of analyzing these filings manually and publishing actionable insights involves both time and cost. By the time such updates are released on platforms like Twitter or financial news sites, the stock price often reacts within minutes, making it difficult and risky to capitalize on these movements.

## Proposed Solution

To overcome this limitation, we developed a fully automated real-time analysis system. Through extensive research, we identified 12–13 filing types that typically cause substantial short-term market reactions. We then created a static knowledge base containing detailed positive and negative impact scenarios for each filing type.

Our system continuously fetches filings from NSE Corporate Announcements every minute. Whenever a filing matches our predefined set, the system:
1. Extracts information from the attached PDF document,
2. Fetches the technical data of the corresponding stock for the past hour, and
3. Combines this data with the positive and negative impact scenarios from our knowledge base.

This combined information is then sent to an LLM through a structured financial prompt, enabling the model to analyze the filing and return a structured output containing:
- A trading signal (1 / -1 / 0 (HOLD)),
- A confidence score, and
- A concise explanation for the decision.

All of this occurs within 1-1.5 minutes, enabling near real-time analysis and execution.

# NSE Trading Pipeline

This repository contains a pipeline for automated trading and sentiment analysis using data from the National Stock Exchange (NSE) of India. The system is designed to scrape announcements, analyze sentiment, generate trading signals, and send notifications via Telegram.

## Overview

The pipeline consists of the following main components:

- **Web Scraper**: Collects the latest announcements from the NSE website.
- **Sentiment Analysis**: Processes announcements to determine market sentiment.
- **Signal Generation**: Uses sentiment and other data to generate trading signals.
- **Signal Validation**: Validates generated signals before acting on them.
- **Notification System**: Sends trading signals and updates to a Telegram channel.

## File Structure

- `nse_web_scraper.py`: Scrapes announcements from the NSE website and saves them to `processed_announcements.json`.
- `nse_sentiment.py`: Analyzes the sentiment of announcements and updates the data.
- `nse.py`: Contains core logic for interacting with the NSE and managing data.
- `nse_signal_generation_prompt.yaml`: Configuration for signal generation logic.
- `nse_signal_validation_prompt.yaml`: Configuration for signal validation logic.
- `telegram_notifier.py`: Sends notifications to Telegram.
- `run_one_live_iteration.py`: Runs a single iteration of the pipeline (scraping, analysis, signal generation, notification).
- `exhausted_keys.json`: Tracks used API keys or tokens.
- `processed_announcements.json`: Stores processed announcements and their sentiment.
- `staticdata.csv`: Contains static reference data.
- `requirements.txt`: Lists required Python packages.

## Pipeline Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd nse_trading_folder
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Telegram

- Set up a Telegram bot and get the API token.
- Update `telegram_notifier.py` with your bot token and chat ID.

### 4. Run the Pipeline

To execute a single live iteration of the pipeline:

```bash
python run_one_live_iteration.py
```

This will:
- Scrape the latest announcements from NSE.
- Analyze sentiment.
- Generate and validate trading signals.
- Send notifications to Telegram.

### 5. Scheduling (Optional)

To automate the pipeline, schedule `run_one_live_iteration.py` using Windows Task Scheduler or a cron job (on Linux).

## How It Works

1. **Scraping**: The scraper fetches new announcements and updates `processed_announcements.json`.
2. **Sentiment Analysis**: Each announcement is analyzed for sentiment (positive, negative, neutral) using rules or models defined in `nse_sentiment.py`.
3. **Signal Generation**: Based on sentiment and other criteria, trading signals are generated using prompts/configs in the YAML files.
4. **Signal Validation**: Signals are validated to avoid false positives.
5. **Notification**: Validated signals are sent to a Telegram channel for action.

## Notes

- API keys and sensitive data should not be committed to the repository.
- The pipeline is modular; you can run or modify individual components as needed.
- For troubleshooting, check log outputs and ensure all dependencies are installed.

## Contributing

Contributions are welcome. Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
