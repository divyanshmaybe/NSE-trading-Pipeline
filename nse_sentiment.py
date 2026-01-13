import os
import re
import json
import time
import yaml
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader
import base64

load_dotenv()

def load_relevant_file_types() -> Dict[str, Dict[str, bool]]:
    try:
        import pandas as pd
        if not os.path.exists("staticdata.csv"):
            return {}
        df = pd.read_csv("staticdata.csv")
        result = {}
        for _, row in df.iterrows():
            file_type = row['file type'].strip()
            pos_impact = row['positive impct '].strip()
            neg_impact = row['negtive impct'].strip()
            positive = pos_impact != "" and pos_impact.lower() != "not applicable for this filing type"
            negative = neg_impact != "" and neg_impact.lower() != "not applicable for this filing type"
            result[file_type] = {"positive": positive, "negative": negative}
        return result
    except Exception:
        return {}

RELEVANT_FILE_TYPES = load_relevant_file_types()

api_keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
GEMINI_API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]

EXHAUSTED_KEYS_FILE = Path(__file__).parent / "exhausted_keys.json"

def load_exhausted_keys():
    """Load exhausted keys for today"""
    try:
        if EXHAUSTED_KEYS_FILE.exists():
            with open(EXHAUSTED_KEYS_FILE, 'r') as f:
                data = json.load(f)
                today = datetime.now().strftime('%Y-%m-%d')
                if data.get('date') == today:
                    return set(data.get('exhausted_keys', []))
    except Exception as e:
        print(f"[WARN] Could not load exhausted keys: {e}")
    return set()

def save_exhausted_keys(exhausted_keys):
    """Save exhausted keys for today"""
    try:
        data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'exhausted_keys': list(exhausted_keys)
        }
        with open(EXHAUSTED_KEYS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save exhausted keys: {e}")


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

        # Quick validation: ensure file is non-empty and contains at least one PDF page
        try:
            full_path = docs_dir / temp_filename
            if full_path.stat().st_size == 0:
                print(f"[WARN] Downloaded PDF is empty: {temp_filename}")
                try:
                    full_path.unlink()
                except:
                    pass
                return ""

            reader = PdfReader(str(full_path))
            num_pages = len(reader.pages) if hasattr(reader, 'pages') else 0
            if num_pages == 0:
                print(f"[WARN] Downloaded PDF has no pages: {temp_filename}")
                try:
                    full_path.unlink()
                except:
                    pass
                return ""

        except Exception as e:
            print(f"[WARN] Could not validate PDF {temp_filename}: {e}")
            # If validation fails, we still return the filename to allow caller decide

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


def get_pos_impact(file_type: str, use_positive: bool) -> str:
    if not use_positive:
        return "not applicable for this filing type"
    try:
        import pandas as pd
        if not os.path.exists("staticdata.csv"):
            return "Positive impact expected"
        df = pd.read_csv("staticdata.csv")
        match = df[df["file type"].str.lower() == file_type.lower()]
        if not match.empty:
            return str(match["positive impct "].values[0])
        else:
            return "Positive impact expected"
    except Exception:
        return "Positive impact expected"


def get_neg_impact(file_type: str, use_negative: bool) -> str:
    if not use_negative:
        return "not applicable for this filing type"
    try:
        import pandas as pd
        if not os.path.exists("staticdata.csv"):
            return "Negative impact expected"
        df = pd.read_csv("staticdata.csv")
        match = df[df["file type"].str.lower() == file_type.lower()]
        if not match.empty:
            return str(match["negtive impct"].values[0])
        else:
            return "Negative impact expected"
    except Exception:
        return "Negative impact expected"


def load_prompt_config(prompt_file: str) -> dict:
    """Load prompt configuration from YAML file"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load prompt config {prompt_file}: {e}")
        return None


def try_api_call_with_fallback(model_func, *args, **kwargs):
    """Try API call with multiple keys fallback, skipping known exhausted keys"""
    exhausted_keys = load_exhausted_keys()

    for i, api_key in enumerate(GEMINI_API_KEYS):
        key_index = i + 1

        # Skip keys known to be exhausted today
        if api_key in exhausted_keys:
            print(f"[API] Skipping exhausted key {key_index}, trying next...")
            continue

        try:
            print(f"[API] Trying key {key_index}/{len(GEMINI_API_KEYS)}...")
            kwargs['api_key'] = api_key
            result = model_func(*args, **kwargs)

            # If this key worked and we had skipped exhausted ones, save the working key index
            # This helps start with the working key for future filings
            if exhausted_keys:
                save_exhausted_keys(exhausted_keys)
                print(f"[API] Key {key_index} working, will start with this key for future filings")

            return result

        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"[API] Key {key_index} quota exhausted, marking as exhausted and trying next...")
                exhausted_keys.add(api_key)
                save_exhausted_keys(exhausted_keys)
                continue
            else:
                # Non-quota error, re-raise
                raise e

    # All available keys exhausted
    raise Exception("All available API keys exhausted or quota limits reached")


def generate_trading_signal(pos_impact: str, neg_impact: str, stocktechdata: str, pdf_filename: str = "") -> dict:
    """
    Generate trading signal using two-step Gemini LLM process.

    In DRY_RUN mode (env var 'DRY_RUN' set) this returns a deterministic stub
    result so the system can be tested without calling external LLM APIs.
    """
    # Dry-run stub
    if os.getenv('DRY_RUN', '').lower() in {'1', 'true', 'yes'}:
        print('[DRY-RUN] generate_trading_signal: returning stubbed signal (no LLM call)')
        # Simple heuristic: return BUY (1) so dry-run generates actionable signals
        return {
            "final_signal": 1,
            "Confidence": 0.75,
            "explanation": "Dry-run stub: simulated BUY signal for testing"
        }

    try:
        if not GEMINI_API_KEYS:
            return {"final_signal": 0, "Confidence": 0, "explanation": "No API keys configured"}

        # Load prompt configurations
        generation_config = load_prompt_config("nse_signal_generation_prompt.yaml")
        validation_config = load_prompt_config("nse_signal_validation_prompt.yaml")

        if not generation_config or not validation_config:
            return {"final_signal": 0, "Confidence": 0, "explanation": "Failed to load prompt configurations"}

        # ===== STEP 1: SIGNAL GENERATION =====
        print(f"[LLM-1] Generating initial signal with {generation_config['model']}...")

        # Build generation prompt
        generation_prompt = generation_config['prompt_template'].format(
            stocktechdata=stocktechdata,
            pos_impact=pos_impact,
            neg_impact=neg_impact
        )

        # Prepare message with PDF
        message_parts = [{"type": "text", "text": generation_prompt}]

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
                    print(f"[INFO] PDF attached for generation: {pdf_filename}")
                except Exception as e:
                    print(f"[WARN] Failed to attach PDF for generation: {e}")

        # Try configured model first, then fall back to sensible defaults if quota / 429 occurs
        gen_primary = generation_config.get('model')
        gen_fallbacks = generation_config.get('fallback_models', ["gemini-2.5-flash", "gemini-2.5", "gemini-2.0-flash"])
        gen_models_to_try = [m for m in ([gen_primary] + gen_fallbacks) if m]

        gen_response = None
        last_exc = None
        used_gen_model = None

        # Validate PDF before attaching: ensure it has pages
        pdf_attached = False
        if pdf_path and pdf_path.exists():
            try:
                reader = PdfReader(str(pdf_path))
                num_pages = len(reader.pages) if hasattr(reader, 'pages') else 0
                if num_pages <= 0:
                    print(f"[WARN] PDF {pdf_filename} has no pages, skipping attachment")
                    try:
                        pdf_path.unlink()
                    except:
                        pass
                    pdf_attached = False
                else:
                    pdf_attached = True
            except Exception as e:
                print(f"[WARN] Could not read PDF {pdf_filename} before attaching: {e}")
                pdf_attached = False

        for model_name in gen_models_to_try:
            print(f"[LLM-1] Trying model '{model_name}' for generation...")

            def create_gen_model(api_key, model_name=model_name):
                # Lazy import to avoid heavy transformer/torch dependencies during module import
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=generation_config.get('temperature', 0.1),
                    api_key=api_key
                )

            def call_gen_model(api_key, model_name=model_name, attach_pdf=True):
                # Build the content for this call; copy message parts and optionally remove file part
                content_parts = list(message_parts)
                if not attach_pdf:
                    content_parts = [p for p in content_parts if p.get('type') != 'file']
                model = create_gen_model(api_key, model_name=model_name)
                return model.invoke([HumanMessage(content=content_parts)])

            try:
                # First try with file if it was successfully validated and attached
                if pdf_attached:
                    try:
                        gen_response = try_api_call_with_fallback(lambda api_key: call_gen_model(api_key, model_name=model_name, attach_pdf=True))
                        used_gen_model = model_name
                        print(f"[LLM-1] Model '{model_name}' succeeded for generation (with PDF)")
                        break
                    except Exception as e:
                        # If the error indicates invalid document (e.g., 'The document has no pages.') retry once without the file
                        msg = str(e)
                        if 'INVALID_ARGUMENT' in msg or 'document has no pages' in msg.lower():
                            print(f"[LLM-1] Model '{model_name}' rejected PDF: retrying without file")
                            try:
                                gen_response = try_api_call_with_fallback(lambda api_key: call_gen_model(api_key, model_name=model_name, attach_pdf=False))
                                used_gen_model = model_name
                                print(f"[LLM-1] Model '{model_name}' succeeded for generation (without PDF)")
                                break
                            except Exception as e2:
                                last_exc = e2
                                # Continue to next model
                                print(f"[LLM-1] Retry without file also failed: {e2}")
                                continue
                        else:
                            last_exc = e
                            # For quota errors, try next model
                            if 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower() or '429' in msg:
                                print(f"[LLM-1] Model '{model_name}' resource/quota issue, trying next model...")
                                continue
                            else:
                                # Non-quota, non-PDF errors should be surfaced
                                raise
                else:
                    gen_response = try_api_call_with_fallback(lambda api_key: call_gen_model(api_key, model_name=model_name, attach_pdf=False))
                    used_gen_model = model_name
                    print(f"[LLM-1] Model '{model_name}' succeeded for generation (no PDF)")
                    break

            except Exception as e:
                last_exc = e
                msg = str(e)
                if 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower() or '429' in msg:
                    print(f"[LLM-1] Model '{model_name}' resource/quota issue, trying next model...")
                    continue
                else:
                    # Non-quota errors should be surfaced
                    raise

        if gen_response is None:
            raise Exception(f"Generation failed for all tried models. Last error: {last_exc}")

        # Extract generation content
        gen_content = gen_response.content if hasattr(gen_response, 'content') else str(gen_response)
        if isinstance(gen_content, list):
            gen_content = '\n'.join([block.get('text', '') if isinstance(block, dict) else str(block) for block in gen_content])

        # Extract JSON from generation
        gen_json_match = re.search(r'```json\s*(.*?)\s*```', gen_content, re.DOTALL)
        if gen_json_match:
            try:
                generation_result = json.loads(gen_json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse generation JSON: {e}")
                print(f"[DEBUG] Raw generation content: {gen_content[:500]}...")
                raise Exception(f"Generation JSON parsing failed: {e}")
        else:
            try:
                generation_result = json.loads(gen_content.strip())
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse generation content as JSON: {e}")
                print(f"[DEBUG] Raw generation content: {gen_content[:500]}...")
                raise Exception(f"Generation content parsing failed: {e}")

        initial_signal = generation_result.get('final_signal', 0)
        initial_confidence = generation_result.get('Confidence', 0)
        explanation = generation_result.get('explanation', '')

        print(f"[LLM-1] Initial result: Signal={initial_signal}, Confidence={initial_confidence}")

        # ===== STEP 2: SIGNAL VALIDATION =====
        print(f"[LLM-2] Validating signal with {validation_config['model']}...")

        # Build validation prompt
        validation_prompt = validation_config['prompt_template'].format(
            signal=initial_signal,
            explanation=explanation
        )

        # Try configured validation model first, then fall back to sensible defaults if quota / 429 occurs
        val_primary = validation_config.get('model')
        val_fallbacks = validation_config.get('fallback_models', [used_gen_model, "gemini-3-flash-preview", "gemini-2.5-flash"])
        val_models_to_try = [m for m in ([val_primary] + val_fallbacks) if m]

        val_response = None
        last_val_exc = None
        used_val_model = None

        for model_name in val_models_to_try:
            print(f"[LLM-2] Trying model '{model_name}' for validation...")

            def create_val_model(api_key, model_name=model_name):
                # Lazy import to avoid heavy transformer/torch dependencies during module import
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=validation_config.get('temperature', 0.1),
                    api_key=api_key
                )

            def call_val_model(api_key, model_name=model_name):
                model = create_val_model(api_key, model_name=model_name)
                return model.invoke([HumanMessage(content=[{"type": "text", "text": validation_prompt}])])

            try:
                val_response = try_api_call_with_fallback(call_val_model)
                used_val_model = model_name
                print(f"[LLM-2] Model '{model_name}' succeeded for validation")
                break
            except Exception as e:
                last_val_exc = e
                msg = str(e)
                if 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower() or '429' in msg:
                    print(f"[LLM-2] Model '{model_name}' resource/quota issue, trying next model...")
                    continue
                else:
                    # Non-quota errors should be surfaced
                    raise

        if val_response is None:
            raise Exception(f"Validation failed for all tried models. Last error: {last_val_exc}")

        # Extract validation content
        val_content = val_response.content if hasattr(val_response, 'content') else str(val_response)
        if isinstance(val_content, list):
            val_content = '\n'.join([block.get('text', '') if isinstance(block, dict) else str(block) for block in val_content])

        # Extract JSON from validation
        validation_success = False
        validation_result = None

        val_json_match = re.search(r'```json\s*(.*?)\s*```', val_content, re.DOTALL)
        if val_json_match:
            try:
                validation_result = json.loads(val_json_match.group(1))
                validation_success = True
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse validation JSON: {e}")
                print(f"[DEBUG] Raw validation content: {val_content[:500]}...")
        else:
            try:
                validation_result = json.loads(val_content.strip())
                validation_success = True
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse validation content as JSON: {e}")
                print(f"[DEBUG] Raw validation content: {val_content[:500]}...")

        if validation_success and validation_result:
            final_signal = validation_result.get('final_signal', initial_signal)
            final_confidence = validation_result.get('Confidence', initial_confidence)
            validation_reasoning = validation_result.get('explanation', '')
            print(f"[LLM-2] Final result: Signal={final_signal}, Confidence={final_confidence}")
        else:
            # Fallback to generation result if validation parsing failed
            print(f"[WARN] Validation parsing failed, using generation result as final signal")
            final_signal = initial_signal
            final_confidence = initial_confidence
            validation_reasoning = "Validation parsing failed - using generation result"
            print(f"[LLM-2] Fallback result: Signal={final_signal}, Confidence={final_confidence}")

        # Cleanup PDF
        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
                print(f"[CLEANUP] Deleted PDF: {pdf_filename}")
            except:
                pass

        # Return final result (validated or fallback)
        return {
            "final_signal": final_signal,
            "Confidence": final_confidence,
            "explanation": f"{explanation}\n\n[VALIDATION]: {validation_reasoning}"
        }

    except Exception as e:
        print(f"[ERROR] Signal generation/validation failed: {e}")

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
