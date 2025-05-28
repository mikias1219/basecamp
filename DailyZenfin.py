import streamlit as st
import requests
import json
import logging
import http.server
import socketserver
import urllib.parse
import webbrowser
import time
from datetime import datetime, timedelta, timezone, time
import uuid
import os
from typing import Dict, Optional, List
import random
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import textwrap
import re
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# Configuration
CLIENT_ID = "a48031889838a6c03a2ad322803be799a734cd15"
CLIENT_SECRET = "b1b01192191e46b51f699d444e01b7fa9ba1cf94"
REDIRECT_URI = "http://localhost:8000/oauth/callback"
BASE_URL = "https://3.basecampapi.com"
TOKEN_FILE = lambda session_id: f"access_token_{session_id}.json"
USED_QUOTES_FILE = "used_quotes.json"
REQUEST_TIMEOUT = 10
TOKEN_EXPIRY = timedelta(days=1)
PEXELS_API_KEY = "Zum5sloqdAGsnMHFm4ICOmEDAxZ4O2tujTCResgQWMug7iGQ7b2DFkbh"
USER_AGENT = "MotivationalPoster (yabsrafekadu28@gmail.com)"
QUOTABLE_API_URL = "https://api.quotable.io/random?tags=famous-quotes"
EAT_TZ = timezone(timedelta(hours=3))  # EAT is UTC+3
CONFIG_FILE = "scheduler_config.json"

# Fallback quotes to avoid repetition
FALLBACK_QUOTES = [
    {"quote": "The only way to do great work is to love what you do.", "author": "Steve Jobs"},
    {"quote": "Success is not final, failure is not fatal.", "author": "Winston Churchill"},
    {"quote": "Do what you can, with what you have, where you are.", "author": "Theodore Roosevelt"},
    {"quote": "Believe you can and you're halfway there.", "author": "Theodore Roosevelt"},
    {"quote": "The future belongs to those who believe in the beauty of their dreams.", "author": "Eleanor Roosevelt"}
]

# Logging setup
try:
    logging.basicConfig(filename="motivational_poster.log", level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
except Exception as e:
    print(f"Failed to initialize logging: {e}")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")

# Config persistence
def save_scheduler_config(account_id: int, project_id: int, message_board_id: int, access_token: str, start_time: str, end_time: str, days: List[str]):
    try:
        config = {
            "account_id": account_id,
            "project_id": str(project_id),
            "message_board_id": str(message_board_id),
            "access_token": access_token,
            "start_time": start_time,  # e.g., "23:00"
            "end_time": end_time,      # e.g., "00:00"
            "days": days               # e.g., ["mon", "tue", "wed", "thu", "fri"]
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f)
        logging.debug(f"Saved scheduler config: {config}")
    except Exception as e:
        logging.error(f"Failed to save scheduler config: {str(e)}")

def load_scheduler_config() -> Optional[Dict]:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            logging.debug(f"Loaded scheduler config: {config}")
            return config
        return None
    except Exception as e:
        logging.error(f"Failed to load scheduler config: {e}")
        return None

# Quote tracking
def load_used_quotes() -> List[Dict]:
    try:
        if os.path.exists(USED_QUOTES_FILE):
            with open(USED_QUOTES_FILE, "r", encoding="utf-8") as f:
                quotes = json.load(f)
                return quotes if isinstance(quotes, list) else []
        return []
    except Exception as e:
        logging.error(f"Failed to load used quotes: {e}")
        return []

def save_used_quote(quote: str, author: str):
    try:
        used_quotes = load_used_quotes()
        used_quotes.append({"quote": quote, "author": author})
        with open(USED_QUOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(used_quotes, f, indent=2)
        logging.info(f"Saved quote to used_quotes: {quote} - {author}")
    except Exception as e:
        logging.error(f"Failed to save used quote: {e}")

# Utility Functions
def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    port = start_port
    for _ in range(max_attempts):
        try:
            with socketserver.TCPServer(("localhost", port), None):
                return port
        except OSError:
            port += 1
    logging.error("No available ports found")
    raise RuntimeError("No available ports found")

def save_access_token(access_token: str, expiry: datetime, session_id: str):
    try:
        data = {"access_token": access_token, "expiry": expiry.isoformat()}
        with open(TOKEN_FILE(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
        logging.info("Access token saved successfully")
    except Exception as e:
        logging.error(f"Failed to save access token: {e}")
        st.error(f"Failed to save access token: {e}")

def load_access_token(session_id: str) -> Optional[Dict]:
    try:
        with open(TOKEN_FILE(session_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        expiry = datetime.fromisoformat(data.get("expiry").replace('Z', '+00:00'))
        if datetime.now(timezone.utc) < expiry:
            return {"access_token": data.get("access_token"), "expiry": expiry}
        else:
            logging.info("Access token expired")
            os.remove(TOKEN_FILE(session_id))
            return None
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logging.debug(f"Token file not found or invalid: {e}")
        return None

def get_access_token(session_id: str) -> Optional[str]:
    token_data = load_access_token(session_id)
    if token_data and token_data.get("access_token"):
        logging.debug("Using existing access token")
        return token_data["access_token"]
    port = find_available_port()
    AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    class OAuthHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path.startswith('/oauth/callback'):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    code = params.get('code', [None])[0]
                    if code:
                        token_response = requests.post(
                            "https://launchpad.37signals.com/authorization/token.json",
                            data={
                                "type": "web_server",
                                "client_id": CLIENT_ID,
                                "client_secret": CLIENT_SECRET,
                                "redirect_uri": REDIRECT_URI,
                                "code": code
                            },
                            timeout=REQUEST_TIMEOUT
                        )
                        if token_response.ok:
                            token_data = token_response.json()
                            access_token = token_data.get("access_token")
                            if access_token:
                                expiry = datetime.now(timezone.utc) + TOKEN_EXPIRY
                                save_access_token(access_token, expiry, session_id)
                                st.session_state[f\'access_token_{st.session_state.session_id}\'] = access_token
                                self.respond_with("Success! You can close this tab.")
                            else:
                                error_msg = token_data.get("error", "No access token")
                                logging.error(f"Token exchange failed: {error_msg}")
                                self.respond_with(f"Token exchange failed: {error_msg}")
                        else:
                            error_msg = token_response.text
                            logging.error(f"Token exchange failed: {error_msg}")
                            self.respond_with(f"Token exchange failed: {error_msg}")
                    else:
                        error_msg = params.get('error', ['No code received'])[0]
                        logging.error(f"OAuth callback error: {error_msg}")
                        self.respond_with(f"Authentication failed: {error_msg}")
                else:
                    logging.error(f"Invalid callback URL: {self.path}")
                    self.respond_with("Invalid callback URL")
            except Exception as e:
                logging.error(f"OAuth handler error: {e}\n{traceback.format_exc()}")
                self.respond_with(f"Authentication error: {e}")
        def respond_with(self, message):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())
    try:
        st.info("Opening browser for Basecamp authorization...")
        logging.info("Initiating OAuth flow")
        webbrowser.open(AUTH_URL)
        with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
            httpd.timeout = 120
            httpd.handle_request()
    except Exception as e:
        logging.error(f"OAuth flow failed: {e}\n{traceback.format_exc()}")
        st.error(f"Authentication failed: {e}. Check your network and try again.")
        return None
    return st.session_state.get("access_token")

def get_account_info(access_token: str) -> Optional[int]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
    try:
        response = requests.get("https://launchpad.37signals.com/authorization.json",
                               headers=headers, timeout=REQUEST_TIMEOUT)
        if response.ok:
            data = response.json()
            accounts = data.get("accounts", [])
            if accounts:
                return accounts[0]["id"]
            logging.error("No accounts found in authorization response")
            st.error("No Basecamp accounts found. Ensure your account is active.")
            return None
        logging.error(f"Failed to get account info: {response.status_code} - {response.text}")
        st.error(f"Failed to get account info: {response.status_code}")
        return None
    except Exception as e:
        logging.error(f"Error fetching account info: {e}\n{traceback.format_exc()}")
        st.error(f"Error fetching account info: {e}")
        return None

def get_projects(account_id: int, access_token: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
    url = f"{BASE_URL}/{account_id}/projects.json"
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.ok:
            projects = response.json()
            project_list = [{
                "name": project['name'],
                "id": project['id'],
                "message_board_id": next((dock["id"] for dock in project.get("dock", [])
                                         if dock["name"] == "message_board" and dock["enabled"]), None)
            } for project in projects]
            logging.info(f"Fetched {len(project_list)} projects for account_id {account_id}")
            return project_list
    except Exception as e:
        logging.error(f"Error fetching projects: {e}\n{traceback.format_exc()}")
        st.error(f"Error fetching projects: {e}")
        return []

def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        file_size = os.path.getsize(image_path)
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            logging.warning(f"Image too large: {file_size} bytes")
            return False
        if file_size == 0:
            logging.error(f"Image file is empty: {image_path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Invalid image file {image_path}: {e}")
        return False

def get_random_quote() -> Dict[str, str]:
    used_quotes = load_used_quotes()
    used_quote_set = {(q["quote"], q["author"]) for q in used_quotes}
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.get(QUOTABLE_API_URL, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if response.ok:
                data = response.json()
                quote = data.get("content")
                author = data.get("author")
                if quote and author and (quote, author) not in used_quote_set:
                    logging.info(f"Fetched unique famous quote: {quote} - {author}")
                    return {"quote": quote, "author": author}
                logging.debug(f"Quote already used or invalid: {quote} - {author}")
            else:
                logging.error(f"Failed to fetch quote: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Error fetching quote: {e}\n{traceback.format_exc()}")
        attempt += 1
    # Fallback to unique quote from FALLBACK_QUOTES
    available_fallbacks = [q for q in FALLBACK_QUOTES if (q["quote"], q["author"]) not in used_quote_set]
    if available_fallbacks:
        quote_data = random.choice(available_fallbacks)
        logging.info(f"Using fallback quote: {quote_data['quote']} - {quote_data['author']}")
        return quote_data
    else:
        # If all fallbacks used, reset used_quotes (rare case)
        logging.warning("All fallback quotes used. Resetting used quotes.")
        with open(USED_QUOTES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        quote_data = random.choice(FALLBACK_QUOTES)
        logging.info(f"Reset used quotes, using: {quote_data['quote']} - {quote_data['author']}")
        return quote_data

def get_random_photo_with_quote() -> Dict:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    # Fetch quote
    quote_data = get_random_quote()
    quote = quote_data["quote"]
    author = quote_data["author"]
    # Fetch Pexels image
    queries = ["river", "mountain", "waterfall", "forest", "lake"]
    query = random.choice(queries)
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=80&orientation=landscape"
    headers = {"Authorization": PEXELS_API_KEY}
    try:
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.ok:
            data = response.json()
            photos = data.get("photos", [])
            nature_photos = [
                photo for photo in photos
                if photo.get("alt") and not any(keyword in photo["alt"].lower() for keyword in ["person", "people", "human", "crowd", "portrait"])
            ]
            if nature_photos:
                photo = random.choice(nature_photos)
                image_url = photo["src"]["large"]
                image_description = photo.get("alt", f"{query.capitalize()} Image")
                image_response = session.get(image_url, timeout=REQUEST_TIMEOUT)
                if image_response.ok:
                    # Process image
                    image = Image.open(io.BytesIO(image_response.content)).convert("RGBA")
                    image = image.resize((800, 400), Image.LANCZOS)
                    # Add semi-transparent overlay
                    overlay = Image.new("RGBA", image.size, (0, 0, 0, 128))
                    image = Image.alpha_composite(image, overlay)
                    # Add quote text
                    draw = ImageDraw.Draw(image)
                    try:
                        font = ImageFont.truetype("Roboto-Bold.ttf", 30)
                        author_font = ImageFont.truetype("Roboto-Bold.ttf", 24)
                    except Exception as e:
                        logging.warning(f"Roboto font not found, falling back to default: {e}")
                        font = ImageFont.load_default()
                        author_font = ImageFont.load_default()
                    # Wrap quote text
                    max_width = 700
                    wrapped_quote = textwrap.wrap(quote, width=25)
                    line_height = 40
                    total_text_height = len(wrapped_quote) * line_height + 30
                    y = (400 - total_text_height) // 2
                    # Draw quote with shadow
                    for line in wrapped_quote:
                        text_width = (draw.textlength(line, font=font) if hasattr(draw, 'textlength') else draw.textsize(line, font=font)[0])
                        x = (800 - text_width) // 2
                        draw.text((x + 2, y + 2), line, fill=(0, 0, 0, 200), font=font)
                        draw.text((x, y), line, fill="white", font=font)
                        y += line_height
                    # Draw author with shadow
                    author_text = f"- {author}"
                    author_width = (draw.textlength(author_text, font=author_font) if hasattr(draw, 'textlength') else draw.textsize(author_text, font=author_font)[0])
                    x_author = (800 - author_width) // 2
                    draw.text((x_author + 2, y + 12), author_text, fill=(0, 0, 0, 200), font=author_font)
                    draw.text((x_author, y + 10), author_text, fill="white", font=author_font)
                    # Save image
                    temp_image_path = f"temp_quote_image_{uuid.uuid4().hex}.png"
                    image.save(temp_image_path, "PNG", optimize=True, quality=85)
                    if not validate_image(temp_image_path):
                        logging.error(f"Invalid or oversized image: {temp_image_path}")
                        os.remove(temp_image_path)
                        raise ValueError("Invalid image file")
                    # Encode for preview
                    with open(temp_image_path, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode()
                    logging.info(f"Generated image with quote: {image_url}")
                    return {
                        "url": temp_image_path,
                        "base64": image_base64,
                        "description": image_description,
                        "quote": quote,
                        "author": author,
                        "date": datetime.now(EAT_TZ).date().isoformat()
                    }
            logging.warning(f"No nature photos without people found for query: {query}")
            st.warning(f"No suitable nature photos found for query: {query}")
        logging.error(f"Failed to fetch image from Pexels: {response.status_code} - {response.text}")
        if response.status_code == 429:
            st.error("Pexels API rate limit exceeded. Try again later or check your API key.")
        else:
            st.error(f"Failed to fetch image from Pexels: {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching image from Pexels: {e}\n{traceback.format_exc()}")
        st.error(f"Error fetching image from Pexels: {e}")
    # Fallback
    return {
        "url": "https://via.placeholder.com/800x400?text=Famous+Quote",
        "base64": None,
        "description": "Famous Quote Image",
        "quote": quote,
        "author": author,
        "date": datetime.now(EAT_TZ).date().isoformat()
    }

def refresh_daily_image():
    st.session_state.daily_quote_image = get_random_photo_with_quote()
    logging.info(f"Refreshed daily quote image: {st.session_state.daily_quote_image['url']}")
    st.session_state.last_image_refresh = datetime.now(EAT_TZ)

def get_daily_quote_image() -> Dict:
    if 'daily_quote_image' not in st.session_state:
        st.session_state.daily_quote_image = get_random_photo_with_quote()
        st.session_state.last_image_refresh = datetime.now(EAT_TZ)
        logging.info(f"Initialized daily quote image: {st.session_state.daily_quote_image['url']}")
    cache_date = datetime.fromisoformat(st.session_state.daily_quote_image['date']).date()
    today = datetime.now(EAT_TZ).date()
    if cache_date != today:
        refresh_daily_image()
    return st.session_state.daily_quote_image

def upload_image_to_basecamp(account_id: int, access_token: str, image_path: str) -> Optional[str]:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    if not validate_image(image_path):
        logging.error(f"Skipping upload due to invalid image: {image_path}")
        return None
    try:
        file_size = os.path.getsize(image_path)
        file_name = "quote_image.png"
        url = f"{BASE_URL}/{account_id}/attachments.json?name={urllib.parse.quote(file_name)}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "image/png",
            "Content-Length": str(file_size)
        }
        logging.debug(f"Uploading image: {image_path}, size: {file_size} bytes, headers: {headers}")
        with open(image_path, "rb") as image_file:
            response = session.post(url, headers=headers, data=image_file.read(), timeout=REQUEST_TIMEOUT)
        if response.status_code == 201:
            data = response.json()
            attachable_sgid = data.get("attachable_sgid")
            if attachable_sgid:
                logging.info(f"Image uploaded to Basecamp, attachable_sgid: {attachable_sgid}")
                return attachable_sgid
            logging.error("No attachable_sgid in response")
            st.error("Image uploaded but no attachable_sgid received")
            return None
        logging.error(f"Failed to upload image to Basecamp: {response.status_code} - {response.text[:500]}")
        st.error(f"Failed to upload image to Basecamp: {response.status_code} - {response.text[:200]}")
        return None
    except Exception as e:
        logging.error(f"Error uploading image to Basecamp: {e}\n{traceback.format_exc()}")
        st.error(f"Error uploading image to Basecamp: {e}")
        return None
    finally:
        if image_path.startswith("temp_quote_image_") and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logging.debug(f"Cleaned up temporary image: {image_path}")
            except Exception as e:
                logging.error(f"Failed to clean up temporary image: {image_path}: {e}")

def parse_mentions(mentions: str) -> Optional[Dict]:
    try:
        pattern = r"Selam Team: Daily Inspiration - (.+) - (.+)"
        match = re.match(pattern, mentions)
        if match:
            return {"quote": match.group(1), "author": match.group(2)}
        logging.warning(f"Could not parse quote and author from mentions: {mentions}")
        return None
    except Exception as e:
        logging.error(f"Error parsing mentions: {e}\n{traceback.format_exc()}")
        return None

def post_message(account_id: int, project_id: int, message_board_id: int, access_token: str, image_url: Optional[str] = None, mentions: Optional[str] = None, quote: Optional[str] = None, author: Optional[str] = None) -> bool:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT, "Content-Type": "application/json"}
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/message_boards/{message_board_id}/messages.json"
    temp_image_path = None
    try:
        if image_url and mentions and quote and author:
            final_mentions = mentions
            final_image_url = image_url
            final_quote = quote
            final_author = author
        else:
            image_data = get_daily_quote_image()
            final_mentions = f"Selam Team: Daily Inspiration - {image_data['quote']} - {image_data['author']}"
            final_image_url = image_data["url"]
            final_quote = image_data["quote"]
            final_author = image_data["author"]
        attachable_sgid = None
        if final_image_url.startswith("temp_quote_image_"):
            temp_image_path = final_image_url
            attachable_sgid = upload_image_to_basecamp(account_id, access_token, final_image_url)
        if attachable_sgid:
            caption = f"{final_quote} - {final_author}"
            payload = {
                "subject": "Daily Inspiration",
                "content": f"<p>{final_mentions}</p><bc-attachment sgid=\"{attachable_sgid}\" caption=\"{caption}\"></bc-attachment>",
                "status": "active"
            }
        else:
            logging.warning("Posting message without image due to upload failure")
            payload = {
                "subject": "Daily Inspiration",
                "content": f"<p>{final_mentions}</p>",
                "status": "active"
            }
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.ok:
            save_used_quote(final_quote, final_author)
            logging.info("Message posted successfully")
            return True
        logging.error(f"Failed to post message: {response.status_code} - {response.text[:500]}")
        st.error(f"Failed to post message: {response.status_code} - {response.text[:200]}")
        return False
    except Exception as e:
        logging.error(f"Error posting message: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error posting message: {str(e)}")
        return False
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logging.debug(f"Cleaned up temporary image: {temp_image_path}")
            except Exception as e:
                logging.error(f"Failed to clean up temporary image {temp_image_path}: {e}")

def schedule_daily_post(scheduler, account_id: int, project_id: int, message_board_id: int, access_token: str, start_time: time, end_time: time, days: List[str]):
    try:
        # Clear existing jobs
        scheduler.remove_all_jobs()
        # Schedule a single post per day within the time range
        for day in days:
            # Calculate a random time within the start_time and end_time
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            if end_minutes <= start_minutes:
                end_minutes += 24 * 60  # Handle cases where end_time is past midnight
            random_minutes = random.randint(start_minutes, end_minutes)
            random_hour = random_minutes // 60
            random_minute = random_minutes % 60
            # Schedule the post
            scheduler.add_job(
                check_and_post,
                trigger=CronTrigger(
                    hour=random_hour,
                    minute=random_minute,
                    day_of_week=day,
                    timezone=EAT_TZ
                ),
                args=[False, account_id, project_id, message_board_id, access_token],
                id=f'daily_post_job_{day}',
                replace_existing=True
            )
            logging.info(f"Scheduled post for {day} at {random_hour:02d}:{random_minute:02d} EAT")
        # Schedule daily image refresh at 00:00 EAT
        scheduler.add_job(
            refresh_daily_image,
            trigger=CronTrigger(hour=0, minute=0, timezone=EAT_TZ),
            id='daily_image_refresh_job',
            replace_existing=True
        )
        logging.info("Daily scheduler configured with random posts in time range")
    except Exception as e:
        logging.error(f"Failed to schedule daily post: {e}")
        st.error(f"Failed to schedule daily post: {e}")

def check_and_post(test_mode: bool = False, account_id: Optional[int] = None, project_id: Optional[int] = None, message_board_id: Optional[int] = None, access_token: Optional[str] = None):
    if not all([account_id, project_id, message_board_id, access_token]):
        config = load_scheduler_config()
        if not config:
            logging.error("No scheduler config found. Cannot post.")
            return False
        account_id = config["account_id"]
        project_id = int(config["project_id"])
        message_board_id = int(config["message_board_id"])
        access_token = config["access_token"]
    if not all([account_id, project_id, message_board_id, access_token]):
        logging.error("Missing required parameters for posting")
        return False
    current_time = datetime.now(EAT_TZ)
    logging.debug(f"Executing check_and_post at {current_time.strftime('%Y-%m-%d %H:%M:%S')} EAT, test_mode={test_mode}")
    success = post_message(account_id, project_id, message_board_id, access_token)
    if success:
        logging.info(f"{'Test' if test_mode else 'Daily'} post successful at {current_time.strftime('%H:%M:%S')}")
    else:
        logging.error(f"{'Test' if test_mode else 'Daily'} post failed at {current_time.strftime('%H:%M:%S')}")
    return success

# Streamlit App
def main():
    try:
        st.set_page_config(page_title="Basecamp Inspirational Quote Poster", layout="wide")
        st.markdown("""
        <style>
        .navbar {
            background-color: #1a73e8;
            padding: 15px;
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        .preview-box {
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: white;
        }
        </style>
        <div class="navbar">Basecamp Inspirational Quote Poster</div>
        """, unsafe_allow_html=True)
        st.write("Automate daily inspirational quote images to your Basecamp project message board.")
        st.markdown("Quotes provided by <a href='https://quotable.io' target='_blank'>Quotable API</a>, images by <a href='https://www.pexels.com' target='_blank'>Pexels API</a>", unsafe_allow_html=True)

        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = uuid.uuid4().hex
            logging.info(f"Initialized session_id: {st.session_state.session_id}")
        if 'access_token' not in st.session_state:
            st.session_state[f\'access_token_{st.session_state.session_id}\'] = None
        if 'account_id' not in st.session_state:
            st.session_state.account_id = None
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = None
        if 'preview_data' not in st.session_state:
            st.session_state.preview_data = None
        if 'scheduler_active' not in st.session_state:
            st.session_state.scheduler_active = False
        if 'scheduler' not in st.session_state:
            st.session_state.scheduler = BackgroundScheduler(timezone=EAT_TZ)
            try:
                st.session_state.scheduler.start()
                logging.info("APScheduler initialized and started")
            except Exception as e:
                logging.error(f"Failed to start APScheduler: {e}")
                st.error(f"Failed to start scheduler: {e}")

        # Sidebar
        st.sidebar.header("Settings")
        if st.sidebar.button("Authenticate with Basecamp"):
            with st.spinner("Authenticating..."):
                access_token = get_access_token(st.session_state.session_id)
                if access_token:
                    st.session_state[f\'access_token_{st.session_state.session_id}\'] = access_token
                    account_id = get_account_info(access_token)
                    if account_id:
                        st.session_state.account_id = account_id
                        st.success("Authentication successful!")
                        logging.info("Authentication successful")
                    else:
                        st.error("Failed to fetch account ID. Ensure your Basecamp account is active.")
                        logging.error("Failed to fetch account ID")
                        if os.path.exists(TOKEN_FILE(st.session_state.session_id)):
                            os.remove(TOKEN_FILE(st.session_state.session_id))
                else:
                    st.error("Authentication failed. Check your Basecamp credentials or network.")
                    logging.error("Authentication failed")

        # Time Range and Days Selection
        st.sidebar.header("Schedule Settings")
        current_time = datetime.now(EAT_TZ)
        default_start_time = time(current_time.hour, current_time.minute)
        default_end_time = (current_time + timedelta(hours=1)).time()
        start_time = st.sidebar.time_input("Start Time (EAT)", value=default_start_time, key="start_time")
        end_time = st.sidebar.time_input("End Time (EAT)", value=default_end_time, key="end_time")
        days_options = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        days_display = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_days = st.sidebar.multiselect(
            "Select Days",
            options=days_display,
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            key="selected_days"
        )
        # Convert display names to cron-compatible day names
        selected_days_cron = [days_options[days_display.index(day)] for day in selected_days]

        # Main content
        if st.session_state[f\'access_token_{st.session_state.session_id}\'] and st.session_state.account_id:
            projects = get_projects(st.session_state.account_id, st.session_state[f\'access_token_{st.session_state.session_id}\'])
            project_names = [p["name"] for p in projects if p["message_board_id"]]
            if project_names:
                st.subheader("Select a Project")
                selected_project_name = st.selectbox("Select Project", project_names, key="project_select")
                st.session_state.selected_project = next((p for p in projects if p["name"] == selected_project_name), None)
                if st.session_state.selected_project:
                    project_id = st.session_state.selected_project["id"]
                    message_board_id = st.session_state.selected_project["message_board_id"]
                    # Save scheduler config
                    save_scheduler_config(
                        st.session_state.account_id,
                        project_id,
                        message_board_id,
                        st.session_state[f\'access_token_{st.session_state.session_id}\'],
                        start_time.strftime("%H:%M"),
                        end_time.strftime("%H:%M"),
                        selected_days_cron
                    )
                    # Preview Post
                    if st.button("Preview Post"):
                        with st.spinner("Generating preview..."):
                            image_data = get_daily_quote_image()
                            st.session_state.preview_data = {
                                "image_url": image_data["url"],
                                "image_base64": image_data["base64"],
                                "basecamp_mentions": f"Selam Team: Daily Inspiration - {image_data['quote']} - {image_data['author']}",
                                "plain_mentions": f"Selam Team: Daily Inspiration - {image_data['quote']} - {image_data['author']}",
                                "quote": image_data["quote"],
                                "author": image_data["author"]
                            }
                            st.success("Preview generated!")
                            logging.info("Preview generated")
                    # Display Preview
                    if st.session_state.preview_data:
                        st.subheader("Post Preview")
                        if st.session_state.preview_data["image_base64"]:
                            st.markdown(f"""
                            <div class="preview-box">
                                <strong>Message:</strong> {st.session_state.preview_data['plain_mentions']}<br>
                                <strong>Image:</strong><br>
                                <img src="data:image/png;base64,{st.session_state.preview_data['image_base64']}" alt="Famous Quote" style='max-width:100%;'>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="preview-box">
                                <strong>Message:</strong> {st.session_state.preview_data['plain_mentions']}<br>
                                <strong>Image:</strong><br>
                                <img src="{st.session_state.preview_data['image_url']}" alt="Famous Quote" style='max-width:100%;'>
                            </div>
                            """, unsafe_allow_html=True)
                    # Daily Scheduler
                    if st.button("Start Daily Scheduler"):
                        with st.spinner("Starting daily scheduler..."):
                            config_data = load_scheduler_config()
                            if config_data and selected_days_cron:
                                st.session_state.scheduler_active = True
                                # Schedule daily post
                                schedule_daily_post(
                                    st.session_state.scheduler,
                                    st.session_state.account_id,
                                    project_id,
                                    message_board_id,
                                    st.session_state[f\'access_token_{st.session_state.session_id}\'],
                                    start_time,
                                    end_time,
                                    selected_days_cron
                                )
                                # Schedule immediate test run (1 minute from now)
                                test_run_time = datetime.now(EAT_TZ) + timedelta(minutes=1)
                                st.session_state.scheduler.add_job(
                                    check_and_post,
                                    trigger=DateTrigger(run_date=test_run_time, timezone=EAT_TZ),
                                    args=[False, st.session_state.account_id, project_id, message_board_id, st.session_state[f\'access_token_{st.session_state.session_id}\']],
                                    id='test_daily_run'
                                )
                                st.info(f"Daily scheduler started. A test post will run at {test_run_time.strftime('%H:%M:%S')} EAT, then daily posts at a random time between {start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')} EAT on {', '.join(selected_days)}. Image refresh at 00:00 EAT.")
                                logging.info(f"Daily scheduler started with time range {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} on {selected_days_cron}")
                            else:
                                st.error("No valid configuration or days selected. Select a project and days, then try again.")
                                logging.error("No valid config or days for daily scheduler")
                    # Stop Scheduler
                    if st.button("Stop Scheduler"):
                        st.session_state.scheduler_active = False
                        st.session_state.scheduler.remove_all_jobs()
                        st.success("Scheduler stopped.")
                        logging.info("Scheduler stopped")
            else:
                st.warning("No projects with message boards found. Ensure you have access to projects with message boards enabled.")
                logging.warning("No projects with message boards found")
        else:
            st.info("Please authenticate with Basecamp to continue.")
            logging.info("Awaiting Basecamp authentication")
    except Exception as e:
        logging.error(f"Main function error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'scheduler' in st.session_state and st.session_state.scheduler.running:
            try:
                st.session_state.scheduler.shutdown()
                logging.info("Scheduler shut down successfully")
            except Exception as e:
                logging.error(f"Failed to shut down scheduler: {e}")

if __name__ == "__main__":
    main()