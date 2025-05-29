import streamlit as st
import requests
import json
import logging
import urllib.parse
import schedule
import time
from datetime import datetime, timedelta, timezone
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
import pytz
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CLIENT_ID = os.getenv("CLIENT_ID", "a48031889838a6c03a2ad322803be799a734cd15")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "b1b01192191e46b51f699d444e01b7fa9ba1cf94")
BASE_URL = os.getenv("BASECAMP_BASE_URL", "https://3.basecampapi.com")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "Zum5sloqdAGsnMHFm4ICOmEDAxZ4O2tujTCResgQWMug7iGQ7b2DFkbh")
QUOTABLE_API_URL = os.getenv("QUOTABLE_API_URL", "https://api.quotable.io/random?tags=leadership|success|wisdom|development|resilience|intelligence")
ZENQUOTES_API_URL = os.getenv("ZENQUOTES_API_URL", "https://zenquotes.io/api/quotes")
USER_AGENT = "MotivationalPoster (yabsrafekadu28@gmail.com)"
REQUEST_TIMEOUT = 10
TOKEN_EXPIRY = timedelta(days=1)
EAT_TZ = pytz.timezone("Africa/Nairobi")  # EAT is UTC+3

# Dynamically determine the base URL for REDIRECT_URI
def get_base_url():
    """
    Get the base URL of the Streamlit app (local or deployed).
    """
    base_url = os.getenv("STREAMLIT_SERVER_ADDRESS", "http://localhost:8501")
    return base_url.rstrip("/")

REDIRECT_URI = f"{get_base_url()}/oauth/callback"

# Fallback quotes to avoid repetition
FALLBACK_QUOTES = [
    {"quote": "Great leaders don’t create followers; they inspire others to become leaders.", "author": "John Quincy Adams"},
    {"quote": "Resilience is not about avoiding obstacles, but about navigating through them with courage.", "author": "Sheryl Sandberg"},
    {"quote": "Success is not the absence of challenges, but the courage to push through them.", "author": "Oprah Winfrey"},
    {"quote": "Intelligence is the ability to adapt to change and learn from failure.", "author": "Stephen Hawking"},
    {"quote": "The greatest development comes from embracing challenges and learning from them.", "author": "Carol Dweck"}
]

# Logging setup
try:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
except Exception as e:
    print(f"Failed to initialize logging: {e}")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Quote tracking functions
def load_used_quotes() -> List[Dict]:
    """
    Load used quotes from session state.
    """
    try:
        quotes = st.session_state.get("used_quotes", [])
        return quotes if isinstance(quotes, list) else []
    except Exception as e:
        logging.error(f"Failed to load used quotes: {e}")
        return []

def save_used_quote(quote: str, author: str):
    """
    Save used quote to session state.
    """
    try:
        used_quotes = load_used_quotes()
        used_quotes.append({"quote": quote, "author": author})
        st.session_state.used_quotes = used_quotes
        logging.info(f"Saved quote to session state: {quote} - {author}")
    except Exception as e:
        logging.error(f"Failed to save used quote: {e}")

# Utility Functions
def retry_request(method_func, url, **kwargs):
    """Retry HTTP request with exponential backoff."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.request(method_func.__name__.split('.')[-1].upper(), url, **kwargs)
        return response
    except Exception as e:
        logging.error(f"Retry request failed for {url}: {e}")
        return None

def get_paginated_results(url: str, headers: Dict) -> List[Dict]:
    """Fetch all pages of results from a Basecamp API endpoint."""
    results = []
    current_url = url
    while current_url:
        try:
            response = retry_request(requests.get, current_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if not response or response.status_code != 200:
                logging.error(f"Failed to fetch paginated results from {current_url}: HTTP {response.status_code if response else 'No response'}")
                return results
            data = response.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
            if 'Link' in response.headers:
                links = response.headers['Link'].split(',')
                next_link = next((link for link in links if 'rel="next"' in link), None)
                if next_link:
                    current_url = next_link.split(';')[0].strip('<>')
                else:
                    current_url = None
            else:
                current_url = None
        except Exception as e:
            logging.error(f"Error fetching paginated results from {current_url}: {e}")
            return results
    return results

def save_access_token(access_token: str, expiry: datetime, session_id: str):
    """
    Save access token and expiry to session state.
    """
    try:
        st.session_state.access_token = access_token
        st.session_state.token_expiry = expiry.isoformat()
        logging.info("Access token saved to session state")
    except Exception as e:
        logging.error(f"Failed to save access token: {e}")
        st.error(f"Failed to save access token: {e}")

def load_access_token(session_id: str) -> Optional[Dict]:
    """
    Load access token and expiry from session state.
    """
    try:
        access_token = st.session_state.get("access_token")
        expiry_str = st.session_state.get("token_expiry")
        if access_token and expiry_str:
            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) < expiry:
                return {"access_token": access_token, "expiry": expiry}
            else:
                logging.info("Access token expired")
                st.session_state.pop("access_token", None)
                st.session_state.pop("token_expiry", None)
                return None
        return None
    except Exception as e:
        logging.error(f"Failed to load access token: {e}")
        return None

def get_access_token(session_id: str) -> Optional[str]:
    """
    Handle Basecamp OAuth flow for both local and deployed environments.
    Returns the access token or None if authentication fails.
    """
    # Check if access token is already in session state
    if st.session_state.get("access_token"):
        logging.debug("Using existing access token from session state")
        return st.session_state.access_token

    # Use dynamic REDIRECT_URI
    AUTH_URL = (
        f"https://launchpad.37signals.com/authorization/new"
        f"?type=web_server&client_id={CLIENT_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
    )

    # Check for OAuth callback (code or error in query parameters)
    query_params = st.query_params.to_dict()
    code = query_params.get("code")
    error = query_params.get("error")

    if error:
        logging.error(f"OAuth callback error: {error}")
        st.error(f"Authentication failed: {error}")
        return None

    if code:
        # Handle OAuth callback: exchange code for access token
        try:
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
                    logging.info("Access token obtained and stored in session state")
                    # Clear query params to prevent re-processing
                    st.query_params.clear()
                    return access_token
                else:
                    error_msg = token_data.get("error", "No access token received")
                    logging.error(f"Token exchange failed: {error_msg}")
                    st.error(f"Token exchange failed: {error_msg}")
                    return None
            else:
                error_msg = token_response.text
                logging.error(f"Token exchange failed: {token_response.status_code} - {error_msg}")
                st.error(f"Token exchange failed: {token_response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error during token exchange: {e}")
            st.error(f"Error during token exchange: {e}")
            return None

    # If no code or error, prompt user to authenticate
    st.info("Please authenticate with Basecamp to continue.")
    st.markdown(f"[Click here to authenticate with Basecamp]({AUTH_URL})")
    logging.info("Displayed authentication link to user")
    return None

def get_account_info(access_token: str) -> Optional[int]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
    try:
        response = requests.get(
            "https://launchpad.37signals.com/authorization.json",
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        if response.ok:
            data = response.json()
            accounts = data.get("accounts", [])
            if accounts:
                return accounts[0]["id"]
            logging.error("No accounts found in authorization response")
            st.error("No Basecamp accounts found. Ensure your account is active.")
            return None
        logging.error(f"Failed to get account info: {response.text}")
        st.error(f"Failed to get account info: {response.status_code}")
        return None
    except Exception as e:
        logging.error(f"Error fetching account info: {e}")
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
                "name": project["name"],
                "id": project["id"],
                "message_board_id": next((dock["id"] for dock in project.get("dock", [])
                                         if dock["name"] == "message_board" and dock["enabled"]), None)
            } for project in projects]
            logging.info(f"Fetched {len(project_list)} projects for account_id {account_id}")
            return project_list
        logging.error(f"Failed to fetch projects: {response.text}")
        st.error(f"Failed to fetch projects: {response.status_code}")
        return []
    except Exception as e:
        logging.error(f"Error fetching projects: {e}")
        st.error(f"Error fetching projects: {e}")
        return []

def get_project_people(account_id: int, project_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/projects/{project_id}/people.json"
    logging.debug(f"Fetching people from: {url} with account_id={account_id}, project_id={project_id}")
    try:
        response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
        if not response or response.status_code != 200:
            error_msg = f"Failed to fetch people for project {project_id}: HTTP {response.status_code if response else 'No response'}"
            logging.error(error_msg)
            if response:
                logging.debug(f"Response details: {response.text}")
            if response and response.status_code == 403:
                st.error("Insufficient permissions to view project members. Ensure your Basecamp account has access.")
            elif response and response.status_code == 404:
                st.error(f"Project {project_id} not found. Verify the project ID or check if it’s archived.")
            else:
                st.error(error_msg)
            return []

        people = get_paginated_results(url, headers)
        logging.info(f"Raw people API response for project {project_id}: {json.dumps(people, indent=2)}")
        valid_people = []
        for p in people:
            person_id = p.get("id")
            person_name = p.get("name")
            person_email = p.get("email_address", "N/A")
            sgid = p.get("attachable_sgid")
            logging.info(f"Processing person: ID={person_id}, Name={person_name}, Email={person_email}, SGID={sgid}, Data={p}")

            if not person_id:
                logging.warning(f"Skipping person with no ID: {p}")
                continue
            if not person_name:
                person_name = f"User_{person_id}"
                logging.warning(f"Person ID {person_id} has no name, using fallback: {person_name}")
            if not sgid:
                logging.warning(f"No attachable_sgid for person ID {person_id}, skipping for tagging")
                continue

            valid_people.append({
                "id": person_id,
                "name": person_name,
                "email_address": person_email,
                "sgid": sgid,
                "title": p.get("title", ""),
                "avatar_url": p.get("avatar_url", "https://bc3-production-assets-cdn.basecamp-static.com/default/avatar?v=1"),
                "company": p.get("company", {}).get("name", "N/A")
            })

        logging.info(f"Fetched {len(valid_people)} valid people for project {project_id}")
        logging.info(f"Valid people: {json.dumps(valid_people, indent=2)}")
        if not valid_people:
            st.warning(f"No people found for project {project_id}. Check project permissions or add people in Basecamp.")
        return valid_people
    except Exception as e:
        logging.error(f"Error fetching people for project {project_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error fetching people for project {project_id}: {str(e)}")
        return []

def format_mentions(person: Dict, is_cc: bool = False) -> str:
    """Format a person's mention in Basecamp <bc-attachment> syntax with all required details."""
    required_fields = ["id", "sgid", "name"]
    if not person or any(not person.get(field) for field in required_fields):
        logging.warning(f"Invalid person data for mention: {person}")
        st.warning(f"Person {person.get('name', 'Unknown')} cannot be tagged due to missing data.")
        return ""
    name = person["name"]
    sgid = person["sgid"]
    person_id = person["id"]
    title = person.get("title", "")
    company = person.get("company", "N/A")
    avatar_url = person.get("avatar_url", "https://bc3-production-assets-cdn.basecamp-static.com/default/avatar?v=1")
    full_title = f"{name}, {title} at {company}".strip(", ")
    mention = (
        f'<bc-attachment sgid="{sgid}" content-type="application/vnd.basecamp.mention">'
        f'<figure>'
        f'<img data-avatar-for-person-id="{person_id}" alt="{name}" title="{full_title}" '
        f'class="avatar" src="{avatar_url}" width="20" height="20">'
        f'<figcaption>{name}</figcaption>'
        f'</figure>'
        f'</bc-attachment>'
    )
    return mention

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
            logging.warning(f"Image file is empty: {image_path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Invalid image file {image_path}: {e}")
        return False

def get_random_quote() -> Dict:
    used_quotes = load_used_quotes()
    used_quote_set = {(q["quote"], q["author"]) for q in used_quotes}
    max_attempts = 5
    attempt = 0

    # Try Quotable API
    while attempt < max_attempts:
        try:
            response = requests.get(QUOTABLE_API_URL, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if response.ok:
                data = response.json()
                quote = data.get("content")
                author = data.get("author")
                if quote and author and (quote, author) not in used_quote_set:
                    logging.info(f"Fetched unique famous quote from Quotable: {quote} - {author}")
                    return {"quote": quote, "author": author}
                logging.debug(f"Quote already used or invalid from Quotable: {quote} - {author}")
            else:
                logging.error(f"Failed to fetch quote from Quotable: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Error fetching quote from Quotable: {e}")
        attempt += 1

    # Try ZenQuotes API as fallback
    try:
        response = requests.get(ZENQUOTES_API_URL, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        if response.ok:
            data = response.json()
            if data and isinstance(data, list):
                available_quotes = [
                    {"quote": q["q"], "author": q["a"]}
                    for q in data
                    if q.get("q") and q.get("a") and (q["q"], q["a"]) not in used_quote_set
                ]
                if available_quotes:
                    quote_data = random.choice(available_quotes)
                    logging.info(f"Fetched unique quote from ZenQuotes: {quote_data['quote']} - {quote_data['author']}")
                    return quote_data
                logging.debug("No unique quotes available from ZenQuotes")
            else:
                logging.error("Invalid response from ZenQuotes API")
        else:
            logging.error(f"Failed to fetch quote from ZenQuotes: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error fetching quote from ZenQuotes: {e}")

    # Fallback to local FALLBACK_QUOTES
    available_fallbacks = [q for q in FALLBACK_QUOTES if (q["quote"], q["author"]) not in used_quote_set]
    if available_fallbacks:
        quote_data = random.choice(available_fallbacks)
        logging.info(f"Using fallback quote: {quote_data['quote']} - {quote_data['author']}")
        return quote_data
    else:
        # Reset used_quotes if all fallbacks are used
        logging.warning("All fallback quotes used. Resetting used quotes.")
        st.session_state.used_quotes = []
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
                        text_width = draw.textlength(line, font=font)
                        x = (800 - text_width) // 2
                        draw.text((x + 2, y + 2), line, fill=(0, 0, 0, 200), font=font)
                        draw.text((x, y), line, fill="white", font=font)
                        y += line_height
                    # Draw author with shadow
                    author_text = f"- {author}"
                    author_width = draw.textlength(author_text, font=author_font)
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

def get_daily_quote_image(force_refresh: bool = False) -> Dict:
    if 'daily_quote_image' not in st.session_state or force_refresh:
        st.session_state.daily_quote_image = get_random_photo_with_quote()
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
        logging.error(f"Error uploading image to Basecamp: {e}")
        st.error(f"Error uploading image to Basecamp: {e}")
        return None
    finally:
        if image_path.startswith("temp_quote_image_") and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logging.debug(f"Cleaned up temporary image: {image_path}")
            except Exception as e:
                logging.error(f"Failed to clean up temporary image {image_path}: {e}")

def parse_mentions(mentions: str) -> Optional[Dict]:
    try:
        pattern = r"Selam Team, - (.+) - (.+)"
        match = re.match(pattern, mentions)
        if match:
            return {"quote": match.group(1), "author": match.group(2)}
        logging.warning(f"Could not parse quote and author from mentions: {mentions}")
        return None
    except Exception as e:
        logging.error(f"Error parsing mentions: {e}")
        return None

def post_message(account_id: int, project_id: int, message_board_id: int, access_token: str, image_url: Optional[str] = None, mentions: Optional[str] = None, quote: Optional[str] = None, author: Optional[str] = None, test_mode: bool = False) -> bool:
    logging.debug("Attempting to post message to Basecamp")
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
            # Force refresh in test mode to get new quote/image each time
            image_data = get_random_photo_with_quote() if test_mode else get_daily_quote_image()
            # Build mentions with project people
            project_people = st.session_state.get('project_people', [])
            if project_people:
                mention_tags = [format_mentions(person) for person in project_people]
                mention_tags = [tag for tag in mention_tags if tag]  # Filter out empty tags
                final_mentions = f"Selam {' '.join(mention_tags)}," if mention_tags else "Selam Team,"
            else:
                final_mentions = "Selam Team,"
                logging.warning("No project people found for mentions")
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
        logging.error(f"Failed to post message: {response.status_code} - {response.text[:200]}")
        st.error(f"Failed to post message: {response.status_code}")
        return False
    except Exception as e:
        logging.error(f"Error posting message: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error posting message: {str(e)}")
        return False
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logging.debug(f"Removed temporary image: {temp_image_path}")
            except Exception as e:
                logging.error(f"Failed to remove temporary image {temp_image_path}: {e}")

def schedule_daily_post(account_id: int, project_id: int, message_board_id: int, access_token: str, schedule_time: str, test_mode: bool = False):
    schedule.clear()  # Clear all scheduled jobs to prevent duplicates
    def job():
        # Check if today is Monday–Friday (0=Monday, 4=Friday)
        if not test_mode and datetime.now(EAT_TZ).weekday() >= 5:
            logging.info(f"Skipping post on {datetime.now(EAT_TZ).strftime('%A')} (weekend)")
            return
        post_message(account_id, project_id, message_board_id, access_token, test_mode=test_mode)
    if test_mode:
        schedule.every(1).minutes.do(job)
        logging.info("Scheduled test post every minute")
        st.info("Scheduled test post every minute")
    else:
        try:
            # Validate time format
            datetime.strptime(schedule_time, "%H:%M")
            schedule.every().day.at(schedule_time).do(job)
            logging.info(f"Scheduled daily post at {schedule_time} EAT (Monday–Friday)")
            st.info(f"Scheduled daily post at {schedule_time} EAT (Monday–Friday)")
        except ValueError:
            logging.error(f"Invalid time format for scheduling: {schedule_time}")
            st.error("Invalid time format. Please use HH:MM (24-hour format).")
            return
    while True:
        logging.debug("Checking for pending scheduled tasks")
        schedule.run_pending()
        time.sleep(60)

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
            background-color: #ffffff;
        }
        </style>
        <div class="navbar">Basecamp Inspirational Quote Poster</div>
        """, unsafe_allow_html=True)
        st.write("Automate daily famous quote images to your Basecamp project message board.")
        st.markdown("Quotes provided by <a href='https://quotable.io/' target='_blank'>Quotable API</a> and <a href='https://zenquotes.io/' target='_blank'>ZenQuotes API</a>, images by <a href='https://www.pexels.com/' target='_blank'>Pexels API</a>", unsafe_allow_html=True)
        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = uuid.uuid4().hex
            logging.info(f"Initialized session_id: {st.session_state.session_id}")
        if 'access_token' not in st.session_state:
            st.session_state.access_token = None
        if 'token_expiry' not in st.session_state:
            st.session_state.token_expiry = None
        if 'account_id' not in st.session_state:
            st.session_state.account_id = None
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = None
        if 'project_people' not in st.session_state:
            st.session_state.project_people = []
        if 'preview_data' not in st.session_state:
            st.session_state.preview_data = None
        if 'schedule_time' not in st.session_state:
            st.session_state.schedule_time = "06:00"
        if 'scheduler_running' not in st.session_state:
            st.session_state.scheduler_running = False
        if 'used_quotes' not in st.session_state:
            st.session_state.used_quotes = []
        # Sidebar
        st.sidebar.header("Settings")
        if st.sidebar.button("Authenticate with Basecamp"):
            with st.spinner("Authenticating..."):
                access_token = get_access_token(st.session_state.session_id)
                if access_token:
                    st.session_state.access_token = access_token
                    account_id = get_account_info(access_token)
                    if account_id:
                        st.session_state.account_id = account_id
                        st.success("Authentication successful!")
                        logging.info("Authentication successful")
                    else:
                        st.error("Failed to fetch account ID")
                        logging.error("Failed to fetch account ID")
                        st.session_state.pop("access_token", None)
                        st.session_state.pop("token_expiry", None)
                else:
                    st.error("Authentication failed")
                    logging.error("Authentication failed")
        # Main content
        if st.session_state.access_token and st.session_state.account_id:
            projects = get_projects(st.session_state.account_id, st.session_state.access_token)
            project_names = [p["name"] for p in projects if p["message_board_id"]]
            if project_names:
                st.subheader("Select a Project")
                selected_project_name = st.selectbox("Select Project", project_names, key="project_select")
                selected_project = next((p for p in projects if p["name"] == selected_project_name), None)
                # Update selected project and fetch people if changed
                if selected_project != st.session_state.selected_project:
                    st.session_state.selected_project = selected_project
                    if selected_project:
                        project_id = selected_project["id"]
                        logging.debug(f"Selected project: {selected_project_name}, ID: {project_id}")
                        st.session_state.project_people = get_project_people(
                            st.session_state.account_id, project_id, st.session_state.access_token
                        )
                        if st.session_state.project_people:
                            logging.info(f"Successfully fetched {len(st.session_state.project_people)} people for project {project_id}")
                        else:
                            logging.warning(f"No people fetched for project {project_id}")
                if st.session_state.selected_project:
                    project_id = st.session_state.selected_project["id"]
                    message_board_id = st.session_state.selected_project["message_board_id"]
                    # Schedule Time Input
                    st.subheader("Schedule Settings")
                    st.session_state.schedule_time = st.text_input("Daily Post Time (HH:MM, 24-hour format)", 
                                                                 value=st.session_state.schedule_time,
                                                                 key="schedule_time_input")
                    # Display next scheduled post time
                    try:
                        schedule_time = datetime.strptime(st.session_state.schedule_time, "%H:%M").time()
                        today = datetime.now(EAT_TZ).date()
                        next_run = datetime.combine(today, schedule_time, tzinfo=EAT_TZ)
                        # Find next weekday (Monday–Friday)
                        while next_run < datetime.now(EAT_TZ) or next_run.weekday() >= 5:
                            next_run += timedelta(days=1)
                        st.info(f"Next scheduled post: {next_run.strftime('%Y-%m-%d %H:%M')} EAT")
                    except ValueError:
                        st.error("Invalid time format. Using default 06:00.")
                        st.session_state.schedule_time = "06:00"
                    # Preview Post
                    if st.button("Preview Post"):
                        with st.spinner("Generating preview..."):
                            image_data = get_daily_quote_image()
                            # Build mentions for preview
                            project_people = st.session_state.get('project_people', [])
                            if project_people:
                                mention_tags = [format_mentions(person) for person in project_people]
                                mention_tags = [tag for tag in mention_tags if tag]
                                plain_mentions = f"Selam {', '.join(['@' + person['name'] for person in project_people])},"
                                basecamp_mentions = f"Selam {' '.join(mention_tags)}," if mention_tags else "Selam Team,"
                            else:
                                plain_mentions = "Selam Team,"
                                basecamp_mentions = "Selam Team,"
                                st.warning("No project people found for mentions. Using 'Selam Team,'")
                            st.session_state.preview_data = {
                                "image_url": image_data["url"],
                                "image_base64": image_data["base64"],
                                "basecamp_mentions": basecamp_mentions,
                                "plain_mentions": plain_mentions,
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
                    # Test Post
                    if st.button("Test Post"):
                        with st.spinner("Posting test message..."):
                            if st.session_state.preview_data:
                                success = post_message(
                                    st.session_state.account_id,
                                    project_id,
                                    message_board_id,
                                    st.session_state.access_token,
                                    image_url=st.session_state.preview_data["image_url"],
                                    mentions=st.session_state.preview_data["basecamp_mentions"],
                                    quote=st.session_state.preview_data["quote"],
                                    author=st.session_state.preview_data["author"]
                                )
                            else:
                                success = post_message(
                                    st.session_state.account_id,
                                    project_id,
                                    message_board_id,
                                    st.session_state.access_token
                                )
                            if success:
                                st.success("Test message posted successfully!")
                                logging.info("Test post successful")
                                st.session_state.preview_data = None
                            else:
                                st.error("Test post failed. Check logs for details.")
                                logging.error("Test post failed")
                    # Test Scheduler
                    if st.button("Test Scheduler (Posts every minute)"):
                        with st.spinner("Starting test scheduler..."):
                            token_data = load_access_token(st.session_state.session_id)
                            if token_data:
                                st.session_state.scheduler_running = False  # Reset
                                scheduler_thread = threading.Thread(
                                    target=schedule_daily_post,
                                    args=(st.session_state.account_id, project_id, message_board_id, 
                                          token_data["access_token"], st.session_state.schedule_time, True)
                                )
                                scheduler_thread.daemon = True
                                scheduler_thread.start()
                                st.session_state.scheduler_running = True
                                st.success("Test scheduler started. Posts every minute.")
                    # Start Daily Scheduler
                    if st.button("Start Daily Scheduler"):
                        with st.spinner("Starting daily scheduler..."):
                            token_data = load_access_token(st.session_state.session_id)
                            if token_data:
                                st.session_state.scheduler_running = False  # Reset
                                scheduler_thread = threading.Thread(
                                    target=schedule_daily_post,
                                    args=(st.session_state.account_id, project_id, message_board_id, 
                                          token_data["access_token"], st.session_state.schedule_time)
                                )
                                scheduler_thread.daemon = True
                                scheduler_thread.start()
                                st.session_state.scheduler_running = True
                                st.success(f"Daily scheduler started. Posts will occur at {st.session_state.schedule_time} EAT on Monday–Friday.")
                    # Automatic scheduler
                    if not st.session_state.scheduler_running:
                        token_data = load_access_token(st.session_state.session_id)
                        if token_data and st.session_state.schedule_time:
                            try:
                                datetime.strptime(st.session_state.schedule_time, "%H:%M")
                                st.info(f"Scheduler started automatically. Posts will occur daily at {st.session_state.schedule_time} EAT on Monday–Friday.")
                                logging.info(f"Starting daily scheduler automatically at {st.session_state.schedule_time}")
                                scheduler_thread = threading.Thread(
                                    target=schedule_daily_post,
                                    args=(st.session_state.account_id, project_id, message_board_id, 
                                          token_data["access_token"], st.session_state.schedule_time)
                                )
                                scheduler_thread.daemon = True
                                scheduler_thread.start()
                                st.session_state.scheduler_running = True
                            except ValueError:
                                st.error("Invalid time format. Please use HH:MM (24-hour format).")
                                logging.error("Invalid time format for scheduler")
                        else:
                            st.error("No valid access token or schedule time. Re-authenticate or set a valid time.")
                            logging.error("No valid access token or schedule time for scheduler")
            else:
                st.warning("No projects with message boards found. Ensure you have access to projects with message boards enabled.")
                logging.warning("No projects with message boards found")
        else:
            st.info("Please authenticate with Basecamp to continue.")
            logging.info("Awaiting Basecamp authentication")
    except Exception as e:
        logging.error(f"Main function error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()