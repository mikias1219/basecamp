import streamlit as st
import requests
import json
import time
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import urllib.parse
import socket
import socketserver
import http.server
import webbrowser
import uuid
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    filename='basecamp_analyzer.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
BASE_URL = "https://3.basecampapi.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.2
API_KEY = "AIzaSyArWCID8FdgwcFJpS_mUJNlLy6QJhMvf5w"  # Gemini API key
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
CHECKIN_DATA_FILE = "checkin_data.json"
TASK_DATA_FILE = "task_data.json"
TOKEN_FILE = "access_token.json"
REPORTS_FILE = "reports.json"
TOKEN_EXPIRY = timedelta(days=1)

CONFIG = {
    "CLIENT_ID": "6c916f56910da5118a83a2bf9c762d2967c430f0",
    "CLIENT_SECRET": "5b139ebdf83caec06caff04c02180e93e1d28374"
}
CLIENT_ID = CONFIG.get("CLIENT_ID")
CLIENT_SECRET = CONFIG.get("CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/oauth/callback"

from datetime import datetime, timedelta, timezone

def save_access_token(token: str, expiry: datetime):
    try:
        # Ensure expiry is UTC-aware
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump({"access_token": token, "expiry": expiry.isoformat()}, f)
        logging.info("Access token saved successfully")
    except Exception as e:
        logging.error(f"Failed to save access token: {str(e)}")
        st.error("Failed to save access token.")

def load_access_token() -> Optional[Dict]:
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            expiry_str = data.get("expiry")
            if not isinstance(expiry_str, str):
                logging.error(f"Invalid expiry format in {TOKEN_FILE}: {expiry_str}")
                os.remove(TOKEN_FILE)
                return None
            try:
                # Parse expiry as UTC-aware datetime
                expiry = datetime.fromisoformat(expiry_str)
                if expiry.tzinfo is None:
                    # If parsed datetime is naive, assume it's UTC
                    expiry = expiry.replace(tzinfo=timezone.utc)
                # Compare with current UTC time
                if datetime.now(timezone.utc) < expiry:
                    return {"access_token": data.get("access_token"), "expiry": expiry}
                else:
                    logging.info("Access token expired, removing token file")
                    os.remove(TOKEN_FILE)
                    return None
            except ValueError as e:
                logging.error(f"Failed to parse expiry: {str(e)}")
                os.remove(TOKEN_FILE)
                return None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.debug(f"Token file not found or invalid: {str(e)}")
        return None
def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                logging.debug(f"Port {port} is available")
                return port
            except OSError:
                port += 1
    logging.error("No available ports found")
    raise OSError("No available ports found")

def retry_request(func, *args, max_retries=MAX_RETRIES, backoff_factor=2, **kwargs):
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            logging.debug(f"Request attempt {attempt + 1}: Status {response.status_code}, URL: {kwargs.get('url', args[1] if len(args) > 1 else 'unknown')}")
            if response.status_code == 200 or response.status_code == 201:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logging.warning(f"Rate limit hit, retrying after {retry_after}s")
                time.sleep(retry_after)
                continue
            elif response.status_code == 401:
                st.error("Authentication failed. Please re-authenticate.")
                logging.error("401 Unauthorized: Invalid or expired token")
                return None
            elif response.status_code == 403:
                st.error("Access denied. Check your Basecamp permissions.")
                logging.error(f"403 Forbidden: Insufficient permissions, Response: {response.text}")
                return None
            elif response.status_code == 404:
                logging.warning(f"404 Not Found: Resource does not exist, Response: {response.text}")
                return response
            elif response.status_code >= 500:
                logging.warning(f"Server error {response.status_code}, retrying")
                time.sleep(backoff_factor ** attempt)
                continue
            else:
                st.error(f"Request failed: {response.status_code}")
                logging.error(f"Unexpected status {response.status_code}: {response.text}")
                return response
        except requests.RequestException as e:
            logging.error(f"Request exception on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
    logging.error("Max retries reached, request failed")
    return None

@st.cache_data(ttl=3600)
def get_paginated_results(url: str, headers: Dict, params: Optional[Dict] = None) -> List[Dict]:
    results = []
    while url:
        response = retry_request(requests.get, url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if response and response.ok:
            data = response.json()
            results.extend(data if isinstance(data, list) else [data])
            link_header = response.headers.get('Link', '')
            next_url = None
            for link in link_header.split(','):
                if 'rel="next"' in link:
                    next_url = link[link.find('<')+1:link.find('>')]
                    break
            url = next_url
            logging.debug(f"Paginated results: {len(results)} items fetched from {url}")
            time.sleep(RATE_LIMIT_DELAY)
        else:
            logging.warning(f"Pagination stopped: Response {response.status_code if response else 'None'}, URL: {url}")
            break
    return results

def get_access_token():
    token_data = load_access_token()
    if token_data:
        logging.debug("Using existing access token")
        return token_data["access_token"]

    port = find_available_port()
    AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"

    class OAuthHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/oauth/callback'):
                params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                code = params.get('code', [None])[0]
                if code:
                    token_response = retry_request(
                        requests.post,
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
                    if token_response and token_response.ok:
                        token_data = token_response.json()
                        access_token = token_data.get("access_token")
                        if access_token:
                            # Use UTC-aware datetime
                            expiry = datetime.now(timezone.utc) + TOKEN_EXPIRY
                            save_access_token(access_token, expiry)
                            st.session_state.access_token = access_token
                            self.respond_with("Success! You can close this tab.")
                        else:
                            logging.error("No access token in response")
                            self.respond_with("Token exchange failed: No access token.")
                    else:
                        logging.error(f"Token exchange failed: {token_response.text if token_response else 'No response'}")
                        self.respond_with("Token exchange failed.")
                else:
                    logging.error("No code found in callback URL")
                    self.respond_with("No code found in callback URL")
            else:
                logging.error(f"Invalid callback URL: {self.path}")
                self.respond_with("Invalid callback URL")

        def respond_with(self, message):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())

    st.info("Opening browser for Basecamp authorization...")
    logging.info("Initiating OAuth flow")
    webbrowser.open(AUTH_URL)
    with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
        httpd.timeout = 120
        httpd.handle_request()

    token = st.session_state.get("access_token")
    if not token:
        logging.error("OAuth flow failed, no access token obtained")
        st.error("Authentication failed. Please try again.")
    return token

def get_account_info(access_token: str) -> Optional[int]:
    if not access_token:
        logging.error("No access token provided for account info")
        return None
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        accounts = data.get("accounts", [])
        if accounts:
            account_id = accounts[0]["id"]
            logging.debug(f"Retrieved account ID: {account_id}")
            return account_id
        logging.warning("No accounts found in authorization response")
        st.error("No Basecamp accounts found.")
    else:
        logging.error(f"Failed to fetch account info: {response.text if response else 'No response'}")
    return None

@st.cache_data(ttl=3600)
def get_all_checkins(account_id: int, access_token: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
    checkins = []
    
    buckets_url = f"{BASE_URL}/{account_id}/projects.json"
    buckets_response = retry_request(requests.get, buckets_url, headers=headers, timeout=REQUEST_TIMEOUT)
    if not buckets_response or not buckets_response.ok:
        logging.error(f"Failed to fetch buckets: Status {buckets_response.status_code if buckets_response else 'None'}")
        st.error("Failed to fetch projects. Check your permissions or Basecamp account setup.")
        return []
    buckets = buckets_response.json()

    for bucket in buckets:
        bucket_id = bucket.get("id")
        questionnaire_id = None
        questionnaire_url = None
        for dock_item in bucket.get("dock", []):
            if dock_item.get("name") == "questionnaire" and dock_item.get("enabled"):
                questionnaire_id = dock_item.get("id")
                questionnaire_url = dock_item.get("url")
                break

        if not questionnaire_id:
            logging.debug(f"No enabled questionnaire found in dock for bucket {bucket_id}")
            continue

        questions_url = f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questionnaires/{questionnaire_id}/questions.json"
        questions_response = retry_request(requests.get, questions_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if questions_response and questions_response.ok:
            questions = questions_response.json()
            logging.debug(f"Fetched {len(questions)} questions for questionnaire {questionnaire_id} in bucket {bucket_id}")
            for question in questions:
                checkins.append({
                    "title": question.get("title", "Untitled Question"),
                    "url": f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questions/{question['id']}",
                    "account_id": account_id,
                    "bucket_id": bucket_id,
                    "question_id": question.get("id"),
                    "questionnaire_id": questionnaire_id,
                    "project_name": bucket.get("name")
                })
        else:
            logging.error(f"Failed to fetch questions for questionnaire {questionnaire_id} in bucket {bucket_id}")
            questionnaire_response = retry_request(requests.get, questionnaire_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if questionnaire_response and questionnaire_response.ok:
                questionnaire_data = questionnaire_response.json()
                questions_url = questionnaire_data.get("questions_url")
                if questions_url:
                    questions_response = retry_request(requests.get, questions_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    if questions_response and questions_response.ok:
                        questions = questions_response.json()
                        for question in questions:
                            checkins.append({
                                "title": question.get("title", "Untitled Question"),
                                "url": f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questions/{question['id']}",
                                "account_id": account_id,
                                "bucket_id": bucket_id,
                                "question_id": question.get("id"),
                                "questionnaire_id": questionnaire_id,
                                "project_name": bucket.get("name")
                            })
    logging.info(f"Fetched {len(checkins)} check-in questions")
    return checkins

def filter_answers_by_date(answers: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        filtered_answers = []
        for answer in answers:
            try:
                created_at = datetime.fromisoformat(answer["created_at"].replace("Z", "+00:00"))
                if start <= created_at <= end:
                    filtered_answers.append({
                        "question_id": answer.get("question_id"),
                        "question_title": answer.get("question_title", "Unknown"),
                        "content": answer.get("content", "").strip(),
                        "creator": answer.get("creator", {}).get("name", "Unknown"),
                        "created_at": answer["created_at"],
                        "bucket_id": answer.get("bucket_id")
                    })
            except ValueError as e:
                logging.error(f"Invalid created_at format in answer: {answer.get('created_at')}, Error: {str(e)}")
                continue
        logging.debug(f"Filtered {len(filtered_answers)} answers from {len(answers)} for {start_date} to {end_date}")
        return filtered_answers
    except ValueError as e:
        logging.error(f"Invalid date format: {str(e)}")
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        return []

def fetch_and_structure_answers(account_id: int, bucket_id: int, question_ids: List[int], access_token: str, start_date: str, end_date: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
    all_answers = []
    for question_id in question_ids:
        url = f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questions/{question_id}/answers.json"
        answers = get_paginated_results(url, headers)
        if not answers and answers != []:
            logging.warning(f"No answers or error for question ID {question_id}: Response {answers}")
            continue
        for answer in answers:
            answer["question_id"] = question_id
            answer["bucket_id"] = bucket_id
            answer["question_title"] = next((q["title"] for q in get_questionnaire_questions(account_id, bucket_id, access_token) if q["id"] == question_id), "Unknown")
        all_answers.extend(answers)
    filtered_answers = filter_answers_by_date(all_answers, start_date, end_date)
    return filtered_answers

@st.cache_data(ttl=3600)
def get_questionnaire_questions(account_id: int, bucket_id: int, access_token: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
    url = f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questionnaires.json"
    questionnaires = get_paginated_results(url, headers)
    questions = []
    for questionnaire in questionnaires:
        questionnaire_id = questionnaire.get("id")
        questions_url = f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questionnaires/{questionnaire_id}/questions.json"
        questions_response = get_paginated_results(questions_url, headers)
        questions.extend(questions_response)
    return questions

@st.cache_data(ttl=3600)
def get_projects(account_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    return [{
        "name": project['name'],
        "id": project['id'],
        "todoset_id": next((dock["id"] for dock in project.get("dock", []) if dock["name"] == "todoset" and dock["enabled"]), None),
        "updated_at": project.get('updated_at')
    } for project in projects]

@st.cache_data(ttl=3600)
def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todosets/{todoset_id}.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.ok:
        todoset = response.json()
        todolists_url = todoset.get("todolists_url", "")
        todolists = get_paginated_results(todolists_url, headers)
        return [{"title": todolist['title'], "id": todolist['id'], "updated_at": todolist.get('updated_at')} for todolist in todolists]
    return []

@st.cache_data(ttl=3600)
def get_tasks(account_id: int, project_id: int, todolist_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists/{todolist_id}/todos.json"
    tasks = get_paginated_results(url, headers)
    task_list = []
    for task in tasks:
        task_id = task.get("id")
        task_response = retry_request(
            requests.get,
            f"{BASE_URL}/{account_id}/buckets/{project_id}/todos/{task_id}.json",
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        if task_response and task_response.ok:
            task_data = task_response.json()
            task_list.append({
                "title": task_data.get("title", "N/A"),
                "status": task_data.get("status", "N/A"),
                "due_on": task_data.get("due_on", "N/A"),
                "id": task_id,
                "assignee": task_data.get("assignee", {}).get("name", "Unassigned"),
                "creator": task_data.get("creator", {}).get("name", "Unknown"),
                "comments": get_task_comments(account_id, project_id, task_id, access_token),
                "updated_at": task_data.get("updated_at", "N/A")
            })
            time.sleep(RATE_LIMIT_DELAY)
    return task_list

def get_task_comments(account_id: int, project_id: int, task_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.status_code == 200:
        return [{
            "content": comment.get("content", "N/A").strip(),
            "created_at": comment.get("created_at", "N/A"),
            "id": comment.get("id", "N/A"),
            "creator": comment.get("creator", {}).get("name", "N/A")
        } for comment in response.json()]
    return []

def get_new_comments(account_id: int, project_id: int, task_id: int, access_token: str, existing_comments: List[Dict]) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.status_code == 200:
        existing_comment_ids = {comment['id'] for comment in existing_comments}
        return [{
            "content": comment.get("content", "N/A").strip(),
            "created_at": comment.get("created_at", "N/A"),
            "id": comment.get("id", "N/A"),
            "creator": comment.get("creator", {}).get("name", "N/A")
        } for comment in response.json() if comment.get("id") not in existing_comment_ids]
    return []

def post_comment(account_id: int, project_id: int, recording_id: int, access_token: str, ui_content: str, api_content: str, people: List[Dict]) -> tuple[bool, str]:
    """Post a comment to a Basecamp task and return success status and UI content for display."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{recording_id}/comments.json"
    data = {"content": api_content}

    # Validate mentions in api_content
    valid_ids = {str(p["id"]) for p in people if p.get("id")}
    mention_pattern = r'<bc-attachment sgid="[^"]+" content-type="application/vnd.basecamp.mention">.*?data-avatar-for-person-id="(\d+)"'
    mentioned_ids = set(re.findall(mention_pattern, api_content, re.DOTALL))
    if mentioned_ids and not mentioned_ids.issubset(valid_ids):
        logging.warning(f"Comment contains invalid person IDs: {mentioned_ids - valid_ids}")
        st.warning("Comment may not tag correctly; unexpected person IDs detected.")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.status_code == 201:
            logging.info(f"Posted comment to task {recording_id}")
            return True, ui_content
        else:
            logging.error(f"Comment post failed: {response.status_code} - {response.text}")
            st.error(f"Failed to post comment: {response.status_code} - {response.text}")
            return False, ui_content
    except Exception as e:
        logging.error(f"Comment post error: {str(e)}")
        st.error(f"Comment post error: {str(e)}")
        return False, ui_content
def get_project_people(account_id: int, project_id: int, access_token: str) -> List[Dict]:
    """Fetch all people in a Basecamp project with sgid for tagging, handling pagination."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)",
        "Accept": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/projects/{project_id}/people.json"
    try:
        response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
        if not response or response.status_code != 200:
            error_msg = f"Failed to fetch people for project {project_id}: HTTP {response.status_code if response else 'No response'}"
            logging.error(error_msg)
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
        if not valid_people:
            st.warning(f"No people found for project {project_id}. Check project permissions or add people in Basecamp.")
        return valid_people
    except Exception as e:
        logging.error(f"Error fetching people for project {project_id}: {str(e)}")
        st.error(f"Error fetching people for project {project_id}: {str(e)}")
        return []
def clean_answer_text(text: str) -> str:
    if not text:
        return ""
    def replace_mention(match):
        figcaption_match = re.search(r'<figcaption>\s*([^<]+)\s*</figcaption>', match.group(0))
        return figcaption_match.group(1).strip() if figcaption_match else ""
    text = re.sub(r'<bc-attachment[^>]*>.*?</bc-attachment>', replace_mention, text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(<br\s*/?>\s*)+|\n+', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def categorize_answer(answer: Dict) -> tuple[bool, bool, bool]:
    content = clean_answer_text(answer.get("content", "")).lower()
    question_title = answer.get("question_title", "").lower()
    learning_keywords = ["learn", "learned", "study", "studied", "understood", "gained", "explored", "discovered"]
    challenge_keywords = ["challenge", "struggle", "struggled", "difficult", "issue", "problem", "obstacle"]
    plan_keywords = ["tomorrow", "plan", "planned", "next", "will", "continue"]
    no_challenge_phrases = ["no challenges", "no challenge", "no issues", "no issue", "no problems", "no problem"]
    is_learning = any(keyword in question_title or keyword in content for keyword in learning_keywords)
    is_challenge = (any(keyword in question_title or keyword in content for keyword in challenge_keywords) and
                   not any(phrase in content for phrase in no_challenge_phrases))
    is_plan = any(keyword in question_title or keyword in content for keyword in plan_keywords)
    return is_learning, is_challenge, is_plan

def extract_challenge_text(text: str) -> str:
    if not text:
        return ""
    cleaned_text = clean_answer_text(text)
    text_lower = cleaned_text.lower()
    no_challenge_phrases = ["no challenges", "no challenge", "no issues", "no issue", "no problems", "no problem"]
    if any(phrase in text_lower for phrase in no_challenge_phrases):
        return ""
    challenge_keywords = ["challenge", "struggle", "struggled", "difficult", "issue", "problem", "obstacle"]
    challenge_pattern = r'\b(' + '|'.join(challenge_keywords) + r')\b'
    match = re.search(challenge_pattern, text_lower)
    if not match:
        return ""
    start_pos = match.start()
    challenge_text = cleaned_text[start_pos:]
    stop_keywords = ["learn", "learned", "study", "studied", "plan", "planned", "tomorrow", "next", "will"]
    end_pos = len(challenge_text)
    for stop_keyword in stop_keywords:
        stop_match = re.search(r'\b' + stop_keyword + r'\b', challenge_text.lower())
        if stop_match and stop_match.start() < end_pos:
            end_pos = stop_match.start()
    sentence_end = re.search(r'[.!?]\s|$', challenge_text)
    if sentence_end and sentence_end.start() < end_pos:
        end_pos = sentence_end.start()
    return challenge_text[:end_pos].strip()

def extract_plan_text(text: str) -> str:
    if not text:
        return ""
    cleaned_text = clean_answer_text(text)
    text_lower = cleaned_text.lower()
    plan_keywords = ["tomorrow", "plan", "planned", "next", "will", "continue"]
    plan_pattern = r'\b(' + '|'.join(plan_keywords) + r')\b'
    match = re.search(plan_pattern, text_lower)
    if not match:
        return ""
    start_pos = match.start()
    plan_text = cleaned_text[start_pos:]
    stop_keywords = ["learn", "learned", "study", "studied", "challenge", "struggle", "problem", "issue"]
    end_pos = len(plan_text)
    for stop_keyword in stop_keywords:
        stop_match = re.search(r'\b' + stop_keyword + r'\b', plan_text.lower())
        if stop_match and stop_match.start() < end_pos:
            end_pos = stop_match.start()
    sentence_end = re.search(r'[.!?]\s|$', plan_text)
    if sentence_end and sentence_end.start() < end_pos:
        end_pos = sentence_end.start()
    return plan_text[:end_pos].strip()

def generate_llm_output(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            output = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No output generated")
            logging.debug("Successfully generated LLM output")
            return output
        logging.error(f"LLM request failed: {response.status_code} - {response.text}")
        return f"Failed to generate output: {response.status_code}"
    except Exception as e:
        logging.error(f"Error generating LLM output: {str(e)}")
        return f"Error generating output: {str(e)}"

def save_checkin_data(data: List[Dict]):
    try:
        with open(CHECKIN_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved {len(data)} check-in answers to {CHECKIN_DATA_FILE}")
    except Exception as e:
        logging.error(f"Failed to save checkin data: {str(e)}")
        st.error("Failed to save check-in data.")

def load_checkin_data() -> List[Dict]:
    try:
        with open(CHECKIN_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.debug(f"Loaded {len(data)} answers from {CHECKIN_DATA_FILE}")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.debug(f"No checkin data file found or invalid: {str(e)}")
        return []

def save_task_data(data: List[Dict]):
    try:
        with open(TASK_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved task data to {TASK_DATA_FILE}")
    except Exception as e:
        logging.error(f"Failed to save task data: {str(e)}")
        st.error("Failed to save task data.")

def load_task_data() -> List[Dict]:
    try:
        with open(TASK_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.debug(f"Loaded task data from {TASK_DATA_FILE}")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.debug(f"No task data file found or invalid: {str(e)}")
        return []

def save_reports(reports: Dict):
    try:
        with open(REPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        logging.info("Saved reports to reports.json")
    except Exception as e:
        logging.error(f"Failed to save reports: {str(e)}")
        st.error("Failed to save reports.")

def is_task_data_up_to_date(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None) -> bool:
    if not existing_data:
        return False
    projects = get_projects(account_id, access_token)
    for project in projects:
        if selected_project and project['name'] != selected_project:
            continue
        existing_project = next((p for p in existing_data if p['project_id'] == project['id']), None)
        if not existing_project or existing_project.get('updated_at') != project['updated_at']:
            return False
        for todolist in existing_project.get('todolists', []):
            for task in todolist.get('tasks', []):
                task_response = retry_request(
                    requests.get,
                    f"{BASE_URL}/{account_id}/buckets/{project['id']}/todos/{task['id']}.json",
                    headers={"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAI"},
                    timeout=REQUEST_TIMEOUT
                )
                if task_response and task_response.ok and task_response.json().get('updated_at') != task.get('updated_at'):
                    return False
    return True

def check_new_task_data(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None) -> tuple:
    start_time = time.time()
    new_data = []
    updated = False
    projects = get_projects(account_id, access_token)
    
    for project in projects:
        if selected_project and project['name'] != selected_project:
            existing_project = next((p for p in existing_data if p['project_id'] == project['id']), None)
            if existing_project:
                new_data.append(existing_project)
            continue
        
        existing_project = next((p for p in existing_data if p['project_id'] == project['id']), None)
        project_data = existing_project or {
            "project_name": project['name'],
            "project_id": project['id'],
            "updated_at": project['updated_at'],
            "todolists": []
        }
        
        if not existing_project or existing_project.get('updated_at') != project['updated_at']:
            updated = True
            if project['todoset_id']:
                todolists = get_todoset(account_id, project['id'], project['todoset_id'], access_token)
                for todolist in todolists:
                    existing_todolist = None
                    if existing_project:
                        existing_todolist = next((t for t in existing_project['todolists'] if t['todolist_id'] == todolist['id']), None)
                    
                    if not existing_todolist or existing_todolist.get('updated_at') != todolist['updated_at']:
                        tasks = get_tasks(account_id, project['id'], todolist['id'], access_token)
                        project_data["todolists"].append({
                            "todolist_name": todolist['title'],
                            "todolist_id": todolist['id'],
                            "updated_at": todolist['updated_at'],
                            "tasks": tasks
                        })
                    elif existing_todolist:
                        project_data["todolists"].append(existing_todolist)
            new_data.append(project_data)
        else:
            new_data.append(project_data)
    
    logging.info(f"Task data check completed in {time.time() - start_time:.2f} seconds")
    return new_data, updated

def generate_smart_reply(task: Dict, comments: List[Dict], mentions: List[Dict], selam_mention: Dict = None, cc_mention: Dict = None) -> tuple[str, str]:
    """Generate a smart reply with names for UI and <bc-attachment> mentions for API."""
    headers = {"Content-Type": "application/json"}
    
    # Handle Selam and Cc mentions
    selam_name = selam_mention.get("name", "") if selam_mention else ""
    cc_name = cc_mention.get("name", "") if cc_mention else ""
    selam_mention_str = format_mentions(selam_mention, is_cc=False) if selam_mention else ""
    cc_mention_str = format_mentions(cc_mention, is_cc=True) if cc_mention else ""
    
    # Handle additional mentions
    mention_names = [person["name"] for person in mentions if person.get("name")]
    mention_tags = [format_mentions(person, is_cc=False) for person in mentions]
    mention_str_ui = ", ".join(mention_names) if mention_names else ""
    mention_str_api = " ".join(mention_tags) if mention_tags else ""
    
    # Validate mentions
    for person in mentions + ([selam_mention] if selam_mention else []) + ([cc_mention] if cc_mention else []):
        if person and not person.get("sgid"):
            logging.warning(f"Person {person.get('name', 'Unknown')} lacks sgid, mention may fail.")
            st.warning(f"Person {person.get('name', 'Unknown')} cannot be tagged properly due to missing data.")
    
    comment_context = "\n".join([f"Comment by {c.get('creator', 'N/A')}: {c.get('content', 'N/A')}" for c in comments]) or "No comments available."
    
    selam_prefix_ui = f"Selam {selam_name}" if selam_name else "Selam"
    cc_suffix_ui = f"Cc {cc_name}" if cc_name else "Looking forward to your thoughts!"
    mentions_ui = f"Mentions: {mention_str_ui}" if mention_str_ui else ""
    
    prompt = f"""
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {comment_context}
    
    Generate a concise, professional reply to the latest comment or task context.
    - Start with '{selam_prefix_ui}'.
    - Acknowledge the latest comment (if any) or task status.
    - Include mentions: {mention_str_ui}.
    - Provide a relevant update, question, or action item.
    - End with '{cc_suffix_ui}'.
    - Use plain names for mentions (e.g., 'Selamawit Getaneh') in the body for display.
    - Ensure the reply is suitable for posting to Basecamp with proper mentions.
    """
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            generated_content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No reply generated")
            if not generated_content.lower().startswith("selam"):
                generated_content = f"{selam_prefix_ui}, {generated_content.lstrip(' ,.')}".strip()
            if cc_name and not generated_content.rstrip().endswith(cc_suffix_ui):
                generated_content = f"{generated_content.rstrip('.')} {cc_suffix_ui}".strip()
            elif not cc_name and not generated_content.rstrip().endswith("Looking forward to your thoughts!"):
                generated_content = f"{generated_content.rstrip('.')} Looking forward to your thoughts!".strip()
            
            # Create API version with <bc-attachment> mentions
            api_content = generated_content
            if selam_name and selam_mention_str:
                api_content = api_content.replace(f"Selam {selam_name}", f"Selam {selam_mention_str}")
            if cc_name and cc_mention_str:
                api_content = api_content.replace(f"Cc {cc_name}", cc_mention_str)
            if mention_names and mention_str_api:
                if "Mentions:" in api_content:
                    api_content = api_content.replace(f"Mentions: {mention_str_ui}", mention_str_api)
                else:
                    api_content = f"{api_content.rstrip('.')} {mention_str_api}".strip()
            
            # Validate API content for mentions
            mention_pattern = r'<bc-attachment sgid="[^"]+" content-type="application/vnd.basecamp.mention">'
            if (selam_mention or cc_mention or mentions) and not re.search(mention_pattern, api_content):
                logging.warning("API content lacks expected mention tags")
                st.warning("Generated reply may not include proper mentions for Basecamp.")
            
            return generated_content, api_content
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown API error")
            logging.error(f"Gemini API error: {response.status_code} - {response.text}")
            error_ui = f"{selam_prefix_ui}, Failed to generate reply: {error_message} {cc_suffix_ui}".strip()
            return error_ui, error_ui
    except Exception as e:
        logging.error(f"Gemini API request failed: {str(e)}")
        error_ui = f"{selam_prefix_ui}, Error generating reply: {str(e)} {cc_suffix_ui}".strip()
        return error_ui, error_ui
def format_mentions(person: Dict, is_cc: bool = False) -> str:
    """Format a person's mention in Basecamp <bc-attachment> syntax with all required details."""
    if not person or not person.get("id") or not person.get("sgid"):
        logging.warning(f"Invalid person data for mention: {person}")
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
    return f"Cc {mention}" if is_cc else mention
def fetch_gemini_insights(task: Dict, comments: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {json.dumps(comments, indent=2)}
    
    Provide insights and automation suggestions:
    - Summarize task and comment context.
    - Suggest automation actions (e.g., reminders, status updates).
    - Identify potential issues or delays.
    - Offer recommendations for task completion.
    """
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights available")
        return "Failed to fetch insights"
    except Exception as e:
        return f"Error fetching insights: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Basecamp AI & Check-in Analyzer", layout="wide")

def main():
    # Custom CSS for styling
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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        border-radius: 8px;
        padding: 10px;
    }
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 8px;
        padding: 10px;
    }
    .output-box {
        border: 1px solid #bdc3c7;
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        margin-top: 10px;
    }
    .comment-box, .answer-box {
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        background-color: #ffffff;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .image-container img {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    <div class="navbar">Basecamp AI & Check-in Analyzer</div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'account_id' not in st.session_state:
        st.session_state.account_id = None
    if 'checkins' not in st.session_state:
        st.session_state.checkins = []
    if 'task_data' not in st.session_state:
        st.session_state.task_data = load_task_data()
    if 'checkin_page' not in st.session_state:
        st.session_state.checkin_page = 1
    if 'comment_page' not in st.session_state:
        st.session_state.comment_page = 1
    if 'items_per_page' not in st.session_state:
        st.session_state.items_per_page = 5

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Check-in Analyzer", "Task Manager", "Settings", "Debug"], key="nav_radio")

    if page == "Dashboard":
        st.header("Dashboard")
        st.write("Manage Basecamp projects, tasks, and check-ins with AI-powered insights and automation.")
        st.markdown("""
        <div class="image-container">
            <img src="https://via.placeholder.com/600x200.png?text=Basecamp+AI+Dashboard" alt="Dashboard Image">
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Quick Start")
        st.markdown("""
        1. **Authenticate**: Go to Settings to connect with Basecamp (valid for 1 day).
        2. **Check-in Analyzer**: Analyze daily check-in responses and generate reports.
        3. **Task Manager**: Manage tasks, post comments, and generate AI insights.
        4. **Debug**: Diagnose issues with data fetching.
        """)
        if st.session_state.task_data or load_checkin_data():
            project_count = len(st.session_state.task_data)
            task_count = sum(len(todolist['tasks']) for project in st.session_state.task_data for todolist in project.get('todolists', []))
            comment_count = sum(len(task['comments']) for project in st.session_state.task_data for todolist in project.get('todolists', []) for task in todolist.get('tasks', []))
            answer_count = len(load_checkin_data())
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Projects", project_count)
            with col2:
                st.metric("Tasks", task_count)
            with col3:
                st.metric("Comments", comment_count)
            st.metric("Check-in Answers", answer_count)
        else:
            st.info("No data loaded. Please authenticate and fetch data in Settings.")

    elif page == "Check-in Analyzer":
        st.header("Check-in Analyzer")
        st.write("Analyze Basecamp Automatic Check-in responses to generate internship reports and insights.")
        st.markdown("""
        <div class="image-container">
            <img src="https://via.placeholder.com/600x200.png?text=Check-in+Analyzer" alt="Check-in Analyzer Image">
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.access_token or not st.session_state.account_id:
            st.error("Please authenticate in the Settings page to proceed.")
            return

        # Fetch check-ins
        if not st.session_state.checkins or st.button("Refresh Check-ins"):
            with st.spinner("Fetching check-ins..."):
                st.session_state.checkins = get_all_checkins(st.session_state.account_id, st.session_state.access_token)
                if not st.session_state.checkins:
                    st.warning("No check-in questions found. Check permissions or questionnaire setup in Basecamp.")
                    return

        # Select check-in
        checkin_options = [f"{c['title']} (Project: {c['project_name']}, Bucket ID: {c['bucket_id']})" for c in st.session_state.checkins]
        selected_checkin = st.selectbox("Select Check-in Question", checkin_options)
        selected_index = checkin_options.index(selected_checkin) if selected_checkin in checkin_options else 0
        selected_checkin_data = st.session_state.checkins[selected_index]

        # Date range selection
        default_end = datetime.now(timezone.utc).date()
        default_start = default_end - timedelta(days=30)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start)
        with col2:
            end_date = st.date_input("End Date", value=default_end)

        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        if st.button("Fetch Check-in Data"):
            with st.spinner("Fetching check-in data..."):
                answers = fetch_and_structure_answers(
                    selected_checkin_data["account_id"],
                    selected_checkin_data["bucket_id"],
                    [selected_checkin_data["question_id"]],
                    st.session_state.access_token,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                if answers:
                    save_checkin_data(answers)
                    st.success(f"Fetched {len(answers)} answers for {start_date} to {end_date}.")
                else:
                    st.warning(f"No answers found for the selected check-in and date range.")

        answers = load_checkin_data()
        if answers:
            st.subheader("Fetched Answers")
            st.write(f"Total answers: {len(answers)}")

            # Pagination for answers
            total_answers = len(answers)
            total_pages = (total_answers + st.session_state.items_per_page - 1) // st.session_state.items_per_page
            start_idx = (st.session_state.checkin_page - 1) * st.session_state.items_per_page
            end_idx = start_idx + st.session_state.items_per_page
            paginated_answers = answers[start_idx:end_idx]

            for answer in paginated_answers:
                cleaned_text = clean_answer_text(answer['content'])
                st.markdown(f"""
                <div class="answer-box">
                    <strong>Creator:</strong> {answer['creator']}<br>
                    <strong>Posted:</strong> {answer['created_at']}<br>
                    <strong>Question:</strong> {answer['question_title']}<br>
                    <strong>Answer:</strong> {cleaned_text}
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.session_state.checkin_page > 1:
                    if st.button("Previous Answers", key="prev_answers"):
                        st.session_state.checkin_page -= 1
            with col3:
                if st.session_state.checkin_page < total_pages:
                    if st.button("Next Answers", key="next_answers"):
                        st.session_state.checkin_page += 1
            with col2:
                st.write(f"Page {st.session_state.checkin_page} of {total_pages}")

            # Categorize answers
            learning_responses = []
            challenge_responses = []
            plan_responses = []
            for answer in answers:
                is_learning, is_challenge, is_plan = categorize_answer(answer)
                if is_learning:
                    learning_responses.append(answer)
                if is_challenge:
                    challenge_text = extract_challenge_text(answer['content'])
                    if challenge_text:
                        answer_copy = answer.copy()
                        answer_copy['challenge_text'] = challenge_text
                        challenge_responses.append(answer_copy)
                if is_plan:
                    plan_text = extract_plan_text(answer['content'])
                    if plan_text:
                        answer_copy = answer.copy()
                        answer_copy['plan_text'] = plan_text
                        plan_responses.append(answer_copy)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Learning Responses", len(learning_responses))
            with col2:
                st.metric("Challenge Responses", len(challenge_responses))
            with col3:
                st.metric("Planned Tasks", len(plan_responses))

            with st.expander("View Learning Responses"):
                for ans in learning_responses:
                    cleaned_text = clean_answer_text(ans['content'])
                    st.write(f"- {ans['creator']}: {cleaned_text}")
            with st.expander("View Challenge Responses"):
                for ans in challenge_responses:
                    st.write(f"- {ans['creator']}: {ans['challenge_text']}")
            with st.expander("View Planned Tasks"):
                for ans in plan_responses:
                    st.write(f"- {ans['creator']}: {ans['plan_text']}")

            st.subheader("Generate Reports")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Weekly Summary"):
                    with st.spinner("Generating summary..."):
                        cleaned_learning = [f"- {clean_answer_text(a['content'])} on {a['created_at']}" for a in learning_responses]
                        cleaned_challenges = [f"- {a['challenge_text']} on {a['created_at']}" for a in challenge_responses]
                        summary_prompt = f"""
                        You are an AI assistant summarizing internship progress based on daily check-in responses from Basecamp. Given the following data from {start_date} to {end_date}:

                        **Learning Responses**:
                        {chr(10).join(cleaned_learning) if cleaned_learning else "- None"}

                        **Challenge Responses**:
                        {chr(10).join(cleaned_challenges) if cleaned_challenges else "- None"}

                        Provide a concise summary (150-200 words) for a weekly internship report. Highlight key learnings and challenges faced by the team, ensuring a professional tone. Focus on common themes, significant achievements, and recurring obstacles. Avoid including specific names of individuals. Structure the summary with two sections: "Key Learnings" and "Challenges Encountered".
                        """
                        summary = generate_llm_output(summary_prompt)
                        st.session_state.summary = summary
                        st.markdown(f"<div class='output-box'><strong>Weekly Summary</strong>:<br>{summary}</div>", unsafe_allow_html=True)
                        save_reports({"summary": summary, "insights": st.session_state.get("insights", "")})

            with col2:
                if st.button("Generate Next Week's Insights"):
                    with st.spinner("Generating insights..."):
                        cleaned_learning = [f"- {clean_answer_text(a['content'])} on {a['created_at']}" for a in learning_responses]
                        cleaned_challenges = [f"- {a['challenge_text']} on {a['created_at']}" for a in challenge_responses]
                        cleaned_plans = [f"- {a['plan_text']} on {a['created_at']}" for a in plan_responses]
                        insights_prompt = f"""
                        You are an AI assistant providing insights for a software development internship team based on Basecamp daily check-in responses from {start_date} to {end_date}:

                        **Learning Responses**:
                        {chr(10).join(cleaned_learning) if cleaned_learning else "- None"}

                        **Challenge Responses**:
                        {chr(10).join(cleaned_challenges) if cleaned_challenges else "- None"}

                        **Planned Tasks**:
                        {chr(10).join(cleaned_plans) if cleaned_plans else "- None"}

                        Analyze the learnings, challenges, and planned tasks to generate actionable insights (150-200 words) for the next week's accomplishments. Identify common goals, potential focus areas, and recommend specific actions to enhance productivity or collaboration. Use a professional, encouraging tone. Avoid mentioning specific individuals. Structure the insights with two sections: "Key Planned Activities" and "Recommended Next Steps".
                        """
                        insights = generate_llm_output(insights_prompt)
                        st.session_state.insights = insights
                        st.markdown(f"<div class='output-box'><strong>Next Week's Insights</strong>:<br>{insights}</div>", unsafe_allow_html=True)
                        save_reports({"summary": st.session_state.get("summary", ""), "insights": insights})
    elif page == "Task Manager":
        st.header("Task Manager")
        st.write("Manage Basecamp tasks, post comments, and generate AI insights.")
        st.markdown("""
        <div class="image-container">
            <img src="https://via.placeholder.com/600x200.png?text=Task+Manager" alt="Task Manager Image">
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.task_data:
            st.info("No task data available. Please fetch data from Settings.")
            return

        project_names = [project['project_name'] for project in st.session_state.task_data]
        selected_project = st.sidebar.selectbox("Select Project", project_names, key="project_select")
        project = next((p for p in st.session_state.task_data if p['project_name'] == selected_project), None)
        
        if project:
            if st.session_state.access_token and st.session_state.account_id:
                if st.sidebar.button("Refresh Task Data"):
                    with st.spinner("Checking for task updates..."):
                        new_data, updated = check_new_task_data(
                            st.session_state.account_id,
                            st.session_state.access_token,
                            st.session_state.task_data,
                            selected_project
                        )
                        st.session_state.task_data = new_data
                        save_task_data(new_data)
                        if updated:
                            st.success(f"Task data updated for {selected_project}")
                        else:
                            st.info(f"No updates for {selected_project}")

            todolist_names = [todolist['todolist_name'] for todolist in project.get('todolists', [])]
            selected_todolist = st.sidebar.selectbox("Select To-do List", todolist_names, key="todolist_select")
            todolist = next((t for t in project['todolists'] if t['todolist_name'] == selected_todolist), None)
            
            if todolist:
                tasks = todolist.get('tasks', [])
                if not tasks:
                    st.info("No tasks in this to-do list.")
                    return
                task_titles = [task['title'] for task in tasks]
                selected_task = st.sidebar.selectbox("Select Task", task_titles, key="task_select")
                task = next((t for t in tasks if t['title'] == selected_task), None)
                
                if task:
                    st.subheader(f"Task: {task['title']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Status**: {task['status']}")
                        st.write(f"**Due Date**: {task['due_on']}")
                    with col2:
                        st.write(f"**Assignee**: {task['assignee']}")
                        st.write(f"**Creator**: {task['creator']}")

                    # Comments with pagination
                    st.subheader("Comments")
                    comments = task['comments']
                    total_comments = len(comments)
                    if total_comments > 0:
                        st.write(f"Total Comments: {total_comments}")
                        start_idx = (st.session_state.comment_page - 1) * st.session_state.items_per_page
                        end_idx = start_idx + st.session_state.items_per_page
                        paginated_comments = comments[start_idx:end_idx]
                        
                        for comment in paginated_comments:
                            st.markdown(f"""
                            <div class="comment-box">
                                <strong>Creator:</strong> {comment['creator']}<br>
                                <strong>Posted:</strong> {comment['created_at']}<br>
                                <strong>Comment:</strong> {comment['content']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        total_pages = (total_comments + st.session_state.items_per_page - 1) // st.session_state.items_per_page
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            if st.session_state.comment_page > 1:
                                if st.button("Previous Comments", key="prev_comments"):
                                    st.session_state.comment_page -= 1
                        with col3:
                            if st.session_state.comment_page < total_pages:
                                if st.button("Next Comments", key="next_comments"):
                                    st.session_state.comment_page += 1
                        with col2:
                            st.write(f"Page {st.session_state.comment_page} of {total_pages}")
                    else:
                        st.info("No comments available.")

                    # Fetch new comments
                    if st.button("Fetch New Comments"):
                        if st.session_state.access_token and st.session_state.account_id:
                            with st.spinner("Fetching new comments..."):
                                new_comments = get_new_comments(
                                    st.session_state.account_id,
                                    project['project_id'],
                                    task['id'],
                                    st.session_state.access_token,
                                    task['comments']
                                )
                                if new_comments:
                                    task['comments'].extend(new_comments)
                                    save_task_data(st.session_state.task_data)
                                    st.success(f"{len(new_comments)} new comments added.")
                                    st.session_state.comment_page = 1
                                else:
                                    st.info("No new comments found.")
                        else:
                            st.error("Please authenticate in Settings.")

                    # Smart Reply with Mentions
                    st.subheader("Smart Reply")
                    people = get_project_people(st.session_state.account_id, project['project_id'], st.session_state.access_token)
                    if not people:
                        st.warning("No people available for tagging in this project. You can still generate and post comments without mentions.")
                    
                    # Initialize session state for suggestions
                    if 'mention_input' not in st.session_state:
                        st.session_state.mention_input = ""
                    if 'selected_mentions' not in st.session_state:
                        st.session_state.selected_mentions = []

                    # Name suggestion input
                    st.write("Type names to mention (suggestions appear as you type):")
                    mention_input = st.text_input("Search for people to mention", value=st.session_state.mention_input, key="mention_input")
                    
                    # Filter people based on input
                    people_names = [p['name'] for p in people if p.get("name") and p.get("name") != "Unknown"]
                    suggestions = [name for name in people_names if mention_input.lower() in name.lower()] if mention_input else []
                    
                    if suggestions:
                        selected_suggestion = st.selectbox("Suggested names:", ["Select a name"] + suggestions, key="suggestion_select")
                        if selected_suggestion != "Select a name" and st.button("Add Mention"):
                            if selected_suggestion not in st.session_state.selected_mentions:
                                st.session_state.selected_mentions.append(selected_suggestion)
                                st.session_state.mention_input = ""  # Clear input after adding
                            else:
                                st.warning(f"{selected_suggestion} is already selected.")
                    
                    # Display selected mentions
                    if st.session_state.selected_mentions:
                        st.write("Selected Mentions:", ", ".join(st.session_state.selected_mentions))
                        if st.button("Clear Mentions"):
                            st.session_state.selected_mentions = []
                    
                    # Selam and Cc dropdowns
                    selam_person_name = st.selectbox("Select person to tag after 'Selam':", ["None"] + people_names, key="selam_person")
                    cc_person_name = st.selectbox("Select person to tag in 'Cc' (optional):", ["None"] + people_names, key="cc_person")
                    
                    if st.button("Generate Smart Reply"):
                        with st.spinner("Generating reply..."):
                            selam_person = next((p for p in people if p.get("name") == selam_person_name), None) if selam_person_name and selam_person_name != "None" else None
                            cc_person = next((p for p in people if p.get("name") == cc_person_name), None) if cc_person_name and cc_person_name != "None" else None
                            mention_people = [p for p in people if p.get("name") in st.session_state.selected_mentions]
                            
                            ui_reply, api_reply = generate_smart_reply(task, task['comments'], mention_people, selam_person, cc_person)
                            if "Failed to generate reply" in ui_reply or "Error generating reply" in ui_reply:
                                st.error(f"Error: {ui_reply}")
                                st.session_state.smart_reply = None
                                st.session_state.smart_reply_api = None
                            else:
                                st.session_state.smart_reply = ui_reply
                                st.session_state.smart_reply_api = api_reply
                                st.markdown(f"<div class='output-box'><strong>Suggested Reply</strong>:<br>{ui_reply}</div>", unsafe_allow_html=True)

                    if 'smart_reply' in st.session_state and st.session_state.smart_reply:
                        edited_reply = st.text_area("Edit Reply", value=st.session_state.smart_reply, key="edit_reply")
                        if st.button("Post Reply"):
                            with st.spinner("Posting reply..."):
                                # Create API version of edited reply
                                api_edited_reply = edited_reply
                                selam_person = next((p for p in people if p.get("name") == selam_person_name), None) if selam_person_name and selam_person_name != "None" else None
                                cc_person = next((p for p in people if p.get("name") == cc_person_name), None) if cc_person_name and cc_person_name != "None" else None
                                mention_people = [p for p in people if p.get("name") in st.session_state.selected_mentions]
                                
                                if selam_person:
                                    api_edited_reply = api_edited_reply.replace(f"Selam {selam_person['name']}", f"Selam {format_mentions(selam_person)}")
                                if cc_person:
                                    api_edited_reply = api_edited_reply.replace(f"Cc {cc_person['name']}", format_mentions(cc_person, is_cc=True))
                                if st.session_state.selected_mentions:
                                    mentions_ui = f"Mentions: {', '.join(st.session_state.selected_mentions)}"
                                    mentions_api = " ".join([format_mentions(p) for p in mention_people])
                                    api_edited_reply = api_edited_reply.replace(mentions_ui, mentions_api) if mentions_ui in api_edited_reply else f"{api_edited_reply} {mentions_api}"
                                
                                success, display_content = post_comment(
                                    st.session_state.account_id,
                                    project['project_id'],
                                    task['id'],
                                    st.session_state.access_token,
                                    edited_reply,
                                    api_edited_reply,
                                    people
                                )
                                if success:
                                    st.success("Reply posted successfully!")
                                    st.markdown(f"<div class='output-box'><strong>Posted Reply</strong>:<br>{display_content}</div>", unsafe_allow_html=True)
                                    new_comments = get_new_comments(
                                        st.session_state.account_id,
                                        project['project_id'],
                                        task['id'],
                                        st.session_state.access_token,
                                        task['comments']
                                    )
                                    task['comments'].extend(new_comments)
                                    save_task_data(st.session_state.task_data)
                                    st.session_state.smart_reply = None
                                    st.session_state.smart_reply_api = None
                                    st.session_state.comment_page = 1
                                    st.session_state.selected_mentions = []  # Clear mentions after posting
                                else:
                                    st.error("Failed to post reply.")

                    # AI Insights
                    st.subheader("AI Insights")
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = fetch_gemini_insights(task, task['comments'])
                            st.markdown(f"<div class='output-box'><strong>Task Insights</strong>:<br>{insights}</div>", unsafe_allow_html=True)

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Basecamp Authentication")
        st.write("Authenticate with Basecamp to fetch project, task, and check-in data. The access token is valid for 1 day.")
        st.markdown("""
        <div class="image-container">
            <img src="https://via.placeholder.com/600x200.png?text=Settings" alt="Settings Image">
        </div>
        """, unsafe_allow_html=True)
        if st.button("Authenticate and Fetch Data"):
            with st.spinner("Authenticating..."):
                access_token = get_access_token()
                if access_token:
                    st.session_state.access_token = access_token
                    account_id = get_account_info(access_token)
                    if account_id:
                        st.session_state.account_id = account_id
                        st.session_state.checkins = []
                        new_data, updated = check_new_task_data(account_id, access_token, st.session_state.task_data)
                        st.session_state.task_data = new_data
                        save_task_data(new_data)
                        st.success("Authentication successful and data fetched!")
                    else:
                        st.error("Failed to fetch account ID.")
                        if os.path.exists(TOKEN_FILE):
                            os.remove(TOKEN_FILE)
                else:
                    st.error("Authentication failed.")

    elif page == "Debug":
        st.header("Debug Information")
        st.write("Diagnose issues with check-in and task fetching.")
        st.markdown("""
        <div class="image-container">
            <img src="https://via.placeholder.com/600x200.png?text=Debug" alt="Debug Image">
        </div>
        """, unsafe_allow_html=True)
        if not st.session_state.access_token or not st.session_state.account_id:
            st.error("Please authenticate in the Settings page to view debug info.")
            return

        st.subheader("Check-in Fetching Debug")
        if st.button("Fetch Raw Check-in Data"):
            with st.spinner("Fetching raw check-in data..."):
                headers = {"Authorization": f"Bearer {st.session_state.access_token}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
                buckets_url = f"{BASE_URL}/{st.session_state.account_id}/projects.json"
                buckets_response = retry_request(requests.get, buckets_url, headers=headers, timeout=REQUEST_TIMEOUT)
                if buckets_response and buckets_response.ok:
                    buckets = buckets_response.json()
                    st.write(f"**Buckets Fetched**: {len(buckets)}")
                    st.json(buckets)
                    for bucket in buckets:
                        bucket_id = bucket.get("id")
                        questionnaire_id = None
                        questionnaire_url = None
                        for dock_item in bucket.get("dock", []):
                            if dock_item.get("name") == "questionnaire" and dock_item.get("enabled"):
                                questionnaire_id = dock_item.get("id")
                                questionnaire_url = dock_item.get("url")
                                break
                        if questionnaire_id:
                            st.write(f"**Questionnaire Found for Bucket {bucket_id}**: ID {questionnaire_id}, URL: {questionnaire_url}")
                            questions_url = f"{BASE_URL}/{st.session_state.account_id}/buckets/{bucket_id}/questionnaires/{questionnaire_id}/questions.json"
                            questions_response = retry_request(requests.get, questions_url, headers=headers, timeout=REQUEST_TIMEOUT)
                            if questions_response and questions_response.ok:
                                questions = questions_response.json()
                                st.write(f"**Questions for Questionnaire {questionnaire_id}**: {len(questions)}")
                                st.json(questions)
                            else:
                                st.error(f"Failed to fetch questions: Status {questions_response.status_code if questions_response else 'None'}")
                        else:
                            st.warning(f"No enabled questionnaire found for bucket {bucket_id}.")

        st.subheader("Task Fetching Debug")
        if st.button("Fetch Raw Task Data"):
            with st.spinner("Fetching raw task data..."):
                projects = get_projects(st.session_state.account_id, st.session_state.access_token)
                st.write(f"**Projects Fetched**: {len(projects)}")
                st.json(projects)
                for project in projects:
                    if project['todoset_id']:
                        todolists = get_todoset(st.session_state.account_id, project['id'], project['todoset_id'], st.session_state.access_token)
                        st.write(f"**Todolists for Project {project['name']}**: {len(todolists)}")
                        st.json(todolists)
                        for todolist in todolists:
                            tasks = get_tasks(st.session_state.account_id, project['id'], todolist['id'], st.session_state.access_token)
                            st.write(f"**Tasks for Todolist {todolist['title']}**: {len(tasks)}")
                            st.json(tasks)

if __name__ == "__main__":
    main()