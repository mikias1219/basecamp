import streamlit as st
import requests
import json
import time
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import socket
import socketserver
import webbrowser
import http.server
import urllib.parse
import uuid

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    filename='basecamp_gemini_app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG_FILE = "config.json"
BASE_URL = "https://3.basecampapi.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 0.2
DEBUG_MODE = True
API_KEY = "AIzaSyArWCID8FdgwcFJpS_mUJNlLy6QJhMvf5w"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
DATA_FILE = "AllData.json"
TOKEN_FILE = "access_token.json"

# Updated configuration
CONFIG = {
    "CLIENT_ID": "6c916f56910da5118a83a2bf9c762d2967c430f0",
    "CLIENT_SECRET": "5b139ebdf83caec06caff04c02180e93e1d28374"
}
CLIENT_ID = CONFIG.get("CLIENT_ID")
CLIENT_SECRET = CONFIG.get("CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/oauth/callback"

# Global access token
ACCESS_TOKEN = None

def save_access_token(token: str):
    """Save access token to a file."""
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"access_token": token}, f)
    logging.info("Access token saved to file")

def load_access_token() -> Optional[str]:
    """Load access token from file if it exists."""
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("access_token")
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error(f"Error loading access token: {str(e)}")
        return None

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                logging.info(f"Port {port} is available")
                return port
            except OSError as e:
                logging.warning(f"Port {port} unavailable: {str(e)}")
                port += 1
    raise OSError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

def retry_request(func, *args, max_retries=MAX_RETRIES, backoff_factor=2, **kwargs):
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logging.warning(f"Rate limit hit, retrying after {retry_after} seconds")
                time.sleep(retry_after)
                continue
            elif response.status_code == 403:
                st.error("403 Forbidden: Check API token permissions")
                logging.error("403 Forbidden error")
                return None
            elif response.status_code == 404:
                st.warning("404 Not Found: Resource may not exist")
                logging.warning("404 Not Found error")
                return response
            elif response.status_code >= 500:
                logging.warning(f"Server error {response.status_code}, retrying...")
                time.sleep(backoff_factor ** attempt)
                continue
            else:
                st.error(f"Request failed with status {response.status_code}")
                logging.error(f"Request failed with status {response.status_code}")
                return response
        except requests.RequestException as e:
            logging.error(f"Request error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
            time.sleep(RATE_LIMIT_DELAY)
        else:
            break
    return results

def get_access_token():
    global ACCESS_TOKEN
    # Try loading from file first
    ACCESS_TOKEN = load_access_token()
    if ACCESS_TOKEN:
        st.session_state.access_token = ACCESS_TOKEN
        logging.info("Using access token from file")
        # Validate token
        if get_account_info(ACCESS_TOKEN):
            return ACCESS_TOKEN
        else:
            logging.warning("Access token invalid, fetching new one")
            os.remove(TOKEN_FILE)
            ACCESS_TOKEN = None
            st.session_state.access_token = None

    try:
        port = find_available_port()
        AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
        
        class OAuthHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                logging.info(f"Received request: {self.path}")
                if self.path.startswith('/oauth/callback'):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    code = params.get('code', [None])[0]
                    logging.info(f"Authorization code: {code}")
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
                            global ACCESS_TOKEN
                            ACCESS_TOKEN = token_data.get("access_token")
                            logging.info(f"Access token obtained: {ACCESS_TOKEN}")
                            st.session_state.access_token = ACCESS_TOKEN
                            save_access_token(ACCESS_TOKEN)
                            self.respond_with("Success! You can close this tab.")
                        else:
                            logging.error(f"Token exchange failed: {token_response.status_code if token_response else 'No response'}")
                            self.respond_with("Token exchange failed.")
                    else:
                        logging.error("No code found in callback URL")
                        self.respond_with("No code found in callback URL")
                else:
                    logging.error(f"Invalid callback path: {self.path}")
                    self.respond_with("Invalid callback URL")
            
            def respond_with(self, message):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())
        
        st.info(f"Opening browser for authorization on port {port}...")
        logging.info(f"Starting OAuth flow on port {port}")
        webbrowser.open(AUTH_URL)
        # Start server and wait for a single request
        with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
            logging.info(f"Starting HTTP server on port {port}")
            httpd.timeout = 120  # Wait up to 120 seconds for callback
            httpd.handle_request()
            logging.info("HTTP server handled request")
        
        if ACCESS_TOKEN:
            logging.info("Access token obtained successfully")
            return ACCESS_TOKEN
        else:
            logging.error("Failed to obtain access token during OAuth flow")
            st.error("Failed to obtain access token")
            return None
    except Exception as e:
        logging.error(f"Failed to obtain access token: {str(e)}")
        st.error(f"Failed to obtain access token: {str(e)}")
        return None

def get_account_info(access_token: str) -> Optional[int]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        if "accounts" in data and data["accounts"]:
            return data["accounts"][0]["id"]
    logging.error("Failed to fetch account ID")
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_projects(account_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    project_list = []
    for project in projects:
        todoset_id = None
        for dock in project.get("dock", []):
            if dock["name"] == "todoset" and dock["enabled"]:
                todoset_id = dock["id"]
        project_list.append((project['name'], project['id'], todoset_id, project.get('updated_at')))
    return project_list

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todosets/{todoset_id}.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.ok:
        todoset = response.json()
        todolists_url = todoset.get("todolists_url", "")
        todolists = get_paginated_results(todolists_url, headers)
        if todolists:
            return [(todolist['title'], todolist['id'], todolist.get('updated_at')) for todolist in todolists]
        else:
            fallback_url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists.json"
            todolists = get_paginated_results(fallback_url, headers)
            return [(todolist['title'], todolist['id'], todolist.get('updated_at')) for todolist in todolists]
    return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_tasks(account_id: int, project_id: int, todolist_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
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
        if not task_response or task_response.status_code != 200:
            continue
        task_info = {
            "title": task.get("title", "N/A"),
            "status": task.get("status", "N/A"),
            "due_on": task.get("due_on", "N/A"),
            "id": task_id,
            "assignee": task.get("assignee", {}).get("name", "Unassigned"),
            "assignee_id": task.get("assignee", {}).get("id", "N/A"),
            "creator": task.get("creator", {}).get("name", "Unknown"),
            "comments": get_task_comments(account_id, project_id, task_id, access_token),
            "updated_at": task.get("updated_at", "N/A")
        }
        task_list.append(task_info)
        time.sleep(RATE_LIMIT_DELAY)
    return task_list

def get_task_comments(account_id: int, project_id: int, task_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    comment_list = []
    if response and response.status_code == 200:
        comments = response.json()
        for comment in comments:
            comment_list.append({
                "content": comment.get("content", "N/A").strip(),
                "created_at": comment.get("created_at", "N/A"),
                "id": comment.get("id", "N/A"),
                "creator": comment.get("creator", {}).get("name", "N/A"),
                "creator_id": comment.get("creator", {}).get("id", "N/A")
            })
    return comment_list

def get_new_comments(account_id: int, project_id: int, task_id: int, access_token: str, existing_comments: List[Dict]) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    new_comments = []
    if response and response.status_code == 200:
        comments = response.json()
        existing_comment_ids = {comment['id'] for comment in existing_comments}
        for comment in comments:
            if comment.get("id") not in existing_comment_ids:
                new_comments.append({
                    "content": comment.get("content", "N/A").strip(),
                    "created_at": comment.get("created_at", "N/A"),
                    "id": comment.get("id", "N/A"),
                    "creator": comment.get("creator", {}).get("name", "N/A"),
                    "creator_id": comment.get("creator", {}).get("id", "N/A")
                })
    return new_comments

def post_comment(account_id: int, project_id: int, recording_id: int, access_token: str, content: str) -> bool:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (raey.abebe@ienetworksolutions.com)",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{recording_id}/comments.json"
    data = {"content": content}
    response = retry_request(
        requests.post,
        url,
        headers=headers,
        json=data,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.status_code == 201:
        logging.info(f"Comment posted successfully for recording {recording_id}")
        return True
    else:
        logging.error(f"Failed to post comment: {response.status_code if response else 'No response'}")
        return False


def generate_smart_reply(task: Dict, comments: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    You are a smart assistant generating a reply for a Basecamp task comment.
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {json.dumps(comments, indent=2)}
    
    Generate a concise, professional reply to the latest comment or task context. The reply should:
    - Acknowledge the latest comment (if any) or task status.
    - Provide a relevant update, question, or action item.
    - Be suitable for posting directly to Basecamp.
    """
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No reply generated")
        else:
            logging.error(f"Gemini API error: {response.status_code} - {response.text}")
            return "Failed to generate smart reply"
    except Exception as e:
        logging.error(f"Gemini API request failed: {str(e)}")
        return f"Error generating reply: {str(e)}"

def fetch_gemini_insights(task: Dict, comments: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    You are a smart assistant analyzing a Basecamp task and its comments.
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {json.dumps(comments, indent=2)}
    
    Provide insights and automation suggestions based on the task and comments. For example:
    - Summarize the task and comment context.
    - Suggest automation actions (e.g., reminders, status updates).
    - Identify potential issues or delays.
    - Offer recommendations for task completion.
    """
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights available")
        else:
            logging.error(f"Gemini API error: {response.status_code} - {response.text}")
            return "Failed to fetch insights from Gemini API"
    except Exception as e:
        logging.error(f"Gemini API request failed: {str(e)}")
        return f"Error fetching insights: {str(e)}"

def save_data(data: List[Dict]):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return DATA_FILE

def load_data() -> List[Dict]:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def is_data_up_to_date(account_id: int, access_token: str, existing_data: List[Dict]) -> bool:
    """Check if existing data is up-to-date by comparing project and task timestamps."""
    if not existing_data:
        return False
    projects = get_projects(account_id, access_token)
    for project_name, project_id, _, project_updated_at in projects:
        existing_project = next((p for p in existing_data if p['project_id'] == project_id), None)
        if not existing_project or existing_project.get('updated_at') != project_updated_at:
            return False
        for todolist in existing_project.get('todolists', []):
            for task in todolist.get('tasks', []):
                task_response = retry_request(
                    requests.get,
                    f"{BASE_URL}/{account_id}/buckets/{project_id}/todos/{task['id']}.json",
                    headers={"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAPIClient"},
                    timeout=REQUEST_TIMEOUT
                )
                if task_response and task_response.ok:
                    if task_response.json().get('updated_at') != task.get('updated_at'):
                        return False
    return True

def check_new_data(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None, selected_todolist: str = None) -> tuple:
    start_time = time.time()
    new_data = []
    updated = False
    projects = get_projects(account_id, access_token)
    
    for project_name, project_id, todoset_id, project_updated_at in projects:
        if selected_project and project_name != selected_project:
            existing_project = next((p for p in existing_data if p['project_id'] == project_id), None)
            if existing_project:
                new_data.append(existing_project)
            continue
        
        existing_project = next((p for p in existing_data if p['project_id'] == project_id), None)
        project_data = existing_project or {
            "project_name": project_name,
            "project_id": project_id,
            "updated_at": project_updated_at,
            "todolists": []
        }
        
        if not existing_project or existing_project.get('updated_at') != project_updated_at:
            if todoset_id:
                todolists = get_todoset(account_id, project_id, todoset_id, access_token)
                for todolist_name, todolist_id, todolist_updated_at in todolists:
                    if selected_todolist and todolist_name != selected_todolist:
                        existing_todolist = None
                        if existing_project:
                            existing_todolist = next((t for t in existing_project['todolists'] if t['todolist_id'] == todolist_id), None)
                        if existing_todolist:
                            project_data["todolists"].append(existing_todolist)
                        continue
                    
                    existing_todolist = None
                    if existing_project:
                        existing_todolist = next((t for t in existing_project['todolists'] if t['todolist_id'] == todolist_id), None)
                    
                    if not existing_todolist or existing_todolist.get('updated_at') != todolist_updated_at:
                        tasks = get_tasks(account_id, project_id, todolist_id, access_token)
                        project_data["todolists"].append({
                            "todolist_name": todolist_name,
                            "todolist_id": todolist_id,
                            "updated_at": todolist_updated_at,
                            "tasks": tasks
                        })
                        if selected_project == project_name and (selected_todolist is None or selected_todolist == todolist_name):
                            updated = True
                    elif existing_todolist:
                        project_data["todolists"].append(existing_todolist)
            new_data.append(project_data)
        else:
            new_data.append(existing_project)
    
    logging.info(f"Data check completed in {time.time() - start_time:.2f} seconds")
    return new_data, updated

# Streamlit UI
st.set_page_config(page_title="Basecamp AI Assistant", layout="wide")

def main():
    # Navbar
    st.markdown("""
    <style>
    .navbar {
        background-color: #1a73e8;
        padding: 10px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    </style>
    <div class="navbar">Basecamp AI Assistant</div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Tasks", "Settings"])

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    if 'access_token' not in st.session_state:
        st.session_state.access_token = load_access_token()
    if 'account_id' not in st.session_state:
        st.session_state.account_id = None
    if 'data_up_to_date' not in st.session_state:
        st.session_state.data_up_to_date = False

    # Load access token at startup
    if not st.session_state.access_token:
        st.session_state.access_token = load_access_token()
        if st.session_state.access_token:
            account_id = get_account_info(st.session_state.access_token)
            if account_id:
                st.session_state.account_id = account_id
            else:
                st.session_state.access_token = None
                os.remove(TOKEN_FILE)

    if page == "Dashboard":
        st.header("Dashboard")
        st.write("Welcome to the Basecamp AI Assistant. Use this app to fetch Basecamp project data and generate AI insights.")
        st.subheader("Workflow Steps")
        st.markdown("""
        1. **Fetch Data**: Go to the Settings page to authenticate with Basecamp and fetch project data.
        2. **View Tasks**: Navigate to the Tasks page to select a project, to-do list, and task.
        3. **Fetch New Comments**: On the Tasks page, click 'Fetch New Comments' to retrieve new task comments.
        4. **Generate Insights**: Click 'Generate Insights' to get AI recommendations from the Gemini API.
        5. **Smart Reply**: Use the 'Generate Smart Reply' button to create and post AI-generated replies.
        6. **Refresh Data**: Use the 'Force Refresh Data' button on the Tasks page to check for updates.
        """)
        if st.session_state.data:
            st.subheader("Data Summary")
            project_count = len(st.session_state.data)
            task_count = sum(len(todolist['tasks']) for project in st.session_state.data for todolist in project.get('todolists', []))
            comment_count = sum(len(task['comments']) for project in st.session_state.data for todolist in project.get('todolists', []) for task in todolist.get('tasks', []))
            st.write(f"Projects: {project_count}")
            st.write(f"Tasks: {task_count}")
            st.write(f"Comments: {comment_count}")
        else:
            st.warning("No data loaded. Fetch data from the Settings page.")

    elif page == "Tasks":
        st.header("Tasks")
        if not st.session_state.data:
            st.warning("No data available. Please fetch data from the Settings page.")
            st.markdown("[Go to Settings](#)")
            return

        # Sidebar for project and task selection
        project_names = [project['project_name'] for project in st.session_state.data]
        selected_project = st.sidebar.selectbox("Select Project", project_names, key="project_select_tasks")
        project = next((p for p in st.session_state.data if p['project_name'] == selected_project), None)
        
        if project:
            todolist_names = [todolist['todolist_name'] for todolist in project.get('todolists', [])]
            selected_todolist = st.sidebar.selectbox("Select To-do List", todolist_names, key="todolist_select_tasks")
            todolist = next((t for t in project['todolists'] if t['todolist_name'] == selected_todolist), None)
            
            # Check for new data only if not up-to-date or forced refresh
            if st.session_state.access_token and st.session_state.account_id:
                if st.sidebar.button("Force Refresh Data"):
                    with st.spinner("Checking for new data..."):
                        new_data, updated = check_new_data(
                            st.session_state.account_id,
                            st.session_state.access_token,
                            st.session_state.data,
                            selected_project,
                            selected_todolist
                        )
                        st.session_state.data = new_data
                        save_data(new_data)
                        st.session_state.data_up_to_date = not updated
                        if updated:
                            st.success(f"New data found for {selected_project}. Data updated in {DATA_FILE}")
                        else:
                            st.info(f"No new data found for {selected_project}.")
                elif not st.session_state.data_up_to_date:
                    with st.spinner("Checking if data is up-to-date..."):
                        st.session_state.data_up_to_date = is_data_up_to_date(
                            st.session_state.account_id,
                            st.session_state.access_token,
                            st.session_state.data
                        )
                        if not st.session_state.data_up_to_date:
                            new_data, updated = check_new_data(
                                st.session_state.account_id,
                                st.session_state.access_token,
                                st.session_state.data,
                                selected_project,
                                selected_todolist
                            )
                            st.session_state.data = new_data
                            save_data(new_data)
                            st.session_state.data_up_to_date = not updated
                            if updated:
                                st.success(f"New data found for {selected_project}. Data updated in {DATA_FILE}")
                            else:
                                st.info(f"No new data found for {selected_project}.")
                        else:
                            st.info("Data is up-to-date. Use 'Force Refresh Data' to check again.")
            
            if todolist:
                tasks = todolist.get('tasks', [])
                if not tasks:
                    st.info("No tasks found in this to-do list.")
                    return
                task_titles = [task['title'] for task in tasks]
                selected_task = st.sidebar.selectbox("Select Task", task_titles, key="task_select_tasks")
                task = next((t for t in tasks if t['title'] == selected_task), None)
                
                if task:
                    st.subheader(f"Task: {task['title']}")
                    st.write(f"**Status**: {task['status']}")
                    st.write(f"**Due Date**: {task['due_on']}")
                    assignee = task.get('assignee', 'Unassigned')
                    st.write(f"**Assignee**: {assignee}")
                    creator = task.get('creator', 'Unknown')
                    st.write(f"**Creator**: {creator}")
                    
                    # Log session state for debugging
                    logging.debug(f"Session state - access_token: {'present' if st.session_state.access_token else 'missing'}, account_id: {'present' if st.session_state.account_id else 'missing'}")
                    
                    # Fetch new comments only
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
                                    save_data(st.session_state.data)
                                    st.success(f"{len(new_comments)} new comments fetched and saved.")
                                else:
                                    st.info("No new comments found.")
                        else:
                            st.error("Access token or account ID missing. Please fetch data from the Settings page.")
                            st.markdown("[Go to Settings](#)")
                            logging.warning("Attempted to fetch comments without access_token or account_id")
                    
                    # Display comments
                    if task['comments']:
                        st.subheader("Comments")
                        for comment in task['comments']:
                            st.markdown(f"**Comment ID: {comment['id']}**")
                            st.write(f"**Creator**: {comment['creator']}")
                            st.write(f"**Created**: {comment['created_at']}")
                            st.write(f"**Content**: {comment['content']}")
                            st.markdown("---")
                    else:
                        st.info("No comments available.")
                    
                    # Smart Reply
                    st.subheader("Smart Reply")
                    if st.button("Generate Smart Reply"):
                        with st.spinner("Generating smart reply..."):
                            reply = generate_smart_reply(task, task['comments'])
                            st.session_state.smart_reply = reply
                            st.markdown(f"**Suggested Reply**: {reply}")
                    
                    if 'smart_reply' in st.session_state and st.button("Post Smart Reply"):
                        if st.session_state.access_token and st.session_state.account_id:
                            with st.spinner("Posting reply..."):
                                success = post_comment(
                                    st.session_state.account_id,
                                    project['project_id'],
                                    task['id'],
                                    st.session_state.access_token,
                                    st.session_state.smart_reply
                                )
                                if success:
                                    st.success("Reply posted successfully!")
                                    # Refresh comments after posting
                                    new_comments = get_new_comments(
                                        st.session_state.account_id,
                                        project['project_id'],
                                        task['id'],
                                        st.session_state.access_token,
                                        task['comments']
                                    )
                                    task['comments'].extend(new_comments)
                                    save_data(st.session_state.data)
                                else:
                                    st.error("Failed to post reply.")
                        else:
                            st.error("Access token or account ID missing.")
                    
                    # Gemini Insights
                    st.subheader("AI Insights")
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = fetch_gemini_insights(task, task['comments'])
                            st.markdown(insights)

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Fetch Basecamp Data")
        st.markdown("""
        Click the button below to authenticate with Basecamp and fetch project data. This will:
        1. Use existing access token if valid, or open a browser for Basecamp authorization.
        2. Save the access token to `access_token.json`.
        3. Fetch projects, to-do lists, tasks, and comments, saving them to `AllData.json`.
        """)
        if st.button("Fetch New Data"):
            with st.spinner("Fetching data from Basecamp..."):
                access_token = get_access_token()
                if access_token:
                    st.session_state.access_token = access_token
                    account_id = get_account_info(access_token)
                    if account_id:
                        st.session_state.account_id = account_id
                        existing_data = load_data()
                        new_data, updated = check_new_data(account_id, access_token, existing_data)
                        st.session_state.data = new_data
                        st.session_state.data_up_to_date = not updated
                        output_file = save_data(new_data)
                        if updated:
                            st.success(f"New data fetched and saved to {output_file}")
                        else:
                            st.info(f"No new data found. Using existing data in {output_file}")
                    else:
                        st.error("Failed to fetch account ID")
                        logging.error("Failed to fetch account ID")
                else:
                    st.error("Failed to obtain access token. Check logs for details.")
                    logging.error("Failed to obtain access token")

if __name__ == "__main__":
    main()