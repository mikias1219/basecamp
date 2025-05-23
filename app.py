import streamlit as st
import requests
import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
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
    filename='basecamp_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG_FILE = "config.json"
BASE_URL = "https://3.basecampapi.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.2
API_KEY = "AIzaSyArWCID8FdgwcFJpS_mUJNlLy6QJhMvf5w"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
DATA_FILE = "AllData.json"
TOKEN_FILE = "access_token.json"
TOKEN_EXPIRY = timedelta(days=1)  # Token valid for 1 day

# Updated configuration
CONFIG = {
    "CLIENT_ID": "6c916f56910da5118a83a2bf9c762d2967c430f0",
    "CLIENT_SECRET": "5b139ebdf83caec06caff04c02180e93e1d28374"
}
CLIENT_ID = CONFIG.get("CLIENT_ID")
CLIENT_SECRET = CONFIG.get("CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/oauth/callback"

def save_access_token(token: str, expiry: datetime):
    """Save access token and expiry to file."""
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"access_token": token, "expiry": expiry.isoformat()}, f)
    logging.info("Access token saved")

def load_access_token() -> Optional[Dict]:
    """Load access token and expiry from file if valid."""
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            expiry_str = data.get("expiry")
            if not isinstance(expiry_str, str):
                logging.error(f"Invalid expiry format in {TOKEN_FILE}: {expiry_str}")
                os.remove(TOKEN_FILE)
                return None
            try:
                expiry = datetime.fromisoformat(expiry_str)
                if datetime.now() < expiry:
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
        logging.info(f"Token file not found or invalid: {str(e)}")
        return None

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
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
                time.sleep(retry_after)
                continue
            elif response.status_code == 403:
                st.error("Invalid API token. Please re-authenticate.")
                return None
            elif response.status_code == 404:
                return response
            elif response.status_code >= 500:
                time.sleep(backoff_factor ** attempt)
                continue
            else:
                st.error(f"Request failed: {response.status_code}")
                return response
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
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
            time.sleep(RATE_LIMIT_DELAY)
        else:
            break
    return results

def get_access_token():
    token_data = load_access_token()
    if token_data:
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
                        save_access_token(access_token, datetime.now() + TOKEN_EXPIRY)
                        st.session_state.access_token = access_token
                        self.respond_with("Success! You can close this tab.")
                    else:
                        self.respond_with("Token exchange failed.")
                else:
                    self.respond_with("No code found in callback URL")
            else:
                self.respond_with("Invalid callback URL")

        def respond_with(self, message):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())

    st.info("Opening browser for Basecamp authorization...")
    webbrowser.open(AUTH_URL)
    with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
        httpd.timeout = 120
        httpd.handle_request()

    return st.session_state.get("access_token")

def get_account_info(access_token: str) -> Optional[int]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        return data["accounts"][0]["id"] if "accounts" in data and data["accounts"] else None
    return None

@st.cache_data(ttl=3600)
def get_projects(account_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    return [(project['name'], project['id'], next((dock["id"] for dock in project.get("dock", []) if dock["name"] == "todoset" and dock["enabled"]), None), project.get('updated_at')) for project in projects]

@st.cache_data(ttl=3600)
def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todosets/{todoset_id}.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.ok:
        todoset = response.json()
        todolists_url = todoset.get("todolists_url", "")
        todolists = get_paginated_results(todolists_url, headers)
        return [(todolist['title'], todolist['id'], todolist.get('updated_at')) for todolist in todolists]
    return []

@st.cache_data(ttl=3600)
def get_tasks(account_id: int, project_id: int, todolist_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
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
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
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
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)"
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

def post_comment(account_id: int, project_id: int, recording_id: int, access_token: str, content: str) -> bool:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (raey.abebe@ienetworksolutions.com)",
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
    return response and response.status_code == 201

def generate_smart_reply(task: Dict, comments: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {json.dumps(comments, indent=2)}
    
    Generate a concise, professional reply to the latest comment or task context.
    - Acknowledge the latest comment (if any) or task status.
    - Provide a relevant update, question, or action item.
    - Be suitable for posting directly to Basecamp.
    """
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No reply generated")
        return "Failed to generate reply"
    except Exception as e:
        return f"Error generating reply: {str(e)}"

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

def is_data_up_to_date(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None) -> bool:
    if not existing_data:
        return False
    projects = get_projects(account_id, access_token)
    for project_name, project_id, _, project_updated_at in projects:
        if selected_project and project_name != selected_project:
            continue
        existing_project = next((p for p in existing_data if p['project_id'] == project_id), None)
        if not existing_project or existing_project.get('updated_at') != project_updated_at:
            return False
        for todolist in existing_project.get('todolists', []):
            for task in todolist.get('tasks', []):
                task_response = retry_request(
                    requests.get,
                    f"{BASE_URL}/{account_id}/buckets/{project_id}/todos/{task['id']}.json",
                    headers={"Authorization": f"Bearer {access_token}", "User-Agent": "BasecampAI"},
                    timeout=REQUEST_TIMEOUT
                )
                if task_response and task_response.ok and task_response.json().get('updated_at') != task.get('updated_at'):
                    return False
    return True

def check_new_data(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None) -> tuple:
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
            updated = True
            if todoset_id:
                todolists = get_todoset(account_id, project_id, todoset_id, access_token)
                for todolist_name, todolist_id, todolist_updated_at in todolists:
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
    # Styling
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
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 10px;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    .comment-box {
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    <div class="navbar">Basecamp AI Assistant</div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'account_id' not in st.session_state:
        st.session_state.account_id = None
    if 'data_up_to_date' not in st.session_state:
        st.session_state.data_up_to_date = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'comments_per_page' not in st.session_state:
        st.session_state.comments_per_page = 5

    # Load access token at startup
    if not st.session_state.access_token:
        st.session_state.access_token = get_access_token()
        if st.session_state.access_token:
            st.session_state.account_id = get_account_info(st.session_state.access_token)
            if not st.session_state.account_id:
                st.session_state.access_token = None
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Tasks", "Settings"], key="nav_radio")

    if page == "Dashboard":
        st.header("Dashboard")
        st.write("Manage your Basecamp projects, tasks, and comments with AI-powered insights.")
        st.subheader("Quick Start")
        st.markdown("""
        1. **Authenticate**: Go to Settings to connect with Basecamp (one-time daily).
        2. **View Tasks**: Check the Tasks page to select projects and tasks.
        3. **Manage Comments**: Fetch new comments or post AI-generated replies.
        4. **Get Insights**: Generate AI insights for tasks and comments.
        """)
        if st.session_state.data:
            project_count = len(st.session_state.data)
            task_count = sum(len(todolist['tasks']) for project in st.session_state.data for todolist in project.get('todolists', []))
            comment_count = sum(len(task['comments']) for project in st.session_state.data for todolist in project.get('todolists', []) for task in todolist.get('tasks', []))
            st.subheader("Data Overview")
            st.write(f"**Projects**: {project_count}")
            st.write(f"**Tasks**: {task_count}")
            st.write(f"**Comments**: {comment_count}")
        else:
            st.info("No data loaded. Please authenticate and fetch data in Settings.")

    elif page == "Tasks":
        st.header("Tasks")
        if not st.session_state.data:
            st.info("No data available. Please fetch data from Settings.")
            st.markdown("[Go to Settings](#settings)")
            return

        # Project and task selection
        project_names = [project['project_name'] for project in st.session_state.data]
        selected_project = st.sidebar.selectbox("Select Project", project_names, key="project_select")
        project = next((p for p in st.session_state.data if p['project_name'] == selected_project), None)
        
        if project:
            todolist_names = [todolist['todolist_name'] for todolist in project.get('todolists', [])]
            selected_todolist = st.sidebar.selectbox("Select To-do List", todolist_names, key="todolist_select")
            todolist = next((t for t in project['todolists'] if t['todolist_name'] == selected_todolist), None)
            
            # Data refresh
            if st.session_state.access_token and st.session_state.account_id:
                if st.sidebar.button("Refresh Data"):
                    with st.spinner("Checking for updates..."):
                        new_data, updated = check_new_data(
                            st.session_state.account_id,
                            st.session_state.access_token,
                            st.session_state.data,
                            selected_project
                        )
                        st.session_state.data = new_data
                        save_data(new_data)
                        st.session_state.data_up_to_date = not updated
                        if updated:
                            st.success(f"Data updated for {selected_project}")
                        else:
                            st.info(f"No updates for {selected_project}")
                elif not st.session_state.data_up_to_date:
                    with st.spinner("Verifying data..."):
                        st.session_state.data_up_to_date = is_data_up_to_date(
                            st.session_state.account_id,
                            st.session_state.access_token,
                            st.session_state.data,
                            selected_project
                        )
                        if not st.session_state.data_up_to_date:
                            new_data, updated = check_new_data(
                                st.session_state.account_id,
                                st.session_state.access_token,
                                st.session_state.data,
                                selected_project
                            )
                            st.session_state.data = new_data
                            save_data(new_data)
                            st.session_state.data_up_to_date = not updated
                            if updated:
                                st.success(f"Data updated for {selected_project}")
                            else:
                                st.info(f"Data is up-to-date for {selected_project}")
            
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
                        start_idx = (st.session_state.current_page - 1) * st.session_state.comments_per_page
                        end_idx = start_idx + st.session_state.comments_per_page
                        paginated_comments = comments[start_idx:end_idx]
                        
                        for comment in paginated_comments:
                            with st.container():
                                st.markdown(f"""
                                <div class="comment-box">
                                    <strong>Creator:</strong> {comment['creator']}<br>
                                    <strong>Posted:</strong> {comment['created_at']}<br>
                                    <strong>Comment:</strong> {comment['content']}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        total_pages = (total_comments + st.session_state.comments_per_page - 1) // st.session_state.comments_per_page
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            if st.session_state.current_page > 1:
                                if st.button("Previous", key="prev_page"):
                                    st.session_state.current_page -= 1
                        with col3:
                            if st.session_state.current_page < total_pages:
                                if st.button("Next", key="next_page"):
                                    st.session_state.current_page += 1
                        with col2:
                            st.write(f"Page {st.session_state.current_page} of {total_pages}")
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
                                    save_data(st.session_state.data)
                                    st.success(f"{len(new_comments)} new comments added.")
                                    st.session_state.current_page = 1
                                else:
                                    st.info("No new comments found.")
                        else:
                            st.error("Please authenticate in Settings.")
                    
                    # Smart Reply
                    st.subheader("Smart Reply")
                    if st.button("Generate Reply"):
                        with st.spinner("Generating reply..."):
                            reply = generate_smart_reply(task, task['comments'])
                            st.session_state.smart_reply = reply
                            st.markdown(f"**Suggested Reply**: {reply}")
                    
                    if 'smart_reply' in st.session_state:
                        st.text_area("Edit Reply", value=st.session_state.smart_reply, key="edit_reply")
                        if st.button("Post Reply"):
                            with st.spinner("Posting reply..."):
                                success = post_comment(
                                    st.session_state.account_id,
                                    project['project_id'],
                                    task['id'],
                                    st.session_state.access_token,
                                    st.session_state.edit_reply
                                )
                                if success:
                                    st.success("Reply posted!")
                                    new_comments = get_new_comments(
                                        st.session_state.account_id,
                                        project['project_id'],
                                        task['id'],
                                        st.session_state.access_token,
                                        task['comments']
                                    )
                                    task['comments'].extend(new_comments)
                                    save_data(st.session_state.data)
                                    st.session_state.current_page = 1
                                else:
                                    st.error("Failed to post reply.")
                    
                    # Insights
                    st.subheader("AI Insights")
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = fetch_gemini_insights(task, task['comments'])
                            st.markdown(insights)

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Basecamp Authentication")
        st.markdown("Authenticate with Basecamp to fetch project data. The access token is valid for 1 day.")
        if st.button("Authenticate and Fetch Data"):
            with st.spinner("Authenticating..."):
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
                        save_data(new_data)
                        if updated:
                            st.success("Data fetched and saved.")
                        else:
                            st.info("Data is up-to-date.")
                    else:
                        st.error("Failed to fetch account ID.")
                else:
                    st.error("Authentication failed.")

if __name__ == "__main__":
    main()