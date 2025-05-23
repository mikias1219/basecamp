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
BASE_URL = "https://3.basecampapi.com"  # Adjust if Basecamp 4 uses a different endpoint
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.2
DATA_FILE = "AllData.json"
TOKEN_EXPIRY = timedelta(days=1)

# Load sensitive data from Streamlit secrets
CLIENT_ID = st.secrets.get("CLIENT_ID", "9e371d51daeb978e7a93cb8183c722aa848a2f9c")
CLIENT_SECRET = st.secrets.get("CLIENT_SECRET", "76b8f666f5d68d7f5b43e02f8ca2ad27e78a02ff")
API_KEY = st.secrets.get("API_KEY", "AIzaSyArWCID8FdgwcFJpS_mUJNlLy6QJhMvf5w")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
REDIRECT_URI = "https://basecampapp.streamlit.app/oauth/callback"

def save_access_token(user_id: str, token: str, expiry: datetime, account_id: Optional[int] = None):
    """Save access token and expiry to session state."""
    st.session_state.access_tokens[user_id] = {
        "access_token": token,
        "expiry": expiry.isoformat(),
        "account_id": account_id
    }
    logging.info(f"Access token saved for user {user_id}")

def load_access_token(user_id: str) -> Optional[Dict]:
    """Load access token and expiry from session state if valid."""
    token_data = st.session_state.access_tokens.get(user_id)
    if token_data and isinstance(token_data.get("expiry"), str):
        try:
            expiry = datetime.fromisoformat(token_data["expiry"])
            if datetime.now() < expiry:
                return {
                    "access_token": token_data["access_token"],
                    "expiry": expiry,
                    "account_id": token_data.get("account_id")
                }
            else:
                logging.info(f"Access token expired for user {user_id}")
                del st.session_state.access_tokens[user_id]
                return None
        except ValueError as e:
            logging.error(f"Failed to parse expiry for user {user_id}: {str(e)}")
            return None
    return None

def retry_request(func, *args, max_retries=MAX_RETRIES, backoff_factor=2, **kwargs):
    """Retry API requests with exponential backoff."""
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
    """Fetch paginated results from Basecamp API."""
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

def handle_oauth_callback():
    """Handle OAuth callback and exchange code for access token."""
    query_params = st.query_params
    code = query_params.get("code")
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
            user_id = st.session_state.user_id
            account_id = get_account_info(user_id, access_token)
            save_access_token(user_id, access_token, datetime.now() + TOKEN_EXPIRY, account_id)
            st.success("Authentication successful! You can now access your Basecamp data.")
            st.query_params.clear()
        else:
            st.error("Token exchange failed.")
    else:
        st.error("No authorization code found in callback URL.")

def authenticate_basecamp():
    """Initiate Basecamp OAuth flow."""
    user_id = st.session_state.user_id
    existing_token = load_access_token(user_id)
    if existing_token:
        st.session_state.access_tokens[user_id]["access_token"] = existing_token["access_token"]
        st.session_state.access_tokens[user_id]["account_id"] = existing_token["account_id"]
        st.success("Using existing valid token.")
        return
    auth_url = (
        f"https://launchpad.37signals.com/authorization/new"
        f"?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    )
    st.markdown(f"[Click here to authenticate with Basecamp]({auth_url})")

def get_account_info(user_id: str, access_token: str) -> Optional[int]:
    """Fetch Basecamp account ID."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        account_id = data["accounts"][0]["id"] if "accounts" in data and data["accounts"] else None
        if account_id:
            st.session_state.access_tokens[user_id]["account_id"] = account_id
        return account_id
    return None

def get_user_info(user_id: str, access_token: str) -> Dict:
    """Fetch Basecamp user profile information."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        return data.get("identity", {})
    return {}

@st.cache_data(ttl=3600)
def get_projects(account_id: int, access_token: str) -> List[tuple]:
    """Fetch Basecamp projects."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    return [(project['name'], project['id'], next((dock["id"] for dock in project.get("dock", []) if dock["name"] == "todoset" and dock["enabled"]), None), project.get('updated_at')) for project in projects]

@st.cache_data(ttl=3600)
def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[tuple]:
    """Fetch to-do lists for a project."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
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
    """Fetch tasks for a to-do list."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
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
    """Fetch comments for a task."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
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
    """Fetch new comments since last check."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)"
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
    """Post a comment to a task."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAI (support@ienetworksolutions.com)",
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
    """Generate a smart reply using Gemini API."""
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
    """Fetch insights using Gemini API."""
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

def save_data(user_id: str, data: List[Dict]):
    """Save user-specific data to file."""
    all_data = load_all_data()
    all_data[user_id] = data
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    return DATA_FILE

def load_all_data() -> Dict:
    """Load all user data from file."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logging.warning(f"Invalid data format in {DATA_FILE}: expected dict, got {type(data)}. Returning empty dict.")
                return {}
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.info(f"Failed to load {DATA_FILE}: {str(e)}. Returning empty dict.")
        return {}

def load_data(user_id: str) -> List[Dict]:
    """Load user-specific data."""
    return load_all_data().get(user_id, [])

def is_data_up_to_date(account_id: int, access_token: str, existing_data: List[Dict], selected_project: str = None) -> bool:
    """Check if data is up-to-date."""
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
    """Check for new or updated data."""
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
st.set_page_config(page_title="Basecamp AI Assistant by IE", layout="wide")

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
    .navbar a {
        color: white;
        margin: 0 15px;
        text-decoration: none;
        font-size: 18px;
    }
    .navbar a:hover {
        text-decoration: underline;
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
    <div class="navbar">
        Basecamp AI Assistant by IE
        <div style="font-size: 18px; margin-top: 10px;">
            <a href="?page=dashboard">Dashboard</a>
            <a href="?page=tasks">Tasks</a>
            <a href="?page=profile">Profile</a>
            <a href="?page=settings">Settings</a>
            <a href="?page=activity">Recent Activity</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if 'access_tokens' not in st.session_state:
        st.session_state.access_tokens = {}
    if 'data' not in st.session_state:
        st.session_state.data = load_data(st.session_state.user_id)
    if 'data_up_to_date' not in st.session_state:
        st.session_state.data_up_to_date = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'comments_per_page' not in st.session_state:
        st.session_state.comments_per_page = 5

    # Handle OAuth callback
    query_params = st.query_params
    if "code" in query_params:
        handle_oauth_callback()
        return

    # Navigation
    page = query_params.get("page", "Dashboard")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Tasks", "Profile", "Settings", "Recent Activity"],
        index=["Dashboard", "Tasks", "Profile", "Settings", "Recent Activity"].index(page),
        format_func=lambda x: f"{x} - {'Overview of projects and tasks' if x == 'Dashboard' else 'Manage tasks and comments' if x == 'Tasks' else 'View user info' if x == 'Profile' else 'Authenticate and configure' if x == 'Settings' else 'View recent updates'}"
    )

    user_id = st.session_state.user_id
    token_data = load_access_token(user_id)
    access_token = token_data["access_token"] if token_data else None
    account_id = token_data["account_id"] if token_data else None

    if page == "Dashboard":
        st.header("Dashboard")
        st.write("Manage your Basecamp projects, tasks, and comments with AI-powered insights from IE.")
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
        if not access_token or not account_id:
            st.error("Please authenticate in Settings.")
            st.markdown("[Go to Settings](#settings)")
            return
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
            if st.sidebar.button("Refresh Data"):
                with st.spinner("Checking for updates..."):
                    new_data, updated = check_new_data(
                        account_id,
                        access_token,
                        st.session_state.data,
                        selected_project
                    )
                    st.session_state.data = new_data
                    save_data(user_id, new_data)
                    st.session_state.data_up_to_date = not updated
                    if updated:
                        st.success(f"Data updated for {selected_project}")
                    else:
                        st.info(f"No updates for {selected_project}")
            elif not st.session_state.data_up_to_date:
                with st.spinner("Verifying data..."):
                    st.session_state.data_up_to_date = is_data_up_to_date(
                        account_id,
                        access_token,
                        st.session_state.data,
                        selected_project
                    )
                    if not st.session_state.data_up_to_date:
                        new_data, updated = check_new_data(
                            account_id,
                            access_token,
                            st.session_state.data,
                            selected_project
                        )
                        st.session_state.data = new_data
                        save_data(user_id, new_data)
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
                        with st.spinner("Fetching new comments..."):
                            new_comments = get_new_comments(
                                account_id,
                                project['project_id'],
                                task['id'],
                                access_token,
                                task['comments']
                            )
                            if new_comments:
                                task['comments'].extend(new_comments)
                                save_data(user_id, st.session_state.data)
                                st.success(f"{len(new_comments)} new comments added.")
                                st.session_state.current_page = 1
                            else:
                                st.info("No new comments found.")
                    
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
                                    account_id,
                                    project['project_id'],
                                    task['id'],
                                    access_token,
                                    st.session_state.edit_reply
                                )
                                if success:
                                    st.success("Reply posted!")
                                    new_comments = get_new_comments(
                                        account_id,
                                        project['project_id'],
                                        task['id'],
                                        access_token,
                                        task['comments']
                                    )
                                    task['comments'].extend(new_comments)
                                    save_data(user_id, st.session_state.data)
                                    st.session_state.current_page = 1
                                else:
                                    st.error("Failed to post reply.")
                    
                    # Insights
                    st.subheader("AI Insights")
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = fetch_gemini_insights(task, task['comments'])
                            st.markdown(insights)

    elif page == "Profile":
        st.header("User Profile")
        if token_data:
            user_info = get_user_info(user_id, access_token)
            st.write(f"**User ID**: {user_id}")
            st.write(f"**Name**: {user_info.get('name', 'N/A')}")
            st.write(f"**Email**: {user_info.get('email_address', 'N/A')}")
            st.write(f"**Account ID**: {account_id or 'N/A'}")
            st.write(f"**Token Expiry**: {token_data['expiry']}")
            if st.button("Log Out"):
                del st.session_state.access_tokens[user_id]
                st.session_state.data = []
                st.session_state.data_up_to_date = False
                st.success("Logged out successfully.")
        else:
            st.info("Not authenticated. Please go to Settings to connect with Basecamp.")

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Basecamp Authentication")
        st.markdown("Authenticate with Basecamp to fetch project data. The access token is valid for 1 day.")
        if st.button("Authenticate and Fetch Data"):
            with st.spinner("Authenticating..."):
                authenticate_basecamp()
                if token_data and account_id:
                    existing_data = load_data(user_id)
                    new_data, updated = check_new_data(account_id, access_token, existing_data)
                    st.session_state.data = new_data
                    st.session_state.data_up_to_date = not updated
                    save_data(user_id, new_data)
                    if updated:
                        st.success("Data fetched and saved.")
                    else:
                        st.info("Data is up-to-date.")

    elif page == "Recent Activity":
        st.header("Recent Activity")
        if not st.session_state.data:
            st.info("No data available. Please fetch data from Settings.")
            st.markdown("[Go to Settings](#settings)")
            return
        for project in st.session_state.data:
            for todolist in project.get("todolists", []):
                for task in todolist.get("tasks", []):
                    for comment in task.get("comments", []):
                        st.markdown(
                            f"**{project['project_name']} > {todolist['todolist_name']} > {task['title']}**: "
                            f"{comment['content']} (by {comment['creator']} at {comment['created_at']})"
                        )

if __name__ == "__main__":
    main()