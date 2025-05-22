import requests
import webbrowser
import http.server
import socketserver
import urllib.parse
import json
import time
import logging
import re
import socket
from typing import List, Dict, Optional
from nltk.tokenize import sent_tokenize
import nltk
from datetime import datetime
import os
from tqdm import tqdm  # For progress bars

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    filename='basecamp_api_robust.log',
    level=logging.DEBUG,  # More verbose for debugging
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG_FILE = "config.json"
BASE_URL = "https://3.basecampapi.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 0.2  # Basecamp recommends 1 request every 0.2 seconds
DEBUG_MODE = True

# Load configuration from config.json
def load_config() -> Dict:
    try:
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"{CONFIG_FILE} not found. Create it with CLIENT_ID and CLIENT_SECRET.")
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        required_keys = ["CLIENT_ID", "CLIENT_SECRET"]
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValueError(f"Missing or empty {key} in {CONFIG_FILE}")
        return config
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Config loading error: {str(e)}")
        print(f"❌ Config error: {str(e)}")
        print(f"Create a {CONFIG_FILE} with the following format:")
        print('{"CLIENT_ID": "your_client_id", "CLIENT_SECRET": "your_client_secret"}')
        exit(1)

CONFIG = load_config()
CLIENT_ID = CONFIG["CLIENT_ID"]
CLIENT_SECRET = CONFIG["CLIENT_SECRET"]

# Global variable to store access token
ACCESS_TOKEN = None

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
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
    """Retry a request with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 5))
                logging.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                print(f"⚠️ Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            elif response.status_code == 403:
                logging.error(f"403 Forbidden: Check API token permissions or account access")
                print(f"❌ 403 Forbidden: Check if your API token has permission to access this resource")
                return None
            elif response.status_code == 404:
                logging.error(f"404 Not Found: {response.text}")
                print(f"⚠️ 404 Not Found: Resource may not exist or is inaccessible")
                return response
            elif response.status_code >= 500:
                logging.error(f"Server error {response.status_code}: {response.text}")
                print(f"⚠️ Server error {response.status_code}: Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor ** attempt)
                    continue
            else:
                logging.error(f"Request failed with status {response.status_code}: {response.text}")
                print(f"❌ Request failed with status {response.status_code}: {response.text[:500]}...")
                return response
        except requests.Timeout:
            logging.error(f"Attempt {attempt + 1} timed out after {REQUEST_TIMEOUT} seconds")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
        except requests.ConnectionError:
            logging.error(f"Attempt {attempt + 1} failed due to connection error")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed with exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
    logging.error(f"Failed after {max_retries} attempts")
    return None

def get_paginated_results(url: str, headers: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """Fetch paginated results from an API endpoint."""
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
            time.sleep(RATE_LIMIT_DELAY)  # Respect rate limits
        else:
            if response and response.status_code == 404:
                logging.info(f"404 Not Found for {url}. Resource may not exist.")
            else:
                logging.error(f"Failed to fetch paginated results from {url}: {response.text if response else 'No response'}")
            break
    return results

def get_access_token():
    """Initiate OAuth flow and retrieve access token."""
    try:
        port = find_available_port()
        REDIRECT_URI = f"http://localhost:{port}/oauth/callback"
        AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
        
        class OAuthHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/oauth/callback'):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    code = params.get('code', [None])[0]
                    
                    if code:
                        logging.info(f"Authorization code received: {code}")
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
                            if ACCESS_TOKEN:
                                logging.info("Access token obtained successfully")
                                print("✅ Access Token obtained")
                                self.respond_with("Success! You can close this tab.")
                            else:
                                logging.error("No access token in response")
                                print("❌ No access token received")
                                self.respond_with("Failed to obtain access token")
                        else:
                            logging.error(f"Token exchange failed: {token_response.text if token_response else 'No response'}")
                            print(f"❌ Token exchange failed: {token_response.text if token_response else 'No response'}")
                            self.respond_with("Token exchange failed. Check your terminal.")
                    else:
                        logging.error("No code found in callback URL")
                        print("❌ No code found in callback URL")
                        self.respond_with("No code found in callback URL")
                
            def respond_with(self, message):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())
        
        print(f"🔗 Opening browser for authorization on port {port}...")
        webbrowser.open(AUTH_URL)
        
        print("🌐 Waiting for authorization...")
        with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
            httpd.handle_request()
        
        return ACCESS_TOKEN
    except OSError as e:
        logging.error(f"Failed to start local server: {str(e)}")
        print(f"❌ Failed to start local server: {str(e)}")
        print("Try closing other applications using the port or restarting your computer.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in OAuth flow: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        return None

def get_account_info(access_token: str) -> Optional[int]:
    """Fetch account information and return account ID."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
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
            account_id = data["accounts"][0]["id"]
            identity = data.get("identity", {})
            full_name = f"{identity.get('first_name', '')} {identity.get('last_name', '')}".strip() or "N/A"
            print(f"✅ Account ID: {account_id}, Name: {full_name}, Email: {identity.get('email_address', 'N/A')}")
            logging.info(f"Account info retrieved: ID={account_id}, Name={full_name}, Email={identity.get('email_address', 'N/A')}")
            return account_id
        else:
            print("❌ No accounts found in response")
            logging.error("No accounts found in response")
            return None
    else:
        print(f"❌ Error getting account info: {response.status_code if response else 'No response'}")
        logging.error(f"Failed to get account info: {response.text if response else 'No response'}")
        return None

def get_projects(account_id: int, access_token: str) -> List[tuple]:
    """Fetch all projects for the account and their to-do set IDs."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    
    if projects:
        print("✅ Projects Found:")
        project_list = []
        for project in projects:
            todoset_id = None
            for dock in project.get("dock", []):
                if dock["name"] == "todoset" and dock["enabled"]:
                    todoset_id = dock["id"]
            print(f"- {project['name']} (ID: {project['id']}, Todoset ID: {todoset_id or 'N/A'})")
            project_list.append((project['name'], project['id'], todoset_id))
        logging.info(f"Retrieved {len(projects)} projects")
        return project_list
    else:
        print("❌ Error fetching projects: No response")
        logging.error("Failed to fetch projects: No response")
        return []

def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[tuple]:
    """Fetch to-do set details for a project with pagination."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todosets/{todoset_id}.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    
    if response and response.ok:
        todoset = response.json()
        todolists_count = todoset.get("todolists_count", 0)
        todolists_url = todoset.get("todolists_url", "")
        print(f"\nℹ️ To-do Set ID {todoset_id} has {todolists_count} to-do list(s)")
        logging.info(f"To-do set {todoset_id} has {todolists_count} to-do lists, URL: {todolists_url}")
        
        if todolists_count == 0:
            print(f"ℹ️ No To-do Lists found in To-do Set ID {todoset_id}")
            logging.info(f"No to-do lists found in to-do set {todoset_id}")
            return []
        
        todolists = get_paginated_results(todolists_url, headers)
        if todolists:
            print(f"\n✅ To-do Lists for Project ID {project_id}:")
            for todolist in todolists:
                print(f"- {todolist['title']} (ID: {todolist['id']})")
            logging.info(f"Retrieved {len(todolists)} to-do lists for to-do set {todoset_id}")
            return [(todolist['title'], todolist['id']) for todolist in todolists]
        else:
            fallback_url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists.json"
            print(f"\n⚠️ Trying fallback to-do lists endpoint: {fallback_url}")
            todolists = get_paginated_results(fallback_url, headers)
            if todolists:
                print(f"\n✅ To-do Lists for Project ID {project_id} (via fallback):")
                for todolist in todolists:
                    print(f"- {todolist['title']} (ID: {todolist['id']})")
                logging.info(f"Retrieved {len(todolists)} to-do lists via fallback")
                return [(todolist['title'], todolist['id']) for todolist in todolists]
            else:
                print(f"❌ Error fetching to-do lists from fallback {fallback_url}")
                logging.error(f"Failed to fetch to-do lists from fallback {fallback_url}")
                return []
    else:
        print(f"❌ Error fetching to-do set: {response.status_code if response else 'No response'}")
        logging.error(f"Failed to fetch to-do set {todoset_id}: {response.text if response else 'No response'}")
        return []

def get_task_comments(account_id: int, project_id: int, task_id: int, headers: Dict) -> List[Dict]:
    """Fetch detailed comments for a specific task using the recordings endpoint."""
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    logging.info(f"Fetching comments for Task ID {task_id} from {url}")
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    comment_list = []

    if response:
        logging.debug(f"Comments endpoint response for Task ID {task_id}: Status {response.status_code}, Headers: {response.headers}, Body: {response.text[:500]}")
        if response.status_code == 200:
            try:
                comments = response.json()
                if comments:
                    for comment in comments:
                        comment_info = {
                            "content": comment.get("content", "N/A").strip(),
                            "created_at": comment.get("created_at", "N/A"),
                            "id": comment.get("id", "N/A"),
                            "creator": comment.get("creator", {}).get("name", "N/A"),
                            "creator_id": comment.get("creator", {}).get("id", "N/A")
                        }
                        comment_list.append(comment_info)
                        print(f"    💬 Comment ID: {comment_info['id']}, Creator: {comment_info['creator']}, Content: {comment_info['content'][:100]}...")
                        logging.info(f"Comment retrieved for Task ID {task_id}: ID={comment_info['id']}, Creator={comment_info['creator']}")
                else:
                    print(f"    ℹ️ No comments found for Task ID {task_id} (empty response)")
                    logging.info(f"No comments returned for Task ID {task_id} (empty JSON response)")
            except requests.exceptions.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON for Task ID {task_id}: {str(e)}, Response: {response.text[:500]}")
                print(f"    ⚠️ Failed to parse comments response for Task ID {task_id} (Status {response.status_code}): {response.text[:500]}...")
        elif response.status_code == 403:
            logging.error(f"403 Forbidden for Task ID {task_id}: Check API token permissions for comments")
            print(f"    ❌ 403 Forbidden for Task ID {task_id}: API token may lack permission to access comments")
            print(f"    Test manually with curl: curl -H 'Authorization: Bearer {headers['Authorization'].split()[1]}' -H 'User-Agent: BasecampAPIClient' '{url}'")
        elif response.status_code == 404:
            logging.warning(f"404 Not Found for Task ID {task_id}: Comments endpoint may not exist or task is inaccessible")
            print(f"    ⚠️ 404 Not Found for Task ID {task_id}: Verify task exists and has comments in Basecamp UI")
            print(f"    Test manually with curl: curl -H 'Authorization: Bearer {headers['Authorization'].split()[1]}' -H 'User-Agent: BasecampAPIClient' '{url}'")
        else:
            logging.error(f"Comments endpoint failed for Task ID {task_id}: Status {response.status_code}, Response: {response.text[:500]}")
            print(f"    ❌ Comments endpoint failed for Task ID {task_id} (Status {response.status_code}): {response.text[:500]}...")
            print(f"    Test manually with curl: curl -H 'Authorization: Bearer {headers['Authorization'].split()[1]}' -H 'User-Agent: BasecampAPIClient' '{url}'")
    else:
        logging.error(f"No response from comments endpoint for Task ID {task_id}: {url}")
        print(f"    ❌ No response from comments endpoint for Task ID {task_id}. Check network or API status.")
    
    if not comment_list and DEBUG_MODE:
        print(f"    ℹ️ Debug: No comments retrieved for Task ID {task_id}. Verify in Basecamp UI and test endpoint manually.")
        logging.debug(f"Debug: No comments retrieved for Task ID {task_id}. Suggested manual test: curl -H 'Authorization: Bearer [token]' -H 'User-Agent: BasecampAPIClient' '{url}'")
    
    logging.info(f"Retrieved {len(comment_list)} comments for Task ID {task_id}")
    return comment_list

def get_tasks(account_id: int, project_id: int, todolist_id: int, access_token: str) -> List[Dict]:
    """Fetch all tasks for a to-do list with detailed comments."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists/{todolist_id}/todos.json"
    tasks = get_paginated_results(url, headers)
    
    if tasks:
        print(f"\n✅ Tasks for To-do List ID {todolist_id}:")
        task_list = []
        for task in tqdm(tasks, desc="Processing tasks", unit="task"):
            task_id = task.get("id")
            # Validate task existence
            task_url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todos/{task_id}.json"
            task_response = retry_request(requests.get, task_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if not task_response or task_response.status_code != 200:
                logging.warning(f"Task ID {task_id} is invalid or inaccessible: Status {task_response.status_code if task_response else 'No response'}")
                print(f"    ⚠️ Skipping Task ID {task_id}: Invalid or inaccessible (Status {task_response.status_code if task_response else 'No response'})")
                continue
            comments = get_task_comments(account_id, project_id, task_id, headers)
            task_info = {
                "title": task.get("title", "N/A"),
                "status": task.get("status", "N/A"),
                "due_on": task.get("due_on", "N/A"),
                "id": task_id,
                "assignee": task.get("assignee", {}).get("name", "N/A"),
                "assignee_id": task.get("assignee", {}).get("id", "N/A"),
                "creator": task.get("creator", {}).get("name", "N/A"),
                "comments": comments
            }
            print(f"- {task_info['title']} (ID: {task_info['id']}, Status: {task_info['status']}, Assignee: {task_info['assignee']})")
            task_list.append(task_info)
            time.sleep(RATE_LIMIT_DELAY)
        logging.info(f"Retrieved {len(task_list)} tasks for to-do list {todolist_id}")
        return task_list
    else:
        print(f"\nℹ️ No Tasks found for To-do List ID {todolist_id}")
        logging.info(f"No tasks found for to-do list {todolist_id}")
        return []

def check_basecamp_status():
    """Check Basecamp API status."""
    try:
        response = requests.get("https://www.basecampstatus.com/", timeout=REQUEST_TIMEOUT)
        if response.ok:
            logging.info("Basecamp status page checked")
            print("ℹ️ Check Basecamp status at: https://www.basecampstatus.com/")
        else:
            logging.warning(f"Failed to check Basecamp status: {response.status_code}")
            print("⚠️ Unable to check Basecamp status")
    except requests.RequestException as e:
        logging.error(f"Error checking Basecamp status: {str(e)}")
        print(f"⚠️ Error checking Basecamp status: {str(e)}")

def extract_text_for_training(json_file: str) -> List[str]:
    """Extract text from JSON data for AI training."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file {json_file} not found")
        print(f"❌ JSON file {json_file} not found")
        return []
    
    texts = []
    for project in data:
        for todolist in project.get("todolists", []):
            for task in todolist.get("tasks", []):
                title = task.get("title", "")
                texts.append(re.sub(r"<[^>]+>", "", title))
                for comment in task.get("comments", []):
                    content = comment.get("content", "")
                    texts.append(re.sub(r"<[^>]+>", "", content))
    
    print(f"✅ Extracted {len(texts)} text snippets for training")
    logging.info(f"Extracted {len(texts)} text snippets from {json_file}")
    return texts

def fetch_basecamp_data() -> List[Dict]:
    """Fetch data from Basecamp API."""
    access_token = get_access_token()
    if not access_token:
        print("❌ Failed to obtain access token. Exiting")
        logging.error("Failed to obtain access token")
        return []
    
    account_id = get_account_info(access_token)
    if not account_id:
        print("❌ Failed to obtain account ID. Exiting")
        logging.error("Failed to obtain account ID")
        return []
    
    projects = get_projects(account_id, access_token)
    if not projects:
        print("❌ No projects found. Exiting")
        logging.error("No projects found")
        check_basecamp_status()
        return []
    
    all_data = []
    for project_name, project_id, todoset_id in tqdm(projects, desc="Processing projects", unit="project"):
        print(f"\n📋 Processing Project: {project_name} (ID: {project_id})")
        logging.info(f"Processing project: {project_name} (ID: {project_id})")
        
        project_data = {
            "project_name": project_name,
            "project_id": project_id,
            "todolists": []
        }
        
        if todoset_id:
            todolists = get_todoset(account_id, project_id, todoset_id, access_token)
            for todolist_name, todolist_id in todolists:
                tasks = get_tasks(account_id, project_id, todolist_id, access_token)
                project_data["todolists"].append({
                    "todolist_name": todolist_name,
                    "todolist_id": todolist_id,
                    "tasks": tasks
                })
        else:
            print(f"ℹ️ No To-do Set found for Project ID {project_id}")
            logging.info(f"No to-do set found for project {project_id}")
        
        all_data.append(project_data)
        time.sleep(RATE_LIMIT_DELAY)
    
    output_file = f"basecamp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✅ Data saved to '{output_file}'")
    logging.info(f"Data saved to {output_file}")
    return all_data

def main():
    """Main function to fetch and display Basecamp task comments."""
    fetch_new_data = input("Fetch new data from Basecamp API? (y/n): ").strip().lower() == 'y'
    
    if fetch_new_data:
        all_data = fetch_basecamp_data()
    else:
        try:
            with open("basecamp_data.json", "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print("✅ Loaded existing data from 'basecamp_data.json'")
            logging.info("Loaded existing data from basecamp_data.json")
        except FileNotFoundError:
            print("❌ No existing data found. Fetching new data...")
            logging.error("basecamp_data.json not found")
            all_data = fetch_basecamp_data()
    
    if not all_data:
        print("❌ No data available to process. Exiting")
        return
    
    print("\n📋 Task Comments Summary:")
    comment_count = 0
    for project in all_data:
        print(f"\nProject: {project['project_name']} (ID: {project['project_id']})")
        for todolist in project.get("todolists", []):
            print(f"  To-do List: {todolist['todolist_name']} (ID: {todolist['todolist_id']})")
            for task in todolist.get("tasks", []):
                print(f"    Task: {task['title']} (ID: {task['id']}, Assignee: {task['assignee']}, Creator: {task['creator']})")
                if task.get("comments"):
                    print("      💬 Comments:")
                    for comment in task["comments"]:
                        print(f"        - ID: {comment['id']}, Creator: {comment['creator']}, Created: {comment['created_at']}")
                        print(f"          Content: {comment['content'][:200]}{'...' if len(comment['content']) > 200 else ''}")
                        comment_count += 1
                else:
                    print("      ℹ️ No comments found for this task")
    
    print(f"\n📊 Total Comments Retrieved: {comment_count}")
    logging.info(f"Total comments retrieved: {comment_count}")
    
    texts = extract_text_for_training("basecamp_data.json")
    if texts:
        print("\nℹ️ Sample extracted text for chatbot training:")
        for text in texts[:5]:
            print(f"- {text[:100]}...")
    
    if comment_count == 0:
        print("\n⚠️ No comments retrieved. Try the following:")
        print("- Verify that tasks have comments in Basecamp UI")
        print("- Check Basecamp status: https://www.basecampstatus.com/")
        print("- Ensure your API token has permission to access comments")
        print("- Test comments endpoint manually with curl:")
        account_id = all_data[0]['project_id'] if all_data else "<account_id>"
        project_id = all_data[0]['project_id'] if all_data else "<project_id>"
        print(f"  curl -H 'Authorization: Bearer {ACCESS_TOKEN or '<your_access_token>'}' -H 'User-Agent: BasecampAPIClient' 'https://3.basecampapi.com/{account_id}/buckets/{project_id}/recordings/<task_id>/comments.json'")
        logging.warning("No comments retrieved; manual testing suggested")

if __name__ == "__main__":
    main()