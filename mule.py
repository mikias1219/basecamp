import streamlit as st
import requests
import json
import time
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import urllib.parse
import socket
import socketserver
import http.server
import webbrowser
import uuid

# Configure logging
logging.basicConfig(
    filename='checkin_analyzer.log',
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
TOKEN_FILE = "access_token.json"
TOKEN_EXPIRY = timedelta(days=1)

CONFIG = {
    "CLIENT_ID": "572dc3146528e31ad74362575d8c847fbc0e48a3",  # Basecamp CLIENT_ID
    "CLIENT_SECRET": "97b3e34abd53c71ab9b78c35b9d259aa61c7bb1e"  # Basecamp CLIENT_SECRET
}
CLIENT_ID = CONFIG.get("CLIENT_ID")
CLIENT_SECRET = CONFIG.get("CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/oauth/callback"

# Utility Functions
def save_access_token(token: str, expiry: datetime):
    try:
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
            expiry = datetime.fromisoformat(data.get("expiry"))
            if datetime.now() < expiry:
                return {"access_token": data.get("access_token"), "expiry": expiry}
            os.remove(TOKEN_FILE)
            logging.info("Access token expired, removed token file")
            return None
    except FileNotFoundError:
        logging.debug("No access token file found")
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
            if response.status_code == 200:
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
                            save_access_token(access_token, datetime.now() + TOKEN_EXPIRY)
                            st.session_state[f\'access_token_{st.session_state.session_id}\'] = access_token
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
    
    # Step 1: Fetch buckets/projects
    buckets_url = f"{BASE_URL}/{account_id}/projects.json"
    buckets_response = retry_request(requests.get, buckets_url, headers=headers, timeout=REQUEST_TIMEOUT)
    if not buckets_response or not buckets_response.ok:
        logging.error(f"Failed to fetch buckets: Status {buckets_response.status_code if buckets_response else 'None'}, Response: {buckets_response.text if buckets_response else 'No response'}")
        st.error("Failed to fetch projects. Check your permissions or Basecamp account setup.")
        return []
    buckets = buckets_response.json()
    logging.debug(f"Fetched {len(buckets)} buckets")

    # Step 2: Process each bucket
    for bucket in buckets:
        bucket_id = bucket.get("id")
        # Look for questionnaire in dock
        questionnaire_id = None
        for dock_item in bucket.get("dock", []):
            if dock_item.get("name") == "questionnaire" and dock_item.get("enabled"):
                questionnaire_id = dock_item.get("id")
                questionnaire_url = dock_item.get("url")
                break

        if not questionnaire_id:
            logging.debug(f"No enabled questionnaire found in dock for bucket {bucket_id}")
            continue

        # Step 3: Try fetching questions directly for the questionnaire
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
                    "questionnaire_id": questionnaire_id
                })
        else:
            logging.error(f"Failed to fetch questions for questionnaire {questionnaire_id} in bucket {bucket_id}: Status {questions_response.status_code if questions_response else 'None'}, Response: {questions_response.text if questions_response else 'No response'}")
            # Fallback: Try fetching questionnaire details
            questionnaire_response = retry_request(requests.get, questionnaire_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if questionnaire_response and questionnaire_response.ok:
                questionnaire_data = questionnaire_response.json()
                logging.debug(f"Questionnaire {questionnaire_id} details: {questionnaire_data}")
                questions_url = questionnaire_data.get("questions_url")
                if questions_url:
                    questions_response = retry_request(requests.get, questions_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    if questions_response and questions_response.ok:
                        questions = questions_response.json()
                        logging.debug(f"Fallback: Fetched {len(questions)} questions for questionnaire {questionnaire_id}")
                        for question in questions:
                            checkins.append({
                                "title": question.get("title", "Untitled Question"),
                                "url": f"{BASE_URL}/{account_id}/buckets/{bucket_id}/questions/{question['id']}",
                                "account_id": account_id,
                                "bucket_id": bucket_id,
                                "question_id": question.get("id"),
                                "questionnaire_id": questionnaire_id
                            })
                    else:
                        logging.error(f"Fallback failed: Status {questions_response.status_code if questions_response else 'None'}, Response: {questions_response.text if questions_response else 'No response'}")
            else:
                logging.error(f"Failed to fetch questionnaire {questionnaire_id}: Status {questionnaire_response.status_code if questionnaire_response else 'None'}, Response: {questionnaire_response.text if questionnaire_response else 'No response'}")

    logging.info(f"Fetched {len(checkins)} check-in questions")
    if not checkins:
        logging.warning("No check-in questions found. Possible causes: no active questionnaires, no questions, or insufficient permissions.")
    return checkins

def filter_answers_by_date(answers: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    try:
        # Parse dates and make them UTC-aware
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
        logging.debug(f"Fetched {len(answers)} raw answers for question ID {question_id}")
        for answer in answers:
            answer["question_id"] = question_id
            answer["bucket_id"] = bucket_id
            answer["question_title"] = next((q["title"] for q in get_questionnaire_questions(account_id, bucket_id, access_token) if q["id"] == question_id), "Unknown")
        all_answers.extend(answers)
    filtered_answers = filter_answers_by_date(all_answers, start_date, end_date)
    logging.debug(f"Structured {len(filtered_answers)} answers")
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
        logging.debug(f"Fetched {len(questions_response)} questions for questionnaire {questionnaire_id}")
    logging.debug(f"Total {len(questions)} questions fetched for bucket {bucket_id}")
    return questions

def save_checkin_data(data: List[Dict]):
    try:
        with open(CHECKIN_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved {len(data)} answers to {CHECKIN_DATA_FILE}")
    except Exception as e:
        logging.error(f"Failed to save checkin data: {str(e)}")
        st.error("Failed to save check-in data.")

def load_checkin_data() -> List[Dict]:
    try:
        with open(CHECKIN_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.debug(f"Loaded {len(data)} answers from {CHECKIN_DATA_FILE}")
            return data
    except FileNotFoundError:
        logging.debug(f"No checkin data file found at {CHECKIN_DATA_FILE}")
        return []
    except Exception as e:
        logging.error(f"Error loading checkin data: {str(e)}")
        return []

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

# Streamlit UI
st.set_page_config(page_title="Basecamp Check-in Analyzer", layout="wide")

def main():
    st.markdown("""
    <style>
    .navbar {
        background-color: #2c3e50;
        padding: 15px;
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput input {
        border-radius: 8px;
        padding: 10px;
    }
    .stSelectbox select {
        border-radius: 8px;
        padding: 10px;
    }
    .stSuccess, .stError, .stInfo {
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
    </style>
    <div class="navbar">Basecamp Check-in Analyzer</div>
    """, unsafe_allow_html=True)

    if 'access_token' not in st.session_state:
        st.session_state[f\'access_token_{st.session_state.session_id}\'] = None
    if 'account_id' not in st.session_state:
        st.session_state.account_id = None
    if 'checkins' not in st.session_state:
        st.session_state.checkins = []

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Settings", "Debug"])

    if page == "Home":
        st.header("Check-in Analysis")
        st.write("Analyze Basecamp Automatic Check-in responses to generate internship reports and insights.")

        if not st.session_state[f\'access_token_{st.session_state.session_id}\'] or not st.session_state.account_id:
            st.error("Please authenticate in the Settings page to proceed.")
            return

        # Fetch all check-ins
        if not st.session_state.checkins or st.button("Refresh Check-ins"):
            with st.spinner("Fetching available check-ins..."):
                st.session_state.checkins = get_all_checkins(st.session_state.account_id, st.session_state[f\'access_token_{st.session_state.session_id}\'])
                if not st.session_state.checkins:
                    st.warning(f"""
                    No check-in questions found. Possible causes:
                    - **No Active Questions**: Questionnaire (e.g., ID 8387114304 in bucket 41303586) may have no questions or is inactive. Check Basecamp UI under 'Automatic Check-ins' for project 'TTA - AI and Machine Learning'.
                    - **Permissions Issue**: The authenticated user may lack access to questionnaire questions. Verify permissions with your Basecamp admin.
                    - **API Error**: Check the 'Debug' page or `checkin_analyzer.log` for detailed error messages (e.g., 403, 404).
                    Try creating a new question in the check-in or contact your Basecamp admin to verify questionnaire status.
                    """)
                    logging.warning("No check-ins retrieved")
                    return

        # Select check-in
        checkin_options = [f"{c['title']} (Bucket ID: {c['bucket_id']}, Questionnaire ID: {c['questionnaire_id']})" for c in st.session_state.checkins]
        selected_checkin = st.selectbox("Select Check-in Question", checkin_options)
        selected_index = checkin_options.index(selected_checkin) if selected_checkin in checkin_options else 0
        selected_checkin_data = st.session_state.checkins[selected_index]

        # Date range selection
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=7)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start)
        with col2:
            end_date = st.date_input("End Date", value=default_end)

        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        if st.button("Fetch Check-in Data", disabled=not st.session_state[f\'access_token_{st.session_state.session_id}\']):
            with st.spinner("Fetching check-in data..."):
                try:
                    answers = fetch_and_structure_answers(
                        selected_checkin_data["account_id"],
                        selected_checkin_data["bucket_id"],
                        [selected_checkin_data["question_id"]],
                        st.session_state[f\'access_token_{st.session_state.session_id}\'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    if answers:
                        save_checkin_data(answers)
                        st.success(f"Fetched and filtered {len(answers)} answers for {start_date} to {end_date}.")
                    else:
                        st.warning(f"""
                        No answers found for the selected check-in and date range ({start_date} to {end_date}).
                        - **Check Date Range**: Ensure answers exist within the selected dates. Try a broader range (e.g., past 30 days).
                        - **Verify Answers**: Confirm answers are posted for '{selected_checkin_data['title']}' in Basecamp.
                        - **Add Test Answer**: Post a test answer in Basecamp for the selected question.
                        """)
                        logging.warning("No answers retrieved for selected check-in and date range")
                        save_checkin_data([])
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    logging.error(f"Error fetching data: {str(e)}", exc_info=True)

        answers = load_checkin_data()
        if answers:
            st.subheader("Fetched Answers")
            st.write(f"Total answers: {len(answers)}")
            learning_responses = [a for a in answers if "learn" in a["question_title"].lower()]
            challenge_responses = [a for a in answers if "challenge" in a["question_title"].lower()]
            planned_tasks = [a for a in answers if "plan" in a["question_title"].lower()]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Learning Responses**: {len(learning_responses)}")
            with col2:
                st.write(f"**Challenge Responses**: {len(challenge_responses)}")
            with col3:
                st.write(f"**Planned Tasks**: {len(planned_tasks)}")

            st.subheader("Generate Reports")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Weekly Summary"):
                    with st.spinner("Generating summary..."):
                        summary_prompt = f"""
                        You are an AI assistant summarizing internship progress based on daily check-in responses from Basecamp. Given the following data from {start_date} to {end_date}:

                        **Learning Responses**:
                        {chr(10).join([f"- {a['content']} on {a['created_at']}" for a in learning_responses])}

                        **Challenge Responses**:
                        {chr(10).join([f"- {a['content']} on {a['created_at']}" for a in challenge_responses])}

                        Provide a concise summary (150-200 words) for a weekly internship report. Highlight key learnings and challenges faced by the team, ensuring a professional tone. Focus on common themes, significant achievements, and recurring obstacles. Avoid including specific names of individuals. Structure the summary with two sections: "Key Learnings" and "Challenges Encountered".
                        """
                        summary = generate_llm_output(summary_prompt)
                        st.session_state.summary = summary
                        st.markdown(f"<div class='output-box'><strong>Weekly Summary</strong>:<br>{summary}</div>", unsafe_allow_html=True)
                        save_reports({"summary": summary, "insights": st.session_state.get("insights", "")})

            with col2:
                if st.button("Generate Next Week's Insights"):
                    with st.spinner("Generating insights..."):
                        insights_prompt = f"""
                        You are an AI assistant providing insights for a software development internship team based on Basecamp daily check-in responses. Given the following planned tasks from {start_date} to {end_date}:

                        **Planned Tasks**:
                        {chr(10).join([f"- {a['content']} on {a['created_at']}" for a in planned_tasks])}

                        Analyze the planned tasks and generate actionable insights (150-200 words) for the next week's accomplishments. Identify common goals, potential focus areas, and recommend specific actions to enhance productivity or collaboration. Use a professional, encouraging tone. Avoid mentioning specific individuals. Structure the insights with two sections: "Key Planned Activities" and "Recommended Next Steps".
                        """
                        insights = generate_llm_output(insights_prompt)
                        st.session_state.insights = insights
                        st.markdown(f"<div class='output-box'><strong>Next Week's Insights</strong>:<br>{insights}</div>", unsafe_allow_html=True)
                        save_reports({"summary": st.session_state.get("summary", ""), "insights": insights})

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Basecamp Authentication")
        if st.button("Authenticate with Basecamp"):
            with st.spinner("Authenticating..."):
                access_token = get_access_token()
                if access_token:
                    st.session_state[f\'access_token_{st.session_state.session_id}\'] = access_token
                    account_id = get_account_info(access_token)
                    if account_id:
                        st.session_state.account_id = account_id
                        st.session_state.checkins = []  # Reset checkins to force refresh
                        st.success("Authentication successful!")
                        logging.info("Authentication successful")
                    else:
                        st.error("Failed to fetch account ID. Please verify your Basecamp credentials.")
                        logging.error("Failed to fetch account ID")
                        if os.path.exists(TOKEN_FILE):
                            os.remove(TOKEN_FILE)
                            logging.info("Removed invalid token file")

    elif page == "Debug":
        st.header("Debug Information")
        st.write("Use this page to diagnose issues with check-in fetching.")
        if not st.session_state[f\'access_token_{st.session_state.session_id}\'] or not st.session_state.account_id:
            st.error("Please authenticate in the Settings page to view debug info.")
            return

        st.subheader("Check-in Fetching Debug")
        if st.button("Fetch Raw Check-in Data"):
            with st.spinner("Fetching raw check-in data..."):
                headers = {"Authorization": f"Bearer {st.session_state[f\'access_token_{st.session_state.session_id}\']}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
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
                                st.error(f"Failed to fetch questions for questionnaire {questionnaire_id}: Status {questions_response.status_code if questions_response else 'None'}, Response: {questions_response.text if questions_response else 'No response'}")
                            # Fetch questionnaire details
                            questionnaire_response = retry_request(requests.get, questionnaire_url, headers=headers, timeout=REQUEST_TIMEOUT)
                            if questionnaire_response and questionnaire_response.ok:
                                questionnaire_data = questionnaire_response.json()
                                st.write(f"**Questionnaire Details for ID {questionnaire_id}**")
                                st.json(questionnaire_data)
                            else:
                                st.error(f"Failed to fetch questionnaire details for ID {questionnaire_id}: Status {questionnaire_response.status_code if questionnaire_response else 'None'}, Response: {questionnaire_response.text if questionnaire_response else 'No response'}")
                        else:
                            st.warning(f"No enabled questionnaire found for bucket {bucket_id}.")
                else:
                    st.error(f"Failed to fetch buckets: Status {buckets_response.status_code if buckets_response else 'None'}, Response: {buckets_response.text if buckets_response else 'No response'}")

        st.subheader("Test Specific Questionnaire")
        questionnaire_id = st.text_input("Enter Questionnaire ID (e.g., 8387114304)", "")
        bucket_id = st.text_input("Enter Bucket ID (e.g., 41303586)", "")
        if st.button("Fetch Questions for Questionnaire"):
            if questionnaire_id and bucket_id:
                with st.spinner("Fetching questions..."):
                    headers = {"Authorization": f"Bearer {st.session_state[f\'access_token_{st.session_state.session_id}\']}", "User-Agent": "BasecampAI (mulukenashenafi84@outlook.com)"}
                    questions_url = f"{BASE_URL}/{st.session_state.account_id}/buckets/{bucket_id}/questionnaires/{questionnaire_id}/questions.json"
                    questions_response = retry_request(requests.get, questions_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    if questions_response and questions_response.ok:
                        questions = questions_response.json()
                        st.write(f"**Questions Fetched**: {len(questions)}")
                        st.json(questions)
                    else:
                        st.error(f"Failed to fetch questions: Status {questions_response.status_code if questions_response else 'None'}, Response: {questions_response.text if questions_response else 'No response'}")
            else:
                st.error("Please enter both Questionnaire ID and Bucket ID.")

def save_reports(reports: Dict):
    try:
        with open("reports.json", "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        logging.info("Saved reports to reports.json")
    except Exception as e:
        logging.error(f"Failed to save reports: {str(e)}")
        st.error("Failed to save reports.")

if __name__ == "__main__":
    main()