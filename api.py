import webbrowser
import http.server
import socketserver
import urllib.parse
import requests

CLIENT_ID = "50a101d6c62602fa263cba350201cf7ab27ab618"
CLIENT_SECRET = "a92a6a7b7ad2127750dd440b5e54d0b626461b1f"
REDIRECT_URI = "http://localhost:8000/oauth/callback"

AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"

class OAuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/oauth/callback'):
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code = params.get('code', [None])[0]

            if code:
                print(f"Authorization code received: {code}")
                token_response = requests.post("https://launchpad.37signals.com/authorization/token.json", data={
                    "type": "web_server",
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uri": REDIRECT_URI,
                    "code": code
                })

                if token_response.ok:
                    token_data = token_response.json()
                    print("✅ Access Token:", token_data.get("access_token"))
                    self.respond_with("Success! You can close this tab.")
                else:
                    print("❌ Token exchange failed:", token_response.text)
                    self.respond_with("Token exchange failed. Check your terminal.")
            else:
                self.respond_with("No code found in callback URL.")

    def respond_with(self, message):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())

print("🔗 Opening browser for authorization...")
webbrowser.open(AUTH_URL)

print("🌐 Waiting for authorization...")
with socketserver.TCPServer(("localhost", 8000), OAuthHandler) as httpd:
    httpd.handle_request()
