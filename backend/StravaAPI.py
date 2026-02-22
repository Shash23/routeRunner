import datetime

from stravalib.client import Client
from dotenv import load_dotenv, set_key
import os


ENV_PATH = ".local.env"

def load_env():
    """Load environment variables from .local.env"""
    load_dotenv(ENV_PATH)
    required_vars = ["STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET"]
    for var in required_vars:
        if not os.getenv(var):
            raise RuntimeError(f"{var} not set in .local.env")


def refresh_token_if_needed(client):
    """Refresh access token if expired or missing"""
    access_token = os.getenv("STRAVA_ACCESS_TOKEN")
    refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
    client.access_token = access_token
    client.refresh_token = refresh_token
    client.client_id = os.getenv("STRAVA_CLIENT_ID")
    client.client_secret = os.getenv("STRAVA_CLIENT_SECRET")

    # If there’s no access token or refresh token, we need to authorize
    if not access_token or not refresh_token:
        print("No access/refresh token found. Starting OAuth flow...")
        return oauth_flow(client)

    # Try refreshing token to make sure it’s valid
    try:
        new_tokens = client.refresh_access_token()
        set_key(ENV_PATH, "STRAVA_ACCESS_TOKEN", new_tokens['access_token'])
        set_key(ENV_PATH, "STRAVA_REFRESH_TOKEN", new_tokens['refresh_token'])
        expires_at = datetime.datetime.fromtimestamp(new_tokens['expires_at'])
        print(f"Access token refreshed. Expires at: {expires_at}")
    except Exception as e:
        print("Refresh failed. Starting OAuth flow...")
        return oauth_flow(client)


def oauth_flow(client):
    """Full OAuth flow: user approves app and generates tokens"""
    CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
    CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
    REDIRECT_URI = "http://localhost/exchange_token"
    SCOPES = "read,activity:read"

    auth_url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&"
        f"approval_prompt=auto&scope={SCOPES}"
    )

    print("Go to this URL in your browser to authorize the app:")
    print(auth_url)
    code = input("Paste the code from the URL here and press Enter: ")

    token_response = client.exchange_code_for_token(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        code=code
    )

    # Save new tokens
    set_key(ENV_PATH, "STRAVA_ACCESS_TOKEN", token_response['access_token'])
    set_key(ENV_PATH, "STRAVA_REFRESH_TOKEN", token_response['refresh_token'])
    expires_at = datetime.datetime.fromtimestamp(token_response['expires_at'])
    print(f"New tokens saved. Access token expires at: {expires_at}")

    # Assign tokens to client
    client.access_token = token_response['access_token']
    client.refresh_token = token_response['refresh_token']


def main():

    load_env()
    client = Client()
    refresh_token_if_needed(client)

    # Now you can safely call Strava API
    athlete = client.get_athlete()
    print(f"Athlete: {athlete.firstname} {athlete.lastname}")

    activities = client.get_activities(limit=5)
    for activity in activities:
        print(f"{activity.name} - {activity.distance} meters")


if __name__ == "__main__":
    main()
