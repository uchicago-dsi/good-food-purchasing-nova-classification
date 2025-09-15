import json
import os
import time

import jwt
import requests

INSTALLATION_ID = "86079416"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
APP_PRIVATE_KEY = os.environ["APP_PRIVATE_KEY"]
APP_CLIENT_ID = os.environ["APP_CLIENT_ID"]
DISCUSSION_BODY = os.environ["DISCUSSION_BODY"]
DISCUSSION_ID = os.environ["DISCUSSION_ID"]

jwt_instance = jwt.JWT()
current_token = None
expiration = 0


def get_token():
    global current_token
    global expiration
    now = int(time.time())
    if now >= expiration - 1:
        expiration = now + 600
        current_jwt = jwt_instance.encode(
            {"iat": now, "exp": expiration, "iss": APP_CLIENT_ID},
            APP_PRIVATE_KEY.encode(),
            algorithm="RS256",
        )
        response = requests.post(
            f"https://api.github.com/app/installations/{INSTALLATION_ID}/access_tokens",
            headers={
                "Authorization": f"Bearer {current_jwt}",
                "Accept": "application/vnd.github+json",
            },
        )
        current_token = response.json().get("token")
    return current_token


def write_comment(text):
    response = requests.post(
        "https://api.github.com/graphql",
        headers={
            "Authorization": f"token {get_token()}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "query": """
mutation AddDiscussionComment {
  addDiscussionComment(input: {
    discussionId: "%s",
    body: "%s"
  }) {
    comment { id }
  }
}
""" % (DISCUSSION_ID, text)
        },
    )
    print(
        f"write_comment {response.status_code = } {response.headers = } {response.text = }"
    )


write_comment("Why, hello there!")
