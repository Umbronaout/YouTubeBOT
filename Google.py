"""
This module contains YouTube related functions
"""
import pickle
import os
from datetime import datetime
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
from google import genai
from google.genai import types

gemini_API_key=""

def Create_Service(client_secret_file, api_name, api_version, *scopes):
    """
    Google service setup
    """
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = None

    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service

    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None

def upload_video(video):
    """
    Creates a request to upload the video and sets its thumbnail
    """
    from googleapiclient.http import MediaFileUpload

    CLIENT_SECRET_FILE = os.path.join('Channels', video.owner.channel_name, 'client_secrets.json')
    API_NAME = 'youtube'
    API_VERSION = 'v3'
    SCOPES = [
                'https://www.googleapis.com/auth/youtube.upload',
                'https://www.googleapis.com/auth/youtube.force-ssl'
              ]

    ### Prepare service ###
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    ### Prepare metadata ###
    # TODO tags are set as 'Reddit' and 'channel_name' without spaces for now
    tags = ['Reddit', video.owner.channel_name.replace(' ','')]
    description = video.description #description limit 5000 chars

    # title - limit 100 chars
    title = video.yt_title
    if len(title) > 100:
        title = title[:97] + '...'

    # Create a request body
    request_body = {
        'snippet': {
            'categoryId': 19,
            'title': title,
            'description': description,
            'tags': tags
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False, 
        },
        'notifySubscribers': False
    }

    # Prepare video
    main_video = MediaFileUpload(os.path.join(video.dir, video.ID + ".mp4"))

    # Upload the video with metadata
    response = service.videos().insert(
        part='snippet,status',
        body=request_body,
        media_body=main_video
    ).execute()

    # Extract video ID from the upload response
    video_id = response.get('id')
    if not video_id:
        raise Exception("Failed to upload video and retrieve video ID.")

    ### Set the thumbnail for the video ###
    thumbnail = MediaFileUpload(os.path.join(video.dir, video.ID + ".png"))
    service.thumbnails().set(
        videoId=video_id,
        media_body=thumbnail
    ).execute()

    ### Upload short and add link to original video ###
    # Create link to the original video
    video_url = f"Watch the full story: https://www.youtube.com/watch?v={video_id}"
    # Create a request body
    request_body_short = {
        'snippet': {
            'categoryId': 19,
            'title': title,
            'description': video_url + "\n" + description + "\n#Short",
            'tags': tags
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False, 
        },
        'notifySubscribers': False
    }

    # Prepare video
    short_video = MediaFileUpload(os.path.join(video.dir, "short_" + video.ID + ".mp4"))

    # Upload the video with metadata
    response = service.videos().insert(
        part='snippet,status',
        body=request_body_short,
        media_body=short_video
    ).execute()

    del service

def generate_text(task, text, type):
    client = genai.Client(
        api_key=gemini_API_key
    )

    generated_text = ""

    prompt = task + text

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        generated_text += chunk.text

    result = []
    sentence = []

    if type == "text":
        return generated_text

    for word in generated_text.split():
        if "[" in word and "]" in word:
            if sentence != []:
                result.append((" ".join(sentence), emotion))
                sentence = []
            emotion = word[1:-1]
        else:
            sentence.append(word)
    
    return result