import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import google.auth.transport.requests
import pickle

class YoutubeClient:
    def __init__(self, client_secret_path, creds_file="token.pickle", scopes=None):
        self.client_secret_path = client_secret_path
        self.creds_file = creds_file 
        self.scopes = ["https://www.googleapis.com/auth/youtube.readonly"] if scopes is None else scopes
        self.youtube = self._load_credentials_and_build_client()

    """
    Loads client secret (loads upon initialization)
    """
    def _load_credentials_and_build_client(self):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        creds = None
        # Load credentials if token.pickle exists
        if os.path.exists(self.creds_file):
            with open(self.creds_file, 'rb') as token:
                creds = pickle.load(token)
        # If there are no valid credentials, go through the OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            else:
                flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path, self.scopes)
                creds = flow.run_local_server(port=0)
            # Save credentials for next time
            with open(self.creds_file, 'wb') as token:
                pickle.dump(creds, token)
        # Build and return the YouTube API client
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", credentials=creds)
        return youtube
    
    def search(self, q, part="snippet", type="video", topicId="/m/05qt0", videoDuration="long", order="viewCount", maxResults=50, pageToken=None, publishedAfter="2014-06-01T00:00:00Z"):    
        request = self.youtube.search().list(
            part=part,
            q=q,
            type=type,
            topicId=topicId,
            videoDuration=videoDuration,
            order=order,
            maxResults=maxResults,
            pageToken=pageToken,
            publishedAfter=publishedAfter
        )
        response = request.execute()        #no need to rraise, if code is 4xx it raises excpetion by itself
        return response

    """ 
    
    returns: exhaustive json of all items
    """
    def exhaustive_search(self, q, part="snippet", type="video", topicId="/m/05qt0", videoDuration="long", order="viewCount", maxResults=50, max_total_results=None,publishedAfter="2014-06-01T00:00:00Z"):
        results = []
        next_page_token = None
        while True:
            response = self.search(
                q=q,
                part=part,
                type=type,
                topicId=topicId,
                videoDuration=videoDuration,
                order=order,
                maxResults=maxResults,
                pageToken=next_page_token,
                publishedAfter=publishedAfter
            )
            items = response.get("items", [])
            results.extend(items)

            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(results) >= max_total_results:
                break
        return results[:max_total_results]
