from youtube_transcript_api import YouTubeTranscriptApi
import sys

def get_ts_robust(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        try:
            transcript = transcript_list.find_transcript(['en', 'id', 'ms', 'en-US', 'en-GB'])
        except:
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                return "No transcript available"
        
        data = transcript.fetch()
        
        if hasattr(data, 'snippets'):
            return " ".join([t.text for t in data.snippets])
        elif isinstance(data, list):
            try:
                if len(data) > 0 and hasattr(data[0], 'text'):
                    return " ".join([t.text for t in data])
                else:
                    return " ".join([t.get('text', '') for t in data if 'text' in t])
            except Exception as e:
                return f"Error extracting from list: {e}"
        else:
            return f"Unknown data format: {type(data)}"
    except Exception as e:
        return f"Error API: {e}"

print(get_ts_robust('jNQXAC9IVRw')[:100])
