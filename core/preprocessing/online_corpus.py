import os
import re
import requests
import time
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_comment_downloader import YoutubeCommentDownloader

def count_words(text):
    if not text: return 0
    return len(re.findall(r'\w+', text))

class OnlineCorpusBuilder:
    def __init__(self, limit_words=100000):
        self.limit_words = limit_words
        self.current_words = 0
        self.is_limit_reached = False
        self.downloaded_files = [] # List of dicts {filename, content}

    def add_content(self, filename, content):
        if self.is_limit_reached:
            return False
        
        words = count_words(content)
        if self.current_words + words > self.limit_words:
            # Add what we can or just add and stop
            self.downloaded_files.append({"filename": filename, "content": content})
            self.current_words += words
            self.is_limit_reached = True
            return True
        else:
            self.downloaded_files.append({"filename": filename, "content": content})
            self.current_words += words
            return True

    def get_youtube_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try to get English or Indonesian manually, or just use first available
            transcript = transcript_list.find_transcript(['en', 'id', 'ms'])
            data = transcript.fetch()
            return " ".join([t['text'] for t in data])
        except Exception:
            # Fallback to any transcript
            try:
                data = YouTubeTranscriptApi.get_transcript(video_id)
                return " ".join([t['text'] for t in data])
            except:
                return None

    def get_youtube_comments(self, video_url):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url, sort_by=1) # 1 = sorted by newest
        results = []
        for comment in comments:
            if self.is_limit_reached:
                break
            
            text = comment.get('text', '')
            author = comment.get('author', 'Unknown')
            time_text = comment.get('time', '')
            
            # Create a pseudo-XML structure for the comment
            comment_str = f"<comment author=\"{author}\" date=\"{time_text}\">\n{text}\n</comment>\n"
            results.append(comment_str)
            
            words = count_words(text)
            self.current_words += words
            if self.current_words >= self.limit_words:
                self.is_limit_reached = True
                break
        return "".join(results)

    def scrape_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')
                # Basic cleaning
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
        except Exception as e:
            print(f"Scrape error for {url}: {e}")
        return None

    def keyword_search(self, keywords, min_match=2):
        # Using DuckDuckGo HTML version for simplicity
        query = " ".join(keywords)
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        links = []
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # DDG HTML results are in links with class 'result__a'
                for a in soup.find_all('a', class_='result__a'):
                    link = a.get('href')
                    if link and link.startswith('http'):
                        links.append(link)
        except Exception as e:
            print(f"Search error: {e}")
        
        # Filter links and scrape
        found_data = []
        for link in links[:20]: # Check first 20 links
            if self.is_limit_reached: break
            
            content = self.scrape_url(link)
            if content:
                # Count matches
                matches = sum(1 for kw in keywords if kw.lower() in content.lower())
                if matches >= min_match:
                    found_data.append((link, content))
                    
                    words = count_words(content)
                    self.current_words += words
                    if self.current_words >= self.limit_words:
                        self.is_limit_reached = True
                        break
        return found_data

def build_online_corpus(mode_type, params, progress_callback=None):
    """
    mode_type: 'youtube', 'links', 'keyword'
    params: dict with necessary parameters
    """
    builder = OnlineCorpusBuilder(limit_words=100000)
    warning = None
    
    if mode_type == "youtube":
        url = params.get('url')
        mode = params.get('mode', 'both') # transcript, comments, both
        
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        video_id = video_id_match.group(1) if video_id_match else None
        
        if not video_id:
            return None, "Invalid YouTube URL"
        
        if mode in ('transcript', 'both'):
            if progress_callback: progress_callback(0.2, "Downloading transcript...")
            ts = builder.get_youtube_transcript(video_id)
            if ts:
                builder.add_content(f"yt_{video_id}_transcript.txt", f"<text type=\"transcript\" video_id=\"{video_id}\" url=\"{url}\">\n{ts}\n</text>")
            else:
                warning = "Could not find transcript for this video."
        
        if not builder.is_limit_reached and mode in ('comments', 'both'):
            if progress_callback: progress_callback(0.5, "Downloading comments...")
            comments = builder.get_youtube_comments(url)
            if comments:
                builder.add_content(f"yt_{video_id}_comments.xml", f"<text type=\"comments\" video_id=\"{video_id}\" url=\"{url}\">\n{comments}\n</text>")
    
    elif mode_type == "links":
        links = params.get('links', [])
        for i, link in enumerate(links[:50]):
            if builder.is_limit_reached: break
            if progress_callback: progress_callback(i/len(links), f"Scraping {link}...")
            content = builder.scrape_url(link)
            if content:
                builder.add_content(f"link_{i}.txt", f"<text url=\"{link}\" source=\"link_collection\">\n{content}\n</text>")
    
    elif mode_type == "keyword":
        keywords = params.get('keywords', [])
        min_match = max(2, len(keywords) - 2)
        if progress_callback: progress_callback(0.1, "Searching for links...")
        found = builder.keyword_search(keywords, min_match)
        for i, (link, content) in enumerate(found):
            # Content already added in keyword_search for limit checking
            builder.downloaded_files.append({"filename": f"kw_{i}.txt", "content": f"<text url=\"{link}\" keywords=\"{','.join(keywords)}\">\n{content}\n</text>"})
            
    if builder.is_limit_reached:
        warning = "Experimental limit reached (max 100,000 words). Corpus built with partial content."
        
    return builder.downloaded_files, warning
