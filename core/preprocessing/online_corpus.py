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
    def __init__(self, limit_words=500000):
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
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            # Try to get English or Indonesian manually, or just use first available
            transcript = transcript_list.find_transcript(['en', 'id', 'ms'])
            data = transcript.fetch()
            if hasattr(data, 'snippets'):
                return " ".join([t.text for t in data.snippets])
            elif isinstance(data, list):
                return " ".join([t['text'] for t in data])
            return None
        except Exception:
            # Fallback to any transcript
            try:
                api = YouTubeTranscriptApi()
                transcript_list = api.list(video_id)
                transcript = next(iter(transcript_list))
                data = transcript.fetch()
                if hasattr(data, 'snippets'):
                    return " ".join([t.text for t in data.snippets])
                elif isinstance(data, list):
                    return " ".join([t['text'] for t in data])
                return None
            except:
                try:
                    api = YouTubeTranscriptApi()
                    data = api.fetch(video_id)
                    if hasattr(data, 'snippets'):
                        return " ".join([t.text for t in data.snippets])
                    elif isinstance(data, list):
                        return " ".join([t['text'] for t in data])
                except:
                    return None

    def get_youtube_comments(self, video_url, max_comments=None):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url, sort_by=1) # 1 = sorted by newest
        results = []
        count = 0
        for comment in comments:
            if self.is_limit_reached:
                break
            if max_comments is not None and count >= max_comments:
                break
            
            text = comment.get('text', '')
            author = comment.get('author', 'Unknown')
            time_text = comment.get('time', '')
            
            # Create a pseudo-XML structure for the comment
            comment_str = f"<comment author=\"{author}\" date=\"{time_text}\">\n{text}\n</comment>\n"
            results.append(comment_str)
            count += 1
            
            words = count_words(text)
            self.current_words += words
            if self.current_words >= self.limit_words:
                self.is_limit_reached = True
                break
        return "".join(results)

    def is_likely_sentence(self, text):
        text = text.strip()
        if len(text) < 15:
            return False
        if not re.search(r'[.!?]$', text):
            return False
        words = text.split()
        if len(words) < 3:
            return False
        return True

    def scrape_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove scripts, styles, headers, footers, navs
                for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                    element.extract()
                
                paragraphs = soup.find_all('p')
                clean_text = []
                for p in paragraphs:
                    text = p.get_text(separator=' ').strip()
                    text = re.sub(r'\s+', ' ', text)
                    if self.is_likely_sentence(text):
                        clean_text.append(text)
                
                if clean_text:
                    return '\n'.join(clean_text)
                else:
                    # Fallback to basic cleaning if no paragraphs found
                    text = soup.get_text(separator=' ')
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            print(f"Scrape error for {url}: {e}")
        return None

    def score_domain(self, url):
        score = 0
        url_lower = url.lower()
        # High priority (text rich)
        if any(d in url_lower for d in ['wikipedia.org', 'medium.com', 'wordpress.com', 'bbc.com', 'cnn.com', 'nytimes.com', 'kompas.com', 'detik.com', 'tribunnews.com']):
            score += 10
        # Low priority (PDFs, JS heavy, sparse text)
        if any(d in url_lower for d in ['academia.edu', 'scribd.com', 'researchgate.net', 'facebook.com', 'instagram.com', 'twitter.com', 'x.com', 'tiktok.com', 'youtube.com']):
            score -= 10
        return score

    def find_keyword_links(self, keywords, num_links=25, language=None, progress_callback=None):
        query_words = keywords.copy()
        query = " ".join(query_words)
        
        # Build language params for Bing News
        lang_param = "&mkt=en-US"
        if language:
            if language.lower() == "indonesian":
                lang_param = "&mkt=id-ID"
            elif language.lower() == "english":
                lang_param = "&mkt=en-US"
                
        links = set()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Bing News RSS returns ~10 links per page. Paginate to get up to num_links.
        page = 0
        while len(links) < num_links and page < 10:  # Max 10 pages to avoid infinite loop
            first_param = f"&first={page * 10 + 1}" if page > 0 else ""
            url = f"https://www.bing.com/news/search?q={query}&format=rss{lang_param}{first_param}"
            
            try:
                if progress_callback: progress_callback((page + 1) / 10.0, f"Searching Bing News (Page {page + 1})...")
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    import xml.etree.ElementTree as ET
                    import urllib.parse
                    root = ET.fromstring(resp.content)
                    items = root.findall('.//item')
                    
                    if not items:
                        break # No more results
                        
                    for item in items:
                        link = item.find('link').text
                        if link:
                            # Extract the actual URL from Bing's redirect link
                            parsed = urllib.parse.urlparse(link)
                            qs = urllib.parse.parse_qs(parsed.query)
                            if 'url' in qs:
                                clean_url = qs['url'][0]
                                links.add(clean_url)
                            elif 'bing.com' not in link:
                                links.add(link)
                else:
                    break # Stop on error
            except Exception as e:
                print(f"RSS Search error: {e}")
                break
                
            page += 1
            import time
            time.sleep(1) # Be nice to Bing
            
        # If we got more than requested, truncate
        links_list = list(links)
        # Score and sort links
        scored_links = [(link, self.score_domain(link)) for link in links_list]
        scored_links.sort(key=lambda x: x[1], reverse=True) # Highest score first
        
        return [link for link, score in scored_links][:num_links]

    def scrape_selected_links(self, links, keywords, progress_callback=None):
        success_logs = []
        for i, link in enumerate(links):
            if self.is_limit_reached: break
            if progress_callback: progress_callback(i/len(links), f"Scraping {link[:50]}...")
            
            content = self.scrape_url(link)
            if content:
                # We could filter by keyword match here, but since the user explicitly selected them,
                # we just scrape and add.
                self.downloaded_files.append({
                    "filename": f"kw_{i}.txt", 
                    "content": f"<text url=\"{link}\" keywords=\"{','.join(keywords)}\">\n{content}\n</text>"
                })
                words = count_words(content)
                self.current_words += words
                success_logs.append(link)
                if self.current_words >= self.limit_words:
                    self.is_limit_reached = True
                    break
        return success_logs


def build_online_corpus(mode_type, params, progress_callback=None):
    """
    mode_type: 'youtube', 'links', 'keyword_scrape_selected'
    params: dict with necessary parameters
    """
    builder = OnlineCorpusBuilder(limit_words=500000)
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
            max_comments = params.get('max_comments')
            comments = builder.get_youtube_comments(url, max_comments=max_comments)
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
    
    elif mode_type == "keyword_scrape_selected":
        keywords = params.get('keywords', [])
        links = params.get('links', [])
        success_logs = builder.scrape_selected_links(links, keywords, progress_callback)
        warning = f"Successfully scraped {len(success_logs)} out of {len(links)} selected links."
        if builder.is_limit_reached:
            warning += " Experimental limit reached (max 500,000 words)."
        return builder.downloaded_files, warning

    elif mode_type == "mastodon":
        urls = params.get('urls', [])
        mode = params.get('mode', 'both')
        for i, url in enumerate(urls[:50]):
            if builder.is_limit_reached: break
            if progress_callback: progress_callback(i/len(urls), f"Fetching Mastodon URL {i+1}/{len(urls)}...")
            domain_match = re.search(r'https?://([^/]+)', url)
            if not domain_match: continue
            domain = domain_match.group(1)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            # Check if it is a specific status ID or a profile
            id_match = re.search(r'/(?:statuses|@[\w.-]+)/(\d+)', url)
            if not id_match:
                id_match = re.search(r'/(\d+)/?$', url)
                
            status_ids = []
            if id_match:
                status_ids.append(id_match.group(1))
            else:
                # Check for profile URL, e.g. /@username
                profile_match = re.search(r'/@([\w.-]+)', url)
                if not profile_match: continue
                username = profile_match.group(1)
                
                try:
                    # 1. Lookup account ID
                    lookup_url = f"https://{domain}/api/v1/accounts/lookup?acct={username}"
                    lr = requests.get(lookup_url, headers=headers, timeout=10)
                    if lr.status_code == 200:
                        acct_id = lr.json().get('id')
                        if acct_id:
                            # 2. Get latest 10 statuses
                            statuses_url = f"https://{domain}/api/v1/accounts/{acct_id}/statuses?limit=10"
                            sr = requests.get(statuses_url, headers=headers, timeout=10)
                            if sr.status_code == 200:
                                status_ids = [s.get('id') for s in sr.json() if s.get('id')]
                except Exception:
                    continue
                    
            for status_id in status_ids:
                if builder.is_limit_reached: break
                try:
                    r = requests.get(f"https://{domain}/api/v1/statuses/{status_id}", headers=headers, timeout=15)
                    if r.status_code != 200: continue
                    status_data = r.json()
                    rc = requests.get(f"https://{domain}/api/v1/statuses/{status_id}/context", headers=headers, timeout=15)
                    context_data = rc.json() if rc.status_code == 200 else {}
                except Exception:
                    continue
                    
                ancestors = context_data.get('ancestors', [])
                descendants = context_data.get('descendants', [])
                import html
                
                def clean_masto_html(html_content):
                    if not html_content: return ""
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for br in soup.find_all("br"): br.replace_with("\n")
                    for p in soup.find_all("p"): p.append("\n")
                    return soup.get_text().strip()
                    
                xml_parts = []
                xml_parts.append(f'<text source="mastodon" thread_url="{html.escape(url)}" status_id="{status_id}">')
                
                def add_masto_status(status_obj, post_type):
                    s_id = status_obj.get('id')
                    parent_id = status_obj.get('in_reply_to_id') or "none"
                    author = status_obj.get('account', {}).get('acct', 'unknown')
                    content_html = status_obj.get('content', '')
                    text = clean_masto_html(content_html)
                    created_at = status_obj.get('created_at', '')[:10]
                    likes = status_obj.get('favourites_count', 0)
                    boosts = status_obj.get('reblogs_count', 0)
                    xml_parts.append(f'  <u author="{html.escape(author)}" date="{created_at}" post_type="{post_type}" likes="{likes}" boosts="{boosts}" id="{s_id}" parent_id="{parent_id}">{html.escape(text)}</u>')
                    
                if mode in ('post', 'both'):
                    for ancestor in ancestors:
                        add_masto_status(ancestor, "ancestor")
                    add_masto_status(status_data, "post")
                if mode in ('replies', 'both'):
                    for descendant in descendants:
                        add_masto_status(descendant, "reply")
                xml_parts.append('</text>')
                xml_content = "\n".join(xml_parts)
                builder.add_content(f"mastodon_{status_id}.xml", xml_content)
 
    elif mode_type == "bluesky":
        urls = params.get('urls', [])
        mode = params.get('mode', 'both')
        for i, url in enumerate(urls[:50]):
            if builder.is_limit_reached: break
            if progress_callback: progress_callback(i/len(urls), f"Fetching BlueSky URL {i+1}/{len(urls)}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            # Check if it is a specific post or a profile
            match_post = re.search(r'profile/([^/]+)/post/([^/]+)', url)
            posts_to_fetch = []
            
            if match_post:
                handle = match_post.group(1)
                rkey = match_post.group(2)
                posts_to_fetch.append((handle, rkey))
            else:
                match_profile = re.search(r'profile/([^/]+)', url)
                if not match_profile: continue
                handle = match_profile.group(1)
                
                try:
                    # 1. Resolve handle to DID
                    resolve_url = f"https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle={handle}"
                    rr = requests.get(resolve_url, headers=headers, timeout=10)
                    if rr.status_code == 200:
                        did = rr.json().get('did')
                        if did:
                            # 2. Get latest 10 posts
                            feed_url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={did}&limit=10"
                            fr = requests.get(feed_url, headers=headers, timeout=10)
                            if fr.status_code == 200:
                                for item in fr.json().get('feed', []):
                                    post_obj = item.get('post', {})
                                    uri = post_obj.get('uri', '')
                                    uri_match = re.search(r'app\.bsky\.feed\.post/([^/]+)', uri)
                                    if uri_match:
                                        posts_to_fetch.append((handle, uri_match.group(1)))
                except Exception:
                    continue
                    
            for handle, rkey in posts_to_fetch:
                if builder.is_limit_reached: break
                at_uri = f"at://{handle}/app.bsky.feed.post/{rkey}"
                api_url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri={at_uri}"
                try:
                    r = requests.get(api_url, headers=headers, timeout=15)
                    if r.status_code != 200: continue
                    thread_data = r.json()
                except Exception:
                    continue
                    
                thread_node = thread_data.get('thread', {})
                def get_bsky_ancestors(node):
                    ancestors = []
                    current = node.get('parent')
                    while current:
                        post = current.get('post')
                        if post: ancestors.append(post)
                        current = current.get('parent')
                    ancestors.reverse()
                    return ancestors
                    
                def get_bsky_descendants(node):
                    descendants = []
                    replies = node.get('replies', [])
                    for reply in replies:
                        post = reply.get('post')
                        if post:
                            descendants.append(post)
                            descendants.extend(get_bsky_descendants(reply))
                    return descendants
                    
                ancestors = get_bsky_ancestors(thread_node)
                main_post = thread_node.get('post')
                descendants = get_bsky_descendants(thread_node)
                
                import html
                xml_parts = []
                xml_parts.append(f'<text source="bluesky" thread_url="{html.escape(url)}" rkey="{rkey}">')
                
                def add_bsky_post(post_obj, post_type):
                    uri = post_obj.get('uri', '')
                    record = post_obj.get('record', {})
                    reply_info = record.get('reply', {})
                    parent_uri = reply_info.get('parent', {}).get('uri') or "none"
                    author = post_obj.get('author', {}).get('handle', 'unknown')
                    text = record.get('text', '')
                    created_at = record.get('createdAt', '')[:10]
                    likes = post_obj.get('likeCount', 0)
                    reposts = post_obj.get('repostCount', 0)
                    
                    post_id = uri.split('/')[-1] if uri else "unknown"
                    parent_id = parent_uri.split('/')[-1] if parent_uri != "none" else "none"
                    xml_parts.append(f'  <u author="{html.escape(author)}" date="{created_at}" post_type="{post_type}" likes="{likes}" reposts="{reposts}" id="{post_id}" parent_id="{parent_id}">{html.escape(text)}</u>')
                    
                if mode in ('post', 'both'):
                    for ancestor in ancestors:
                        add_bsky_post(ancestor, "ancestor")
                    if main_post:
                        add_bsky_post(main_post, "post")
                if mode in ('replies', 'both'):
                    for descendant in descendants:
                        add_bsky_post(descendant, "reply")
                xml_parts.append('</text>')
                xml_content = "\n".join(xml_parts)
                builder.add_content(f"bluesky_{rkey}.xml", xml_content)

    if builder.is_limit_reached:
        warning = "Experimental limit reached (max 100,000 words). Corpus built with partial content."
        
    return builder.downloaded_files, warning
