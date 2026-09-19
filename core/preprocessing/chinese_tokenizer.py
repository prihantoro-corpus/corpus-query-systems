import os
import math
import re

class ChineseSegmenter:
    def __init__(self, unigram_file, bigram_file=None):
        self.maxwordlength = 8
        self.cwords = {}
        self.bigrams = {}
        self.maxdlen = 0
        
        # Numbers
        self.cnumbers = "零○一二三四五六七八九十百千万亿０１２３４５６７８９第"
        self.cnumbersonly = "零○二三四五六七八九十百千万亿"
        numberdesc_str = self.cnumbers + "多半数几俩卅两壹贰叁肆伍陆柒捌玖拾伯仟％‰．点.-0123456789"
        self.numberdesc = set(numberdesc_str)
        self.NUMBERLOG = math.log(34298 / 1000000)
        
        # Wide ASCII
        self.wascii = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ．ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ－-"
        
        # Foreign
        cforeign_str = (
            "阿埃艾爱安奥澳巴保鲍贝本比宾波伯柏勃卜布茨达戴德登迪蒂丁都顿多俄厄恩尔法菲费芬"
            "夫福弗佛盖甘冈哥戈格根古哈海合赫胡华霍基吉加伽贾杰捷金喀卡凯柯科可克肯库拉"
            "莱来赖兰劳勒雷累黎里利莉烈林琳卢鲁伦罗洛马玛麦迈曼梅蒙米摩莫墨默姆穆那娜纳乃"
            "内尼妮努诺帕佩裴蓬皮匹泼普奇齐乔切冉萨塞桑瑟森沙莎舍什史士斯丝舒索苏塔泰坦特图土托瓦万"
            "维温文沃乌伍西希谢辛休逊雅亚延耶伊印尤泽扎詹诸兹腓胥"
        )
        self.cforeign = set(cforeign_str)
        
        # Surnames
        self.surname = set("李王張张劉刘陳陈楊杨黃黄趙赵周吳吴徐孫孙朱馬马胡郭林何高梁鄭郑羅罗宋謝谢唐韓韩曹許许鄧邓蕭萧肖馮冯曾程蔡彭潘袁于董余蘇苏葉叶呂吕魏蔣蒋田杜丁沈姜范江傅鍾钟盧卢汪戴崔任陸陆廖姚方金邱夏譚谭韋韦賈贾鄒邹石熊孟秦閻阎薛侯雷白龍龙段郝孔邵史毛常萬万顧顾賴赖武康賀贺嚴严尹錢钱施牛洪龔龚佘麥麦莊庄路黎符邢倪陶葛")
        self.NAMELOG = math.log(932 / 1000000)
        self.uncommonsurname = set("车成全韩赖连路明牛权时水文席应英于查费")
        
        # Not in name
        self.notname = set("的说对在和是被最所那这有将会与於他为")
        self.cpunctuation = set("、：，。★〖〗（）⊙～【】―・？！“”　")
        self.notname.update(self.cpunctuation)
        
        # Dates
        self.ctime = set("年月日")
        
        self.load_dictionaries(unigram_file, bigram_file)
        
    def load_dictionaries(self, wlist, w2list):
        if not os.path.exists(wlist):
            return
            
        with open(wlist, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%%"):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word, logfreq = parts[0], parts[1]
                    l = len(word)
                    if l > self.maxwordlength:
                        continue
                    self.cwords[word] = float(logfreq)
                    if l > self.maxdlen:
                        self.maxdlen = l
                        
        if w2list and os.path.exists(w2list):
            with open(w2list, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        words, logfreq = parts[0], parts[1]
                        # Expecting "word1 word2" format
                        if ' ' in words:
                            self.bigrams[words] = float(logfreq)

    def is_chinese_char(self, char):
        if not char:
            return False
        c = ord(char[0])
        if c < 0x25CB or c > 0xFF00 or (0x3000 <= c <= 0x301F) or (0x3041 <= c <= 0x309F) or (0x30A1 <= c <= 0x30FF):
            return False
        return True

    def is_chinese_name(self, word):
        if any(c in self.notname for c in word):
            return False
        if len(word) <= 3 and word and word[0] in self.surname:
            last2chars = word[-2:] if len(word) >= 2 else ""
            if last2chars not in self.cwords:
                return True
        return False

    def is_all_in_set(self, word, char_set):
        return all(c in char_set for c in word)

    def cPw(self, w, p):
        p = p or ""
        bigram_key = f"{p} {w}"
        if bigram_key in self.bigrams and p in self.cwords:
            return self.bigrams[bigram_key] - self.cwords[p]
        else:
            # Backoff
            if w in self.cwords:
                return self.cwords[w]
            elif self.is_chinese_name(w):
                return self.NAMELOG
            elif self.is_all_in_set(w, self.cforeign):
                return self.NAMELOG - 1
            elif self.is_all_in_set(w, self.numberdesc):
                return self.NUMBERLOG
            else:
                return math.log(1 / (1000000 * (10 ** (3 * len(w)))))

    def segment_chinese(self, text):
        # Iterative DP Viterbi
        n = len(text)
        if n == 0:
            return []
            
        # dp[i] will store a list of tuples: (probability, prev_index, last_word)
        # representing the best segmentation up to index i
        dp = [{} for _ in range(n + 1)]
        dp[0] = {"": (0.0, -1)} # prev_word -> (prob, prev_index)
        
        for i in range(1, n + 1):
            max_len = min(self.maxdlen, i)
            for j in range(max(0, i - max_len), i):
                word = text[j:i]
                
                # Check heuristics to skip nonsensical splits
                if j > 0 and i < n:
                    fc = word[-1]
                    rc = text[i] # First char of remainder
                    if (fc in self.numberdesc and rc in self.numberdesc) or \
                       (fc in self.cforeign and rc in self.cforeign) or \
                       (fc in self.numberdesc and rc in self.ctime):
                        continue
                        
                for prev_word, (prev_prob, prev_idx) in dp[j].items():
                    prob = prev_prob + self.cPw(word, prev_word)
                    
                    if word not in dp[i] or prob > dp[i][word][0]:
                        dp[i][word] = (prob, j)
                        
        if not dp[n]:
            # Fallback if all paths skipped (very rare, usually heuristics are safe)
            return [text[i:i+1] for i in range(n)]
            
        # Find best end state
        best_last_word = max(dp[n].keys(), key=lambda w: dp[n][w][0])
        
        # Backtrack
        words = []
        curr_idx = n
        curr_word = best_last_word
        
        while curr_idx > 0:
            words.append(curr_word)
            _, prev_idx = dp[curr_idx][curr_word]
            
            if prev_idx == 0:
                break
                
            # Find the word that ended at prev_idx and led to curr_word
            # This is slightly tricky, we need to find which prev_word at prev_idx gave the max prob for curr_word
            best_prev = ""
            best_p = float('-inf')
            for pw, (p_prob, p_idx) in dp[prev_idx].items():
                trans_prob = p_prob + self.cPw(curr_word, pw)
                if trans_prob > best_p:
                    best_p = trans_prob
                    best_prev = pw
            curr_word = best_prev
            curr_idx = prev_idx
            
        return words[::-1]

    def segment_line(self, line):
        outlines = []
        linelen = len(line)
        i = 0
        while i < linelen:
            char = line[i]
            if char.isspace():
                i += 1
                continue
                
            if self.is_chinese_char(char):
                next_idx = i
                while next_idx < linelen and self.is_chinese_char(line[next_idx]):
                    next_idx += 1
                chinese_str = line[i:next_idx]
                words = self.segment_chinese(chinese_str)
                outlines.extend(words)
                i = next_idx
            else:
                if char == '<':
                    sgmlend = line.find('>', i + 1)
                    if sgmlend > 0:
                        curwlen = sgmlend + 1 - i
                        outlines.append(line[i:i+curwlen])
                        i += curwlen
                        continue
                if re.match(r'\w', char):
                    j = i + 1
                    while j < linelen:
                        nextchar = line[j]
                        if self.is_chinese_char(nextchar):
                            break
                        if not re.match(r'[\w.:/%$-]', nextchar):
                            break
                        j += 1
                    # check date
                    if j < linelen and line[j] in self.ctime:
                        if re.match(r'^[\d' + self.cnumbers + r'-]+$', line[i:j]):
                            j += 1
                    outlines.append(line[i:j])
                    i = j
                else:
                    j = i + 1
                    while j < linelen:
                        nextchar = line[j]
                        if re.match(r'\w', nextchar) or self.is_chinese_char(nextchar):
                            break
                        j += 1
                    outlines.append(line[i:j])
                    i = j
                    
        return outlines

    def tokenize_chinese(self, text):
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        result = []
        for s in sentences:
            tokens = self.segment_line(s)
            if tokens:
                result.append(tokens)
        return result

_instance = None

def get_tokenizer():
    global _instance
    if _instance is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tt_dir = os.path.join(base_dir, 'treetagger', 'tokenizer', 'chinese', 'zh-tokenise', 'zh-tokenise')
        unigram_file = os.path.join(tt_dir, 'chinese-3c.utf8')
        bigram_file = os.path.join(tt_dir, 'chinese-2c.utf8')
        
        # Load only if files exist
        if os.path.exists(unigram_file):
            _instance = ChineseSegmenter(unigram_file, bigram_file)
    return _instance

def tokenize_chinese_text(text):
    tokenizer = get_tokenizer()
    if tokenizer:
        return tokenizer.tokenize_chinese(text)
    return None
