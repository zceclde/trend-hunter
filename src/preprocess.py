import re
from typing import List, Dict, Any
import opencc

class TextPreprocessor:
    def __init__(self):
        # OpenCC converter for Traditional to Simplified Chinese
        self.converter = opencc.OpenCC('t2s')
        
        # Regex patterns for cleaning
        self.img_pattern = re.compile(r'<image:[a-f0-9]+>', re.IGNORECASE)
        self.doge_pattern = re.compile(r'<emoji:(?:doge|二哈)>', re.IGNORECASE)
        self.emoji_pattern = re.compile(r'<emoji:[^>]+>', re.IGNORECASE)
        self.mention_pattern = re.compile(r'@[^\s@]+')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_artifact_pattern = re.compile(r'i2\s+bahamut\s+tw\s+editor\s+emotion\s*(?:gif)?', re.IGNORECASE)
        self.kaomoji_junk = re.compile(r'[QAQorzhhhh2333\(\)\（\）\<\>]{4,}', re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        # 1. Preserve universal gaming sentiment & sarcasm markers
        text = re.sub(r'<emoji:(?:doge|二哈)>', ' [doge] ', text, flags=re.I)
        text = re.sub(r'<emoji:(?:huaji|滑稽)>', ' [滑稽] ', text, flags=re.I)
        text = re.sub(r'(?:<emoji:(?:clown|小丑)>|🤡)', ' [小丑] ', text, flags=re.I)
        text = re.sub(r'(?:<emoji:(?:sweat|流汗|汗)>|😅)', ' [流汗] ', text, flags=re.I)
        text = re.sub(r'(?:<emoji:(?:cry|大哭|哭)>|😭)', ' [大哭] ', text, flags=re.I)
        text = re.sub(r'(?:<emoji:(?:watermelon|吃瓜)>|🍉)', ' [吃瓜] ', text, flags=re.I)
        
        # Normalize viral slang homophone puns (e.g. 蚌埠住了 -> 绷不住了)
        text = text.replace('蚌埠住了', '绷不住了')
        
        # 2. Strip images, other remaining emojis, mentions, and urls
        text = self.img_pattern.sub('', text)
        text = self.emoji_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.html_artifact_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        
        # 3. Convert Traditional Chinese to Simplified Chinese
        text = self.converter.convert(text)
        
        # 4. Normalize excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches messages by stitching parent context, normalizing Chinese variants,
        and tagging bot/spam metadata.
        """
        # Map ID to raw content for reply stitching
        id_to_msg = {m["id"]: m for m in messages}
        
        processed = []
        for m in messages:
            msg_id = m["id"]
            parent_id = m.get("parent_id")
            raw_title = m.get("title") or ""
            raw_content = m.get("content") or ""
            raw_full = (raw_title + " " + raw_content).strip()
            
            cleaned_content = self.clean_text(raw_full)
            user_name = m.get("user_name") or ""
            
            # Check for known automated bot signatures
            is_bot = False
            if "管家" in user_name or "Q群管家" in raw_full or ("欢迎来到" in raw_full and "群公告" in raw_full):
                is_bot = True
                
            # Reply Context Stitching: only stitch when the child reply is short/context-dependent
            # and the parent is a peer post rather than a generic '详情见图' official bulletin
            stitched_text = cleaned_content
            if parent_id and parent_id in id_to_msg:
                parent_msg = id_to_msg[parent_id]
                parent_raw = (parent_msg.get("title") or "") + " " + (parent_msg.get("content") or "")
                parent_cleaned = self.clean_text(parent_raw)
                is_official_bulletin = "详情见图" in parent_cleaned
                
                if not is_official_bulletin and (
                    len(cleaned_content) <= 10
                    or any(w in cleaned_content for w in ["确实", "+1", "同意", "真实", "我也"])
                ):
                    stitched_text = f"{cleaned_content} {parent_cleaned[:60]}"
            
            processed.append({
                "id": msg_id,
                "parent_id": parent_id,
                "platform": m.get("platform", "Unknown"),
                "channel": m.get("channel", "Unknown"),
                "user_name": user_name,
                "created_at": m.get("created_at", ""),
                "date": m.get("created_at", "")[:10],
                "raw_text": raw_full,
                "cleaned_text": cleaned_content,
                "stitched_text": stitched_text,
                "is_bot": is_bot
            })
            
        return processed

