import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple

def parse_iso_datetime(dt_str: str) -> datetime:
    """Parses ISO timestamp string into datetime object."""
    # Format e.g.: 2026-06-01T00:19:51+08:00 or with microseconds
    # Python 3.11+ fromisoformat handles standard timezone offsets
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        # Fallback manual parsing if format varies
        clean_str = dt_str[:19]
        return datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")

def load_data(
    input_path: str,
    as_of: str,
    versions_path: str = "versions.csv"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads voices.jsonl and versions.csv strictly enforcing the temporal gate.
    Any message created after the as_of date is discarded immediately.
    """
    # Enforce end-of-day cutoff for the specified as_of date
    as_of_cutoff_str = f"{as_of}T23:59:59.999999+08:00"
    as_of_date_prefix = as_of # "YYYY-MM-DD"
    
    valid_messages = []
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            created_at = msg.get("created_at", "")
            
            # Strict chronological gate:
            # 1. Fast string check: if date prefix > as_of, drop
            msg_date = created_at[:10]
            if msg_date > as_of_date_prefix:
                continue
            # 2. Strict timestamp comparison within the same date
            if created_at > as_of_cutoff_str:
                continue
                
            valid_messages.append(msg)
            
    # Load versions.csv if available
    versions = []
    if os.path.exists(versions_path):
        with open(versions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_date = row.get("date", "").strip()
                if event_date and event_date <= as_of_date_prefix:
                    versions.append({
                        "date": event_date,
                        "version": row.get("version", "").strip(),
                        "notes": row.get("notes", "").strip()
                    })
                    
    # Sort messages chronologically
    valid_messages.sort(key=lambda x: x.get("created_at", ""))
    
    return valid_messages, versions

