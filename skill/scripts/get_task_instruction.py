#!/usr/bin/env python3
"""
Get task instruction from the server.
"""
import requests
import json
import sys

SERVER_URL = "http://127.0.0.1:8000"

def get_instruction():
    try:
        response = requests.get(f"{SERVER_URL}/instruction", timeout=10)
        if response.status_code == 200:
            result = response.json()
            instruction = result.get("task_instruction", "")
            print(f"Instruction: {instruction}")
            return {"task_instruction": instruction}
        else:
            print(f"Failed to get instruction: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Getting instruction failed: {e}")
        sys.exit(1)
 
if __name__ == "__main__":
    get_instruction()
