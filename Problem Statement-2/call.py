"""Trigger an outbound call via the running server.

Usage:
    python3 call.py --to "+91XXXXXXXXXX" --name "Dewansh"
"""
import argparse
import requests
from config import SERVER_URL, DEFAULT_PATIENT

parser = argparse.ArgumentParser()
parser.add_argument("--to",         required=True,                      help="Patient phone number e.g. +91XXXXXXXXXX")
parser.add_argument("--name",       default=DEFAULT_PATIENT["name"])
parser.add_argument("--medication", default=DEFAULT_PATIENT["medication"])
parser.add_argument("--dosage",     default=DEFAULT_PATIENT["dosage"])
args = parser.parse_args()

resp = requests.post(f"{SERVER_URL}/make-call", json={
    "to":         args.to,
    "name":       args.name,
    "medication": args.medication,
    "dosage":     args.dosage,
})
print(resp.json())
