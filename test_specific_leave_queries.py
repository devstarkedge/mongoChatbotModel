import os, sys, django, re
from datetime import datetime, time as dtime
from bson import ObjectId
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, '.')
django.setup()

from rag.services import mongo_service, structured_query

db = mongo_service.get_db()
ws_id = "6a759a7611d309503d9cee15"

def test_query(text):
    print(f"\n==========================================")
    print(f"QUERY: '{text}'")
    period = structured_query._extract_period(text)
    print("Period extracted:", period)
    
    is_who_on_leave = bool(re.search(r"\bwho\s+is\s+on\s+leave\b|\bis\s+.*?\s+on\s+leave\b", text, re.IGNORECASE))
    is_superlative = bool(structured_query._LEAVE_COUNT_SUPERLATIVE_PATTERN.search(text))
    print(f"is_who_on_leave: {is_who_on_leave}, is_superlative: {is_superlative}")
    
    clauses = []
    if period and "start" in period and "end" in period:
        # Extend end to 23:59:59
        p_start = datetime.combine(period["start"].date(), dtime.min)
        p_end = datetime.combine(period["end"].date(), dtime.max)
        clauses.append({
            "$or": [
                {"startDate": {"$gte": p_start, "$lte": p_end}},
                {"endDate": {"$gte": p_start, "$lte": p_end}},
            ]
        })
    if is_who_on_leave:
        clauses.append({"status": re.compile(r"^approved$", re.IGNORECASE)})
        
    final_query = structured_query._and_merge(
        structured_query._workspace_filter("leaverequests", ws_id, db),
        *clauses
    )
    docs = list(db["leaverequests"].find(final_query))
    print("Found docs:", len(docs))
    for d in docs:
        u = db["users"].find_one({"_id": d.get("requester")})
        uname = u.get("name") if u else str(d.get("requester"))
        lt = db["leavetypes"].find_one({"_id": d.get("leaveType")})
        ltname = lt.get("name") if lt else str(d.get("leaveType"))
        print(f"  • {uname} — {ltname} ({d.get('startDate')} to {d.get('endDate')}) [{d.get('status')}]")

test_query("who is on leave on 13 nov")
test_query("who has least leaves in oct")
