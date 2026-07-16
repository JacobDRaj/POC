import json
import hashlib



def _make_query_cache_key(user_query, schema, conversation_memory=None):
    """Create a stable cache key for repeated user queries."""
    try:
        schema_str = json.dumps(schema, sort_keys=True)
        memory_str = json.dumps(conversation_memory or {}, sort_keys=True)
        combined = f"{user_query.strip().lower()}||{schema_str}||{memory_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    except Exception:
        return hashlib.sha256(user_query.strip().lower().encode('utf-8')).hexdigest()


