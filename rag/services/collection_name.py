#!/usr/bin/env python3
"""
List all collection names from a MongoDB URI.

Usage:
    python list_collections.py "mongodb://user:pass@host:27017/mydb"
    python list_collections.py "mongodb+srv://user:pass@cluster.mongodb.net/mydb"
"""

import sys
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, OperationFailure


def get_collection_names(mongo_uri: str) -> list[str]:
    """Connect to MongoDB via URI and return sorted collection names."""
    # Extract DB name from URI (everything after the last '/', minus query string)
    parsed = urlparse('mongodb+srv://shopifychdev_db_user:Starkedge123@cluster0.dmixu3k.mongodb.net/test_copy')
    db_name = parsed.path.lstrip("/").split("?")[0] or None

    if not db_name:
        raise ValueError(
            "No database name found in URI. "
            "Append '/<dbname>' to the connection string."
        )

    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # verify connection

        db = client[db_name]
        names = db.list_collection_names()
        return sorted(names)

    except ConfigurationError as e:
        raise ValueError(f"Invalid MongoDB URI: {e}") from e
    except ConnectionFailure as e:
        raise ConnectionError(f"Could not connect to MongoDB: {e}") from e
    except OperationFailure as e:
        raise PermissionError(
            f"Authentication/authorization failed: {e}"
        ) from e
    finally:
        if client is not None:
            client.close()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mongodb-uri>", file=sys.stderr)
        return 1

    uri = sys.argv[1]
    try:
        collections = get_collection_names(uri)
    except (ValueError, ConnectionError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if not collections:
        print("(no collections found)")
        return 0

    for name in collections:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())