"""
Run: python diagnose_get_known_people.py

Directly calls the exact line get_known_people() swallows, so we can see
the real exception instead of a silent empty set.
"""
import rag
import traceback

print("Collection count:", rag.get_collection().count())

try:
    all_data = rag.get_collection().get(include=["metadatas"])
    print("get() succeeded.")
    metas = all_data.get("metadatas", [])
    print(f"Returned {len(metas)} metadata entries.")
    people = set()
    empty_person_count = 0
    for meta in metas:
        if meta and meta.get("person"):
            people.add(meta["person"])
        else:
            empty_person_count += 1
    print("Distinct person keys found:", sorted(people))
    print("Entries with no/empty person field:", empty_person_count)
except Exception:
    print("get() RAISED AN EXCEPTION — this is what get_known_people() was silently swallowing:")
    traceback.print_exc()
