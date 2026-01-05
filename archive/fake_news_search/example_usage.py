from retrieval_pipeline import retrieve_evidence_for_claim

def main():
    claim = input("Enter a claim to retrieve evidence for:\n> ").strip()
    print(f"\nSearching evidence for: \"{claim}\"\n")

    evidence_list = retrieve_evidence_for_claim(claim, num_results=5)
    if not evidence_list:
        print("No results found.")
        return

    for idx, ev in enumerate(evidence_list, 1):
        print(f"[{idx}] {ev['title']}")
        print(f"    URL    : {ev['url']}")
        print(f"    Snippet: {ev['snippet'][:120].strip()}...")
        if ev['raw_text']:
            print(f"    RawTxt : {ev['raw_text'][:120].strip()}...")
        else:
            print("    RawTxt : <could not retrieve body text>")
        print()

if __name__ == "__main__":
    main()
