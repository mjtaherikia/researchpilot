
import argparse
import json
import sys

from researchpilot.corpus import (
    CorpusError,
    get_document,
    search_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="researchpilot",
        description="ResearchPilot local corpus utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="Search the local document corpus",
    )
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)

    get_parser = subparsers.add_parser(
        "get",
        help="Retrieve a document by ID",
    )
    get_parser.add_argument("document_id")

    args = parser.parse_args()

    try:
        if args.command == "search":
            result = search_documents(args.query, args.top_k)
        else:
            result = get_document(args.document_id)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except (ValueError, KeyError, CorpusError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
