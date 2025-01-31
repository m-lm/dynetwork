import preprocess
import relationships
import snapshots
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["preprocess", "relationships", "snapshots", "all"], nargs="?", const="all", default="all")
    args = parser.parse_args()

    if args.op == "all":
        preprocess.preprocess()
        relationships.extract_relationships()
        snapshots.generate_snapshots()
    elif args.op == "preprocess":
        preprocess.preprocess()
    elif args.op == "relationships":
        relationships.extract_relationships()
    elif args.op == "snapshots":
        snapshots.generate_snapshots()


if __name__ == "__main__":
    main()