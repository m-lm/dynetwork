import preprocess
import relationships
import visualization
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["preprocess", "relationships", "all"], nargs="?", const="all", default="all")
    args = parser.parse_args()

    if args.op == "all":
        preprocess.preprocess()
        relationships.extract_relationships()
    elif args.op == "preprocess":
        preprocess.preprocess()
    elif args.op == "relationships":
        relationships.extract_relationships()

if __name__ == "__main__":
    main()