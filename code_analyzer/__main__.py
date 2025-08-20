import sys
from code_analyzer import analyzer

"""Main entry point for the code analyzer, in order to run from commandline without docker."""
if __name__ == "__main__":
    sys.exit(analyzer.main())