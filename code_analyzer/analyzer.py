import argparse
import sys
import traceback
from pathlib import Path
from syntax_tree_chunk_parser import SyntaxTreeChunkParser
from hf_model_downloader import HFModelDownloader
from hf_model_pipeline import HFModelPipeline


SUPPORTED_FILES_TO_LANGUAGES = {".c": "c", # language names should correspond to tree-sitter language names
                                ".cpp": "cpp",
                                ".cc": "cpp",
                                ".cxx": "cpp",
                                ".h": "cpp",
                                ".hpp": "cpp",
                                ".hh": "cpp",
                                ".hxx": "cpp"}

DEFAULT_MAX_CHARS_PER_CHUNK = 8000  # this limit should be safe for gemma-3.1b-it with max_length of 8192 tokens, could be overridden by command line argument

PROMPT = """
<start_of_turn>user
You are a C/C++ security expert. Follow these steps:
1. Identify all vulnerabilities in the provided, numbered code.
2. Reply using exactly this template (one per vulnerability):
vulnerability at line <line number>: <desc>; fix suggestion: <fix>
No extra text or commentary.

Now analyze this code (header context may be repeated across chunks):
<<CODE>>
{code}
<<END CODE>>
<end_of_turn><start_of_turn>model
"""
DEFAULT_MODEL_ID = "google/gemma-3-1b-it"


class UnsupportedFileError(Exception):
    """Raised when an unsupported file extension is encountered."""

    def __init__(self, suffix: str):
        super().__init__(f"Unsupported file extension: {suffix}")
        self.suffix = suffix


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the C/C++ security analyzer chunker.
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="C/C++ security analyzer chunker")
    parser.add_argument("code_file_path", help="Path to the C/C++ file to analyze")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"HF model ID, default is {DEFAULT_MODEL_ID}")
    parser.add_argument("--model-path", default=None, help="Local model path - if already downloaded, use this instead of model ID")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK,
                        help=f"Max characters per code chunk (one chunk per prompt), default is {DEFAULT_MAX_CHARS_PER_CHUNK}")
    parser.add_argument("--prompt", type=str, default=PROMPT,
                        help="Prompt template to use for the model, should contain {code} placeholder, use --debug to see the full prompt")
    parser.add_argument("--treat-h-as-c", action="store_true", default=False,
                        help="Treat .h files as C code, default is False - .h files are treated as C++")
    parser.add_argument("--no-header-context", action="store_false", default=True, help="Do not add code header context to each code chunk for better model understanding")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    return parser.parse_args()


def _read_code_file(code_path: str) -> str:
    """Read and return the contents of the given code file.
    Parameters
    ----------
    code_path : str
        Path to the code file.
    Returns
    -------
    str
        Contents of the code file.
    """
    try:
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {code_path}")
        sys.exit(1)


def _extract_header_context(lines: list[str]) -> tuple[str, int]:
    """
    Pull top-of-file context that aids understanding (e.g., includes and using/typedefs).
    Stops once hit the first non-preprocessor/non-declaration block or an empty line run.
    Parameters
    ----------
    lines : list of str
        List of lines from the code file.
    Returns
    -------
    tuple
        A tuple containing the header context as a string and the index where the body starts.
        If no header is found, returns an empty string and the length of lines.
    """
    header = []
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            # keep a single blank line for spacing then stop
            header.append(line)
            return "".join(header), i + 1
        if _is_header_line(stripped_line):
            header.append(line)
            continue
        # stop at the first likely body line
        return "".join(header) + ("\n" if header and header[-1].strip() else ""), i
    return "".join(header), len(lines)


def _is_header_line(line: str) -> bool:
    """ Check if a line is part of the header context.
    Parameters
    ----------
    line : str
        The line to check.
    Returns
    -------
    bool
        True if the line is a preprocessor directive, type declaration, or namespace/class/struct definition.
    """
    return (line.startswith("#include") or line.startswith("#define") or line.startswith("using ") or
            line.startswith("typedef") or line.startswith("namespace ") or line.startswith("struct ") or
            line.startswith("class "))


def _get_language_from_path(file_path: str, mapping: dict[str, str]) -> str:
    """
    Determine the Tree-sitter language based on the file extension.
    Returns 'c' for `.c` files, 'cpp' for C++-related extensions
    Parameters
    ----------
    file_path : str
        Path to the code file.
    mapping : dict
        A mapping of file extensions to language identifiers.
        E.g., {".c": "c", ".cpp": "cpp", ".h": "cpp"}.
    Returns
    -------
    str
        The language identifier ('c' or 'cpp').
    Raises
    ------
    UnsupportedFileError
        If the file extension is not supported.
    """
    file_type = Path(file_path).suffix.lower()
    if file_type in mapping:
        return mapping[file_type]
    else:
        raise UnsupportedFileError(file_type)


def _get_model_path(model_id: str, model_path: str | None) -> str:
    """Get the model path based on command line arguments.
    Parameters
    ----------
    model_id : str
        The Hugging Face model ID (e.g., "google/gemma-3-1b-it").
    model_path : str or None
        Local path to the model if already downloaded, otherwise None.
    Returns
    -------
    str
        Path to the model.
    """
    if model_path:
        return model_path
    else:
        return HFModelDownloader.get_local_model_path(model_id)


def _analyze_code(pipeline: HFModelPipeline, prompt: str, lines_num: int, header_context: str,
                  header_stop_index: int, code: str, parser: SyntaxTreeChunkParser, debug: bool, no_header_context: bool) -> None:
    """Analyze the code using the specified language and number of lines.
    Parameters
    ----------
    pipeline : HFModelPipeline
        The Hugging Face model pipeline to use for analysis.
    prompt : str
        The prompt template to use for the model, should contain {code} placeholder.
    lines_num : int
        Total number of lines in the code.
    header_context : str
        Header context to prepend to each code chunk.
    header_stop_index : int
        Index where the header context ends in the code.
    code : str
        The code to analyze.
    parser : SyntaxTreeChunkParser
        The parser to use for breaking the code into chunks.
    debug : bool
        Whether to enable debug output.
    no_header_context : bool
        If True, do not prepend header context to each code chunk.
    """
    print("\nAnalyzer output:\n")
    start_index = header_stop_index if header_context else 0
    chunk_iter = parser.parse_code_into_chunks(code, start_index=start_index)
    for chunk in chunk_iter:
        if header_context and (not no_header_context):
            chunk = header_context + "\n" + chunk

        chunk_prompt = prompt.format(
            code=chunk,
            start_line=1,
            total_lines=lines_num,
        )
        result = pipeline.get_model_response(chunk_prompt, return_full_text=debug)
        print(result)


def _run_analyzer(args: argparse.Namespace) -> int:
    code_path = args.code_file_path
    code = _read_code_file(code_path)

    lines = code.splitlines(keepends=True)
    number_of_lines = len(lines)
    header_context, header_stop_index = _extract_header_context(lines)

    language_mapping = dict(SUPPORTED_FILES_TO_LANGUAGES)
    if args.treat_h_as_c:
        language_mapping[".h"] = "c"
    language = _get_language_from_path(code_path, language_mapping)

    model_path = _get_model_path(args.model_id, args.model_path)

    pipeline = HFModelPipeline(model_path)

    code_parser = SyntaxTreeChunkParser(language, args.max_chars)
    _analyze_code(pipeline, args.prompt, number_of_lines, header_context, header_stop_index, code,
                  code_parser, args.debug, args.no_header_context)

    print("\nAnalysis complete.\n")
    return 0


def main() -> int:
    args = _parse_arguments()
    try:
        return _run_analyzer(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"\nError: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
