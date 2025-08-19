import argparse
import sys
from pathlib import Path
from hf_model_downloader import HFModelDownloader
from hf_model_pipeline import HFModelPipeline
# Only import Parser from tree_sitter
from tree_sitter import Parser
# Try to import get_language from tree_sitter_languages if available
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception:
    _ts_get_language = None
from typing import Iterable, Tuple, List

CPP_FILE_SUFFICES = ('.cpp', '.cc', '.cxx', '.hpp', '.h', '.hh', '.hxx')
SUPPORTED_LANGUAGES = ('c', 'cpp')

MAX_LINES_PER_CHUNK =400 # this limit should be safe for gemma-3.1b-it with max_length of 8192 tokens, could be overridden by command line argument
MAX_CHARS_PER_CHUNK = 12000 # similar

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
GEMMA3_1B_IT_MODEL_ID = "google/gemma-3-1b-it"

def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the C/C++ security analyzer chunker.
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="C/C++ security analyzer chunker")
    parser.add_argument("code_file_path", help="Path to the C/C++ file to analyze")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it", help="HF model ID (overrides default)")
    parser.add_argument("--model-path", default=None, help="Local model path (if already downloaded)")
    parser.add_argument("--max-lines", type=int, default=MAX_LINES_PER_CHUNK, help="Max lines of code per prompt")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_CHUNK,
                        help="Max characters per chunk (hard cap)")
    parser.add_argument("--prompt", type=str, default=PROMPT,
                        help="Prompt template to use for the model (default is Gemma-3.1b-it prompt), should contain {code} placeholder")
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

def _get_language_from_path(file_path: str) -> str:
    """
    Determine the Tree-sitter language based on the file extension.
    Returns 'c' for `.c` files, 'cpp' for C++-related extensions,
    otherwise prints error to stderr and exits.
    Parameters
    ----------
    file_path : str
        Path to the code file.
    Returns
    -------
    str
        The language identifier ('c' or 'cpp').
    Raises
    ------
    SystemExit
        If the file extension is not supported.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == '.c':
        return 'c'
    elif suffix in CPP_FILE_SUFFICES:
        return 'cpp'
    else:
        print(f"Unsupported file: {suffix}", file=sys.stderr)
        sys.exit(1)

def _get_model_path(args: argparse.Namespace) -> str:
    """Get the model path based on command line arguments.
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    Returns
    -------
    str
        Path to the model.
    """
    model_id = args.model_id or GEMMA3_1B_IT_MODEL_ID
    if args.model_path:
        return args.model_path
    else:
        return HFModelDownloader.get_local_model_path(model_id)


def _build_ts_parser(lang_name: str) -> Parser:
    """Build and return a Tree-sitter parser for the specified language.
    Parameters
    ----------
    lang_name : str
        The programming language ('c' or 'cpp').
    Returns
    -------
    Parser
        A Tree-sitter parser configured for the specified language.
    Raises
    ------
    ValueError
        If an unsupported language is specified.
    """
    if lang_name not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang_name}")

    if _ts_get_language is None:
        raise RuntimeError(
            "tree_sitter_languages is not installed. Install it or pin requirements as in Option A."
        )

    parser = Parser()
    ts_lang = _ts_get_language('c' if lang_name == 'c' else 'cpp')
    parser.set_language(ts_lang)
    return parser


def ast_split_into_chunks(code_str: str,
                          lines: List[str],
                          lang_name: str,
                          max_chars: int = MAX_CHARS_PER_CHUNK,
                          ) -> Iterable[Tuple[int, int]]:
    """
    Split the given code string into chunks based on AST nodes and line spans.

    Strategy
    --------
    1) Parse the code with tree-sitter for the given language ("c" or "cpp").
    2) Collect top-level nodes (children of the translation unit) with their
       line spans and byte spans.
    3) Greedily pack consecutive nodes into a chunk until adding the next node
       would exceed `max_chars`.
    4) Yield (start_idx, end_idx) as 0-based **line indices** where `end_idx` is
       *exclusive* so you can slice like `lines[start_idx:end_idx]`.
    5) Apply `overlap` lines between consecutive chunks.

    Fallback
    --------
    If parsing fails or no nodes are found, fall back to a simple line-based
    greedy chunker that respects `max_chars` and `overlap`.
    """
    # Helper: simple line-based fallback generator
    def _fallback_chunks() -> Iterable[Tuple[int, int]]:
        n = len(lines)
        i = 0
        while i < n:
            total = 0
            start = i
            while i < n and total + len(lines[i]) <= max_chars:
                total += len(lines[i])
                i += 1
            end = max(i, start + 1)  # ensure forward progress
            yield start, end
            if end >= n:
                break
            i = end

    # Quick exits / guards
    if not code_str:
        yield from ()
        return

    parser = _build_ts_parser(lang_name)
    try:
        tree = parser.parse(code_str.encode("utf-8"))
        root = tree.root_node
    except Exception:
        # If anything goes wrong, use fallback
        yield from _fallback_chunks()
        return

    # Collect top-level nodes and merge leading comment blocks into the next node
    # We produce "units" where each unit is either:
    #  - a non-comment node optionally preceded by an adjacent comment block (no blank lines between), or
    #  - a standalone comment block if it is not followed by code.
    units: List[Tuple[int, int, int]] = []  # (start_line, end_line_inclusive, approx_len)

    def _only_blank_between(lo: int, hi: int) -> bool:
        """Return True if lines[lo:hi+1] are empty/whitespace-only (bounds inclusive)."""
        if lo > hi:
            return True
        for k in range(max(0, lo), min(len(lines) - 1, hi) + 1):
            if lines[k].strip():
                return False
        return True

    try:
        pending_comment_start: int | None = None
        pending_comment_end: int | None = None
        pending_comment_bytes: int = 0

        for child in root.children:
            # Ignore zero-length / extraneous nodes
            if child.start_byte >= child.end_byte:
                continue

            node_start = child.start_point[0]   # 0-based
            node_end = child.end_point[0]       # inclusive
            node_bytes = child.end_byte - child.start_byte
            is_comment = getattr(child, "type", "") == "comment"

            if is_comment:
                # Start or extend a pending leading comment block
                if pending_comment_start is None:
                    pending_comment_start = node_start
                    pending_comment_end = node_end
                    pending_comment_bytes = node_bytes
                else:
                    # Extend if contiguous or separated only by blank lines
                    if (pending_comment_end is not None
                        and _only_blank_between(pending_comment_end + 1, node_start - 1)):
                        pending_comment_end = max(pending_comment_end, node_end)
                        pending_comment_bytes += node_bytes
                    else:
                        # Previous comment block stands alone (no following code yet)
                        units.append((pending_comment_start, pending_comment_end, pending_comment_bytes))
                        pending_comment_start = node_start
                        pending_comment_end = node_end
                        pending_comment_bytes = node_bytes
                continue

            # Non-comment node
            # If we have a pending comment block directly leading into this node (no non-blank gap),
            # merge it by extending the unit start upward.
            if pending_comment_start is not None and pending_comment_end is not None:
                if _only_blank_between(pending_comment_end + 1, node_start - 1):
                    unit_start = pending_comment_start
                    unit_bytes = pending_comment_bytes + node_bytes
                    # Clear pending comment since it's consumed
                    pending_comment_start = None
                    pending_comment_end = None
                    pending_comment_bytes = 0
                else:
                    # The pending comment block is separated by code/blank run → emit it standalone
                    units.append((pending_comment_start, pending_comment_end, pending_comment_bytes))
                    pending_comment_start = None
                    pending_comment_end = None
                    pending_comment_bytes = 0
                    unit_start = node_start
                    unit_bytes = node_bytes
            else:
                unit_start = node_start
                unit_bytes = node_bytes

            # Emit the merged unit (comment block + this node)
            units.append((unit_start, node_end, unit_bytes))

        # If comments remain at the end without a following node, emit them as a standalone unit
        if pending_comment_start is not None and pending_comment_end is not None:
            units.append((pending_comment_start, pending_comment_end, pending_comment_bytes))

    except Exception:
        yield from _fallback_chunks()
        return

    if not units:
        yield from _fallback_chunks()
        return

    # Greedy pack into chunks by character budget
    i = 0
    n = len(units)
    # Initial start is the start of the first node
    next_start_line = max(0, units[0][0])

    while i < n:
        start_line = max(0, next_start_line)
        total_chars = 0
        end_line_incl = units[i][1]
        j = i
        while j < n:
            node_start, node_end_incl, node_chars = units[j]
            # If this node begins far after our chosen start (due to overlap), advance start once
            if j == i and node_start < start_line:
                # Keep start_line as-is; the chunk will slice from start_line
                pass
            if total_chars + node_chars > max_chars and j > i:
                break
            if total_chars + node_chars > max_chars and j == i:
                # Ensure forward progress even if a single node exceeds max_chars
                # by letting it be alone in a chunk.
                total_chars = node_chars
                end_line_incl = node_end_incl
                j += 1
                break
            total_chars += node_chars
            end_line_incl = max(end_line_incl, node_end_incl)
            j += 1

        end_exclusive = end_line_incl + 1
        # Clamp to file length
        end_exclusive = min(end_exclusive, len(lines))
        start_line = min(start_line, end_exclusive - 1)

        yield start_line, end_exclusive

        if end_exclusive >= len(lines):
            break

        # Prepare for next chunk
        next_start_line = end_exclusive

        # Move i to the first node that intersects or is after next_start_line
        while i < n and units[i][1] < next_start_line:
            i += 1
        if i >= n:
            break


def _analyze_code(prompt: str, lang: str, lines_num: int) -> None:
    """Analyze the code using the specified language and number of lines.
    Parameters
    ----------
    prompt : str
        The prompt template to use for the model, should contain {code} placeholder.
    lang : str
        The programming language of the code (e.g., 'c' or 'cpp').
    lines_num : int
        The total number of lines in the code.
    """
    print("\nAnalyzer output:\n")
    chunk_iter = ast_split_into_chunks(code, lines, max_chars=args.max_chars, lang_name=lang)
    for start_idx, end_idx in chunk_iter:
        chunk = "".join(
            f"Line {lineno + 1}: \t{line}"
            for lineno, line in enumerate(lines[start_idx:end_idx], start=start_idx)
        )

        # Append to header context for better model understanding (avoid duplicating if we're still inside the header)
        if start_idx > 0 and header_ctx:
            chunk = header_ctx + "\n" + chunk

        chunk_prompt = prompt.format(
            code=chunk,
            start_line=start_idx + 1,
            total_lines=lines_num,
        )
        result = pipeline.get_model_response(chunk_prompt)
        print(result)

if __name__ == "__main__":
    args = _parse_arguments()
    code_path = args.code_file_path
    code = _read_code_file(code_path)

    lines = code.splitlines(keepends=True)
    number_of_lines = len(lines)
    header_ctx, header_stop_idx = _extract_header_context(lines)

    file_type = _get_language_from_path(code_path)

    model_path = _get_model_path(args)

    pipeline = HFModelPipeline(model_path)

    _analyze_code(args.prompt, file_type, number_of_lines)

    print("\nAnalysis complete.\n")