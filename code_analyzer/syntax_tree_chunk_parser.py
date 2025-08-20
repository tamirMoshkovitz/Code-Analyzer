from tree_sitter import Parser
from tree_sitter_languages import get_language as _ts_get_language
from typing import Iterable, Tuple, List


class SyntaxTreeChunkParser:
    """A parser that splits code into chunks based on the syntax tree.

    This parser uses Tree-sitter to parse code and split it into chunks
    based on the abstract syntax tree (AST) nodes, while respecting line spans.
    It can also fall back to a simple greedy line-based chunking strategy if
    parsing fails or no nodes are found.
    """

    def __init__(self, lang_name: str, max_chars: int):
        self._lang_name = lang_name
        self._max_chars = max_chars
        self._build_ts_parser(lang_name)

    def parse_code_into_chunks(self, code: str, start_index: int = 0) -> Iterable[str]:
        """Parse the given code into chunks based on the syntax tree.
        Parameters
        ----------
        code : str
            The source code to parse.
        start_index : int, optional
            The line index to start parsing from. Defaults to 0.
        Yields
        -------
        str
            Chunks of code, each containing a header with line numbers.
        """
        tree = self._parser.parse(code.encode("utf-8"))
        root = tree.root_node
        lines = code.splitlines(keepends=True)

        start_index = max(0, min(start_index, len(lines)))

        units: List[Tuple[int, int, int]] = []  # (start_line, end_line_inclusive, approx_len)
        for child in root.children:
            if child.start_byte >= child.end_byte:
                continue
            node_start = child.start_point[0]
            node_end = child.end_point[0]
            if node_end < start_index:
                continue
            node_start = max(node_start, start_index)
            node_bytes = child.end_byte - child.start_byte
            units.append((node_start, node_end, node_bytes))

        if not units:
            chunk = "".join(
                f"Line {lineno + 1}: \t{line}"
                for lineno, line in enumerate(lines[start_index:], start=start_index)
            )
            if chunk:
                yield chunk
            return

        # Greedy pack into chunks by character budget
        i = 0
        n = len(units)
        # Initial start is the start of the first node
        next_start_line = max(start_index, units[0][0])

        while i < n:
            chunk_start = max(start_index, next_start_line)
            total_chars = 0
            end_line_incl = units[i][1]
            j = i
            while j < n:
                node_start, node_end_incl, node_chars = units[j]
                # If this node begins far after our chosen start (due to overlap), advance start once
                if j == i and node_start < chunk_start:
                    # Keep chunk_start as-is; the chunk will slice from chunk_start
                    pass
                if total_chars + node_chars > self._max_chars and j > i:
                    break
                if total_chars + node_chars > self._max_chars and j == i:
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
            chunk_start = min(chunk_start, end_exclusive - 1)
            chunk_start = max(chunk_start, start_index)

            chunk = "".join(
                f"Line {lineno + 1}: \t{line}"
                for lineno, line in enumerate(lines[chunk_start:end_exclusive], start=chunk_start)
            )
            yield chunk

            if end_exclusive >= len(lines):
                break

            # Prepare for next chunk
            next_start_line = end_exclusive

            # Move i to the first node that intersects or is after next_start_line
            while i < n and units[i][1] < next_start_line:
                i += 1
            if i >= n:
                break

    def _build_ts_parser(self, lang_name: str) -> None:
        """Initialize the Tree-sitter parser with the specified language.
        Parameters
        ----------
        lang_name : str
            The name of the programming language to use for parsing.
        Raises
        -------
        ValueError
            If the specified language is not supported by Tree-sitter.
        """
        self._parser = Parser()
        ts_lang = _ts_get_language(lang_name)
        self._parser.set_language(ts_lang)
