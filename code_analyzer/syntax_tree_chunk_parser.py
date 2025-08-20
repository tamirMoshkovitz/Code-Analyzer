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

        start_index = max(0, min(len(lines), start_index))

        units = self._collect_units(root, start_index)

        if not units:
            chunk = "".join(
                f"Line {lineno + 1}: \t{line}"
                for lineno, line in enumerate(lines[start_index:], start=start_index)
            )
            if chunk:
                yield chunk
            return

        for chunk_start, end_exclusive in self._iter_chunk_ranges(units, len(lines), start_index):
            chunk = "".join(
                f"Line {lineno + 1}: \t{line}"
                for lineno, line in enumerate(lines[chunk_start:end_exclusive], start=chunk_start)
            )
            yield chunk


    def _collect_units(self, root, start_index: int) -> List[Tuple[int, int, int]]:
        """Collect units from the syntax tree root node.
        Parameters
        ----------
        root : tree_sitter.Node
            The root node of the syntax tree.
        start_index : int
            The line index to start collecting units from.
        Returns
        -------
        List[Tuple[int, int, int]]
            A list of tuples, each containing (start_line, end_line_inclusive, byte_length)."""
        units: List[Tuple[int, int, int]] = []
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
        return units

    def _iter_chunk_ranges(self, units: List[Tuple[int, int, int]], n_lines: int, start_index: int
                           )-> Iterable[Tuple[int, int]]:
        """Iterate over ranges of code chunks based on the collected units.
        Parameters
        ----------
        units : List[Tuple[int, int, int]]
            A list of tuples, each containing (start_line, end_line_inclusive, byte_length).
        n_lines : int
            The total number of lines in the code.
        start_index : int
            The line index to start iterating from.
        Yields
        -------
        Tuple[int, int]
            A tuple containing the start line and end line (exclusive) of each chunk.
        """
        if not units:
            return
        i = 0
        n = len(units)
        next_start_line = max(start_index, units[0][0])

        while i < n:
            chunk_start = max(start_index, next_start_line)
            total_chars = 0
            end_line_incl = units[i][1]
            j = i
            while j < n:
                node_start, node_end_incl, node_chars = units[j]
                if j == i and node_start < chunk_start:
                    pass
                if total_chars + node_chars > self._max_chars and j > i:
                    break
                if total_chars + node_chars > self._max_chars and j == i:
                    total_chars = node_chars
                    end_line_incl = node_end_incl
                    j += 1
                    break
                total_chars += node_chars
                end_line_incl = max(end_line_incl, node_end_incl)
                j += 1

            end_exclusive = min(end_line_incl + 1, n_lines)
            chunk_start = min(chunk_start, end_exclusive - 1)
            chunk_start = max(chunk_start, start_index)
            yield chunk_start, end_exclusive

            if end_exclusive >= n_lines:
                break

            next_start_line = end_exclusive
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
