"""This module contains utilities for parsing Python source code. Its primary tasks include the
following: 1) stringifying Python source code; 2) traversing Abstract Syntax Trees, especially to
locate imports; and 3) preparing and cleaning source code for reuse

Related
-------
:mod:`hyperparameter_hunter.library_helpers.keras_optimization_helper`
    Uses :mod:`hyperparameter_hunter.utils.parsing_utils` to prepare for Keras optimization

Notes
-----
Many of these utilities are modified versions of utilities originally from the `Hyperas` library.
Thank you to the Hyperas creators, and contributors for their excellent work and fascinating
approach to Keras hyperparameter optimization. Without them, Keras hyperparameter optimization in
`hyperparameter_hunter` would be far less pretty"""
##################################################
# Import Miscellaneous Assets
##################################################
from ast import NodeVisitor, parse
from inspect import getsource
from operator import attrgetter
import os
import re


def stringify_model_builder(build_fn):
    """Get the stringified Python source code of `build_fn`

    Parameters
    ----------
    build_fn: Callable
        A Keras model-building function

    Returns
    -------
    build_fn_str: Strings
        A stringified version of `build_fn`"""
    build_fn_str = remove_comments(getsource(build_fn))
    return build_fn_str


def build_temp_model_file(build_fn_str, source_script):
    """Construct a string containing extracted imports from both `build_fn_str`

    Parameters
    ----------
    build_fn_str: Str
        The stringified source code of a callable
    source_script: Str
        Absolute path to a Python file. Expected to end with '.py', or '.ipynb'

    Returns
    -------
    temp_file_Str: Str
        Combination of extracted imports, and clean `build_fn_str` in Python script format"""
    source_script_contents = read_source_script(source_script)

    builder_imports = extract_imports(build_fn_str)
    source_imports = extract_imports(source_script_contents)

    cleaned_builder_str = remove_imports(remove_comments(build_fn_str))

    temp_file_str = ""
    temp_file_str += source_imports
    temp_file_str += builder_imports.replace("#coding=utf-8", "")
    temp_file_str += "\n\n"
    temp_file_str += cleaned_builder_str

    return temp_file_str


def read_source_script(filepath):
    """Read the contents of `filepath`

    Parameters
    ----------
    filepath: Str
        Absolute path to a Python file. Expected to end with '.py', or '.ipynb'

    Returns
    -------
    source: Str
        The contents of `filepath`"""
    if filepath.endswith(".ipynb"):
        with open(filepath, "r") as f:
            from nbconvert import PythonExporter
            import nbformat

            notebook = nbformat.reads(f.read(), nbformat.NO_CONVERT)
            exporter = PythonExporter()
            source, _ = exporter.from_notebook_node(notebook)
    else:
        with open(filepath, "r") as f:
            source = f.read()

    return source


def write_python_source(source_str, filepath="temp_modified.py"):
    """Save `source_str` to the file located at `filepath`

    Parameters
    ----------
    source_str: String
        The content to write to the file at `filepath`
    filepath: String
        The filepath of the file to which `source_str` should be written"""
    try:
        with open(filepath, "w") as f:
            f.write(source_str)
            f.close()
    except FileNotFoundError:
        os.mkdir(os.path.split(filepath)[0])
        write_python_source(source_str, filepath=filepath)


##################################################
# Modified Hyperas Assets (Utilities)
##################################################
class ImportParser(NodeVisitor):
    def __init__(self):
        # """(Taken from `hyperas.utils`)"""
        self.lines = []
        self.line_numbers = []

    def visit_Import(self, node):
        line = "import {}".format(self._import_names(node.names))
        self._visit_helper(line, node)

    def visit_ImportFrom(self, node):
        line = "from {}{} import {}".format(
            node.level * ".", node.module or "", self._import_names(node.names)
        )
        self._visit_helper(line, node)

    def _visit_helper(self, line, node):
        if self._import_asnames(node.names) != "":
            line += " as {}".format(self._import_asnames(node.names))
        self.line_numbers.append(node.lineno)
        self.lines.append(line)

    # noinspection PyMethodMayBeStatic
    def _import_names(self, names):
        return ", ".join(map(attrgetter("name"), names))

    # noinspection PyMethodMayBeStatic
    def _import_asnames(self, names):
        asname = map(attrgetter("asname"), names)
        return "".join(filter(None, asname))


def extract_imports(source):
    """(Taken from `hyperas.utils`). Construct a string containing all imports from `source`

    Parameters
    ----------
    source: String
        A stringified fragment of source code

    Returns
    -------
    imports_str: String
        The stringified imports from `source`"""
    tree = parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    import_lines = ["#coding=utf-8\n"]

    for line in import_parser.lines:
        if "print_function" in line:
            import_lines.append(line + "\n")
        elif ("_pydev_" in line) or ("java.lang" in line):
            continue  # Skip imports for PyCharm, and Eclipse
        else:
            import_lines.append("try:\n    {}\nexcept:\n    pass\n".format(line))

    imports_str = "\n".join(import_lines)
    return imports_str


def remove_imports(source):
    """(Taken from `hyperas.utils`). Remove all imports statements from `source` fragment

    Parameters
    ----------
    source: String
        A stringified fragment of source code

    Returns
    -------
    non_import_lines: String
        `source`, less any lines containing imports"""
    tree = parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    lines = source.split("\n")  # Source including all comments
    lines_to_remove = set(import_parser.line_numbers)
    non_import_lines = [
        _line for _i, _line in enumerate(lines, start=1) if _i not in lines_to_remove
    ]
    return "\n".join(non_import_lines)


def remove_comments(source):
    """(Taken from `hyperas.utils`). Remove all comments from `source` fragment

    Parameters
    ----------
    source: String
        A stringified fragment of source code

    Returns
    -------
    string: String
        `source`, less any comments"""
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)  # Remove '''...''' comments
    string = re.sub(re.compile("(?<!(['\"]).)#[^\n]*?\n"), "\n", string)  # Remove #...\n comments
    return string
