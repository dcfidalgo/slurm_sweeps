"""Script to generate the API docs in the README. ChatGPT is my friend."""
import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any, Literal

import slurm_sweeps as ss


def get_fully_qualified_class_name(cls):
    module = getattr(cls, "__module__", None)
    if module is None or module == str.__class__.__module__:
        return cls.__name__ if cls is not None else ""

    name = module + "." + cls.__name__
    if name == "inspect._empty":
        return ""

    return name


def extract_signature(
    obj: Any, name: str, class_or_def: Literal["class", "def"] = "class"
):
    if dataclasses.is_dataclass(obj):
        signature = inspect.signature(obj)
        signature_md = f"@dataclass\nclass {name}:\n    {'\n    '.join(str(par) for par in signature.parameters.values())}"

    elif isinstance(obj, property):
        return_type = get_fully_qualified_class_name(
            obj.fget.__annotations__.get("return")
        )
        if return_type:
            return_type = f" -> {return_type}"
        signature_md = f"@property\ndef {name}(){return_type}"

    elif inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
        return_type = get_fully_qualified_class_name(signature.return_annotation)
        if return_type:
            return_type = f" -> {return_type}"
        args = ",\n    ".join(str(par) for par in signature.parameters.values())
        signature_md = f"{class_or_def} {name}(\n    {args}\n){return_type}"

    else:
        signature_md = f"{name}: {obj}"

    return f"```python\n{signature_md}\n```\n"


def extract_desc(docstring: str) -> str:
    pattern = r"^\s*(.*?)\s*(?=\nArgs:|\nAttributes:|\nReturns:|\nRaises:|\Z)"

    matches = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

    # return re.sub(r'\s+', ' ', matches[0].strip())
    if matches:
        return matches[0]
    return ""


def extract_args(docstring: str) -> str:
    """Extract Args section of the doc string. ChatGPT was my friend."""
    # Regular expression to match argument names and descriptions
    pattern = r"^\s\s\s\s(\w+):\s(.*?)(?=\n\s\s\s\s\w+:|Returns:|Raises:|\Z)"

    # Find all matches
    matches = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

    # Put matches in dict and clean up description
    arguments = {name: re.sub(r"\s+", " ", desc.strip()) for name, desc in matches}

    # Markdown string for the Arguments
    if matches:
        args_md = "**Arguments**\n" + "\n".join(
            f"- `{n}`: {d}" for n, d in arguments.items()
        )
    else:
        args_md = ""

    return args_md


def extract_returns(docstring: str) -> str:
    """Only matches one line!"""
    pattern = r"Returns:\n\s{4}(.*)"
    matches = re.findall(pattern, docstring)
    if matches:
        returns_md = f"**Returns**\n- {matches[0].strip()}"
    else:
        returns_md = ""

    return returns_md


def extract_raises(docstring: str) -> str:
    """Each line is a bullet point!"""
    pattern = r"Raises:\n(.*)"
    matches = re.findall(pattern, docstring, re.DOTALL)
    if matches:
        items = [item.strip() for item in matches[0].split("\n    ")]
        raises_md = "**Raises**\n" + "\n".join(f"- {item}" for item in items)
    else:
        raises_md = ""

    return raises_md


def extract_desc_args_returns_raises(obj: Any):
    docstring = inspect.getdoc(obj)
    if docstring:
        md = ""
        desc = extract_desc(docstring)
        args = extract_args(docstring)
        returns = extract_returns(docstring)
        raises = extract_raises(docstring)
        if desc:
            md += f"{desc}\n\n"
        if args:
            md += f"{args}\n\n"
        if returns:
            md += f"{returns}\n\n"
        if raises:
            md += f"{raises}\n\n"
        return md

    return "\n"


def method2md(
    method, name, level: int, prefix: str = "slurm_sweeps", title: str = "FUNC "
):
    header = f"{'#'*level} {title}`{prefix + '.' if prefix else ''}{name}`\n"
    signature = extract_signature(method, name, "def")
    docs = extract_desc_args_returns_raises(method)

    return f"{header}{signature}{docs}"


def methods2md(cls, level: int):
    md = ""
    for name, method in inspect.getmembers(cls):  # , predicate=inspect.isfunction):
        # Skip private methods
        # print(name)
        if name.startswith("_"):
            continue
        if dataclasses.is_dataclass(cls) and name in cls.__dataclass_fields__:
            continue

        md += method2md(method, name, level=level, prefix=cls.__name__, title="")

    return md


def class2md(cls, level: int = 3, prefix: str = "slurm_sweeps") -> str:
    heading = f"{'#'*level} CLASS `{prefix}.{cls.__name__}`\n"
    init = extract_signature(cls, cls.__name__)
    docs = extract_desc_args_returns_raises(cls)
    methods = methods2md(cls, level + 1)

    return f"{heading}{init}{docs}{methods}"


if __name__ == "__main__":
    mds = [
        method2md(ss.log, "log", level=3),
        class2md(ss.Experiment),
        class2md(ss.SweepConfig),
        class2md(ss.AshaConfig),
        class2md(ss.TpeConfig),
        class2md(ss.SlurmConfig),
        class2md(ss.Result),
        class2md(ss.Trial),
    ]

    preview = Path("./docs/preview.md")
    with preview.open("w") as file:
        file.write("\n".join(mds))
