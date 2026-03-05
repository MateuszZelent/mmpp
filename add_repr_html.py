#!/usr/bin/env python3
"""Auto-add _repr_html_ to all PlotAccessor classes that don't have one.

This script:
1. Finds all PlotAccessor classes across the codebase
2. Introspects their public methods 
3. Generates and inserts _repr_html_ method using the shared utility
"""
import ast
import re
import sys
from pathlib import Path

BASE = Path("/home/kkingstoun/git/containers_admin2/compute-lib/mmpp/mmpp")

def find_plot_accessors():
    """Find all PlotAccessor classes and check if they have _repr_html_."""
    results = []
    for py_file in BASE.rglob("*.py"):
        try:
            source = py_file.read_text()
        except Exception:
            continue
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and "PlotAccessor" in node.name:
                has_repr = any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "_repr_html_"
                    for item in node.body
                )
                # Collect public methods
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = item.name
                        if name.startswith("_") and name != "__call__":
                            continue
                        # Get args
                        args = []
                        for arg in item.args.args:
                            if arg.arg != "self":
                                args.append(arg.arg)
                        # Get docstring
                        ds = ast.get_docstring(item) or ""
                        first_line = ds.split("\n")[0].strip() if ds else ""
                        methods.append((name, args, first_line))
                
                results.append({
                    "file": str(py_file.relative_to(BASE)),
                    "abs_path": str(py_file),
                    "class_name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "has_repr_html": has_repr,
                    "methods": methods,
                })
    return results

def generate_repr_html(class_name, methods):
    """Generate _repr_html_ method code."""
    # Determine relative import depth based on class location
    entries = []
    for name, args, desc in methods:
        if name in ("__init__", "__call__", "__repr__", "__len__", "__getitem__", "__iter__"):
            continue
        sig = f".{name}({', '.join(args[:3])}{'...' if len(args) > 3 else ''})"
        desc_text = desc or name.replace("_", " ").title()
        entries.append((sig, desc_text, f"Parameters: {', '.join(args) or 'none'}"))
    
    if not entries:
        return None
    
    lines = []
    lines.append("")
    lines.append("    def _repr_html_(self) -> str:")
    lines.append("        from mmpp._repr_helpers import plot_accessor_html")
    lines.append(f"        return plot_accessor_html(\"{class_name}\", [")
    for sig, desc, tip in entries:
        lines.append(f"            (\"{sig}\",")
        lines.append(f"             \"{desc}\",")
        lines.append(f"             \"{tip}\"),")
    lines.append("        ])")
    return "\n".join(lines)

def insert_repr_html(file_path, class_name, end_line, code):
    """Insert _repr_html_ before the last line of the class."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Find the last method (__repr__ or end of class)
    # Insert before the end of the class
    insert_at = end_line  # end_line is 1-indexed
    
    # Find __repr__ method - insert after it
    for i in range(end_line - 1, max(end_line - 20, 0), -1):
        if i < len(lines) and "def __repr__" in lines[i]:
            # Find the end of __repr__
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip() and not lines[j].startswith(" " * 8) and not lines[j].startswith("\t\t"):
                    insert_at = j
                    break
            else:
                insert_at = min(i + 5, end_line)
            break
    
    code_lines = [line + "\n" for line in code.split("\n")]
    lines[insert_at:insert_at] = code_lines
    
    with open(file_path, "w") as f:
        f.writelines(lines)
    
    return insert_at

def main():
    accessors = find_plot_accessors()
    print(f"Found {len(accessors)} PlotAccessor classes\n")
    
    needs_fix = [a for a in accessors if not a["has_repr_html"]]
    already_has = [a for a in accessors if a["has_repr_html"]]
    
    print(f"Already have _repr_html_: {len(already_has)}")
    for a in already_has:
        print(f"  ✅ {a['class_name']} ({a['file']})")
    
    print(f"\nNeed _repr_html_: {len(needs_fix)}")
    for a in needs_fix:
        print(f"  ❌ {a['class_name']} ({a['file']})")
        meths = [(n, a, d) for n, a, d in a["methods"] if not n.startswith("_")]
        for name, args, desc in meths:
            print(f"       .{name}({', '.join(args[:2])}) - {desc[:60]}")
    
    if "--apply" in sys.argv:
        print("\n--- APPLYING ---\n")
        for a in needs_fix:
            code = generate_repr_html(a["class_name"], a["methods"])
            if code is None:
                print(f"  SKIP {a['class_name']} (no public methods)")
                continue
            try:
                line = insert_repr_html(a["abs_path"], a["class_name"], a["line_end"], code)
                print(f"  ✅ Added _repr_html_ to {a['class_name']} at line {line} in {a['file']}")
            except Exception as e:
                print(f"  ❌ Failed {a['class_name']}: {e}")

if __name__ == "__main__":
    main()
