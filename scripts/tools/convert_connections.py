import re
import os
from pathlib import Path

PROJECT = Path(r"C:\users\joshd\canompx3")
DIRS = [PROJECT / "pipeline", PROJECT / "trading_app"]
SKIP_FILES = {"ingest_dbn.py", "ingest_dbn_mgc.py", "ingest_dbn_daily.py"}
modified_files = []
total_conversions = 0

def get_indent(line):
    return line[: len(line) - len(line.lstrip())]

def convert_try_finally(lines):
    conversions = 0
    result = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(con|conn)\s*=\s*(duckdb\.connect\(.*?\)|_connect\(.*?\))\s*$", lines[i])
        if not m:
            result.append(lines[i])
            i += 1
            continue
        base_indent = m.group(1)
        var_name = m.group(2)
        connect_expr = m.group(3)
        j = i + 1
        while j < len(lines) and lines[j].strip() == "": j += 1
        if not (j < len(lines) and lines[j].strip() == "try:" and get_indent(lines[j]) == base_indent):
            result.append(lines[i])
            i += 1
            continue
        try_line = j
        depth = 1
        k = try_line + 1
        finally_line = None
        while k < len(lines):
            stripped = lines[k].strip()
            line_indent = get_indent(lines[k])
            if stripped == "" or line_indent != base_indent:
                k += 1
                continue
            if stripped == "try:":
                depth += 1
            elif stripped.startswith("finally:"):
                if depth == 1:
                    finally_line = k
                    break
                else:
                    depth -= 1
            k += 1
        if finally_line is None:
            result.append(lines[i])
            i += 1
            continue
        close_pattern = var_name + ".close()"
        fb_indent = base_indent + "    "
        fb_end = finally_line + 1
        while fb_end < len(lines):
            line = lines[fb_end]
            if line.strip() == "":
                fb_end += 1
                continue
            if get_indent(line).startswith(fb_indent):
                fb_end += 1
            else:
                break
        has_close = any(close_pattern in lines[fl] for fl in range(finally_line + 1, fb_end))
        if not has_close:
            result.append(lines[i])
            i += 1
            continue
        try_body = []
        except_blocks = []
        in_except = False
        for idx in range(try_line + 1, finally_line):
            line = lines[idx]
            stripped = line.strip()
            line_indent = get_indent(line)
            if line_indent == base_indent and (stripped.startswith("except") or stripped == "else:"):
                in_except = True
            if in_except:
                except_blocks.append(line)
            else:
                try_body.append(line)
        extra_finally = []
        for fl in range(finally_line + 1, fb_end):
            if close_pattern not in lines[fl] and lines[fl].strip() != "": extra_finally.append(lines[fl])
        result.append(base_indent + "with " + connect_expr + " as " + var_name + ":\n")
        if except_blocks:
            result.append(base_indent + "    try:\n")
            for line in try_body:
                if line.strip() == "": result.append(line)
                else:
                    old = get_indent(line)
                    extra = old[len(base_indent):] if old.startswith(base_indent) else ""
                    result.append(base_indent + "    " + extra + line.lstrip())
            for line in except_blocks:
                if line.strip() == "": result.append(line)
                else:
                    old = get_indent(line)
                    extra = old[len(base_indent):] if old.startswith(base_indent) else ""
                    result.append(base_indent + "    " + extra + line.lstrip())
        else:
            for line in try_body: result.append(line)
        for line in extra_finally: result.append(line)
        conversions += 1
        i = fb_end
        continue
    return result, conversions

def convert_bare_close(lines):
    conversions = 0
    result = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(con|conn)\s*=\s*(duckdb\.connect\(.*?\)|_connect\(.*?\))\s*$", lines[i])
        if not m:
            result.append(lines[i])
            i += 1
            continue
        base_indent = m.group(1)
        var_name = m.group(2)
        connect_expr = m.group(3)
        j = i + 1
        while j < len(lines) and lines[j].strip() == "": j += 1
        if j < len(lines) and lines[j].strip() == "try:" and get_indent(lines[j]) == base_indent:
            result.append(lines[i])
            i += 1
            continue
        if j < len(lines) and lines[j].strip().startswith("with "):
            result.append(lines[i])
            i += 1
            continue
        close_pattern = var_name + ".close()"
        close_line = None
        for k in range(i + 1, len(lines)):
            stripped = lines[k].strip()
            if stripped == close_pattern:
                close_line = k
                break
        if close_line is None:
            result.append(lines[i])
            i += 1
            continue
        body_lines = lines[i + 1 : close_line]
        result.append(base_indent + "with " + connect_expr + " as " + var_name + ":\n")
        for line in body_lines:
            if line.strip() == "": result.append(line)
            else:
                old_indent = get_indent(line)
                if old_indent.startswith(base_indent):
                    extra = old_indent[len(base_indent):]
                    result.append(base_indent + "    " + extra + line.lstrip())
                else:
                    result.append(base_indent + "    " + line.lstrip())
        i = close_line + 1
        conversions += 1
        continue
    return result, conversions

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f: content = f.read()
    if "duckdb.connect(" not in content and "= _connect(" not in content: return 0
    if ".close()" not in content: return 0
    lines = content.splitlines(keepends=True)
    lines, c1 = convert_try_finally(lines)
    lines, c2 = convert_bare_close(lines)
    total = c1 + c2
    if total > 0:
        with open(filepath, "w", encoding="utf-8") as f: f.writelines(lines)
    return total

def main():
    global total_conversions
    for d in DIRS:
        for root, dirs, files in os.walk(d):
            dirs[:] = [dd for dd in dirs if dd != "__pycache__"]
            for fname in sorted(files):
                if not fname.endswith(".py"): continue
                if fname in SKIP_FILES: continue
                fpath = Path(root) / fname
                n = process_file(fpath)
                if n > 0:
                    modified_files.append((str(fpath.relative_to(PROJECT)), n))
                    total_conversions += n
    print(f"Total conversions: {total_conversions}")
    print(f"Files modified: {len(modified_files)}")
    for f, n in modified_files: print(f"  {f}: {n} conversion(s)")

if __name__ == "__main__": main()
