"""
Minimal C preprocessor for OpenCUDA.

Handles:
  #define NAME VALUE          — object-like text substitution
  #define NAME(params) body   — function-like macro expansion
  #define NAME                — define without value (for #ifdef)
  #undef NAME                 — remove a define
  #ifdef / #ifndef / #else / #elif / #endif — conditional compilation
  #include "name"             — quoted-include, resolved against the include
                                paths supplied to preprocess() (see
                                CLI -I).  Angle-bracket includes
                                (#include <...>) and unresolvable
                                quoted includes are silently skipped
                                (system headers).
  // and /* */ comments — already handled by lexer
"""

from __future__ import annotations
import re
from pathlib import Path


def _expand_func_macro(body: str, params: list[str], args: list[str]) -> str:
    """Substitute macro parameters into body with whole-word matching."""
    result = body
    for param, arg in zip(params, args):
        # Only replace whole-word occurrences of the parameter
        result = re.sub(r'\b' + re.escape(param) + r'\b', arg, result)
    # Stringify operator ## (token pasting) — strip spaces around ##
    result = re.sub(r'\s*##\s*', '', result)
    return result


def _find_macro_call_args(text: str, start: int) -> tuple[list[str], int] | None:
    """Parse macro arguments starting from '(' at position start.

    Returns (arg_list, end_pos) where end_pos is after the closing ')'.
    Handles nested parentheses. Returns None if no '(' at start.
    """
    if start >= len(text) or text[start] != '(':
        return None
    depth = 0
    args = []
    cur_arg = []
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '(':
            depth += 1
            if depth > 1:
                cur_arg.append(ch)
        elif ch == ')':
            depth -= 1
            if depth == 0:
                args.append(''.join(cur_arg).strip())
                return args, i + 1
            else:
                cur_arg.append(ch)
        elif ch == ',' and depth == 1:
            args.append(''.join(cur_arg).strip())
            cur_arg = []
        else:
            cur_arg.append(ch)
        i += 1
    return None  # unmatched paren


def _eval_if_expr(expr: str, obj_defines: dict[str, str]) -> bool:
    """Evaluate a #if / #elif expression after expanding macros.

    Handles: integer literals, defined(NAME), !defined(NAME), simple
    comparisons (==, !=, <, <=, >, >=), && and ||, and arithmetic (+/-).
    Undefined macros are treated as 0 per the C standard.
    """
    # Expand object-like macros in the expression (repeat for nested macros)
    expanded = expr
    for _ in range(8):
        prev = expanded
        for name, value in sorted(obj_defines.items(), key=lambda x: -len(x[0])):
            expanded = re.sub(r'\b' + re.escape(name) + r'\b', value, expanded)
        if expanded == prev:
            break

    # Replace defined(NAME) → 1/0
    def _replace_defined(m):
        name = m.group(1)
        return '1' if (name in obj_defines) else '0'
    expanded = re.sub(r'defined\s*\(\s*(\w+)\s*\)', _replace_defined, expanded)
    expanded = re.sub(r'defined\s+(\w+)', _replace_defined, expanded)

    # Replace any remaining identifiers (undefined macros) with 0
    expanded = re.sub(r'\b[A-Za-z_]\w*\b', '0', expanded)

    # Strip suffixes like ULL, LL, U, L from integer literals
    expanded = re.sub(r'(\d+)[uUlL]+', r'\1', expanded)

    # Evaluate with Python's integer arithmetic (safe: no function calls left)
    try:
        result = eval(expanded, {"__builtins__": {}})  # noqa: S307
        return bool(result)
    except Exception:
        return True  # conservative: include on parse error


_INCLUDE_DEPTH_LIMIT = 32


def _resolve_includes(source: str, include_paths: list[Path],
                      visited: set[str], depth: int = 0) -> str:
    """Recursively inline #include "name" directives by reading the file
    from one of include_paths and splicing its contents in place.

    - Quoted includes (#include "name") are resolved against include_paths
      in order; the first hit wins.  If unresolvable, the line is dropped
      (matches the prior silent-skip behavior so missing system headers
      don't break compilation).
    - Angle-bracket includes (#include <name>) are always skipped.  Those
      are NVIDIA / libc++ headers OpenCUDA can't usefully inline.
    - Idempotency: each absolute file is included at most once per
      compilation, simulating #pragma once / include guards.  Prevents
      cycles and duplicate-definition errors when several headers pull
      in a common dependency.
    - Depth-limited at 32 to defend against pathological cycles that
      slip past the visited set.
    """
    if depth > _INCLUDE_DEPTH_LIMIT:
        return source
    out = []
    for line in source.split('\n'):
        stripped = line.lstrip()
        # Allow optional whitespace between '#' and the directive name
        m = re.match(r'#\s*include\s+"([^"]+)"', stripped)
        if m:
            name = m.group(1)
            resolved = None
            for d in include_paths:
                cand = (d / name).resolve()
                if cand.is_file():
                    resolved = cand
                    break
            if resolved is None:
                out.append('')  # silently drop unresolvable
                continue
            key = str(resolved)
            if key in visited:
                out.append('')  # already included — idempotent
                continue
            visited.add(key)
            inner = resolved.read_text(encoding='utf-8')
            out.append(_resolve_includes(inner, include_paths, visited,
                                          depth + 1))
            continue
        # Angle-bracket include — skip (system headers)
        if re.match(r'#\s*include\s+<', stripped):
            out.append('')
            continue
        out.append(line)
    return '\n'.join(out)


def preprocess(source: str, include_paths: list[Path] | None = None) -> str:
    """Apply #include resolution, #define substitutions, and conditional
    compilation.  Quoted includes are resolved against include_paths
    (defaults to empty); angle-bracket includes are skipped.  See
    _resolve_includes for the include policy."""
    if include_paths:
        source = _resolve_includes(source, list(include_paths), visited=set())
    # object-like: name → replacement string
    obj_defines: dict[str, str] = {}
    # function-like: name → (params list, body string)
    func_defines: dict[str, tuple[list[str], str]] = {}
    output_lines = []

    # Conditional compilation state.
    # Each entry is (currently_including, seen_true_branch).
    # - currently_including: True iff lines in this block should be emitted.
    # - seen_true_branch: True once any branch of this #if/#ifdef has been taken.
    cond_stack: list[tuple[bool, bool]] = []

    def _is_active() -> bool:
        """True iff all enclosing conditional blocks are currently included."""
        return all(inc for inc, _ in cond_stack)

    def _is_defined(name: str) -> bool:
        return name in obj_defines or name in func_defines

    # Join backslash-continued lines (line-splicing) before tokenizing.
    # A line ending with \ (after stripping trailing whitespace) is joined
    # to the next line, with the \ and newline removed.
    raw_lines = source.split('\n')
    joined: list[str] = []
    pending = ''
    for raw in raw_lines:
        if raw.rstrip().endswith('\\'):
            pending += raw.rstrip()[:-1]  # strip trailing backslash
        else:
            joined.append(pending + raw)
            pending = ''
    if pending:
        joined.append(pending)

    for line in joined:
        stripped = line.strip()
        # C allows optional whitespace between '#' and the directive name:
        # '# define', '#  ifdef', etc. Normalize to remove that whitespace.
        stripped = re.sub(r'^#\s+', '#', stripped)

        # Conditional directives are processed regardless of current active state
        # (so nested #ifdef/#endif are properly matched).
        if stripped.startswith('#ifdef') or stripped.startswith('#ifndef'):
            m = re.match(r'#ifn?def\s+(\w+)', stripped)
            name = m.group(1) if m else ''
            is_defined = _is_defined(name)
            taking = is_defined if stripped.startswith('#ifdef') else not is_defined
            # Only actually include if enclosing context is active
            taking = taking and _is_active()
            cond_stack.append((taking, taking))
            output_lines.append('')
            continue

        if stripped.startswith('#if ') or stripped == '#if':
            rest = stripped[3:].strip()
            taking = _eval_if_expr(rest, obj_defines) and _is_active()
            cond_stack.append((taking, taking))
            output_lines.append('')
            continue

        if stripped.startswith('#elif'):
            if cond_stack:
                _, seen = cond_stack[-1]
                if seen:
                    # A previous branch was taken — skip all remaining elif/else
                    cond_stack[-1] = (False, True)
                else:
                    rest = stripped[5:].strip()
                    taking = _eval_if_expr(rest, obj_defines) and _is_active()
                    cond_stack[-1] = (taking, taking)
            output_lines.append('')
            continue

        if stripped == '#else':
            if cond_stack:
                _, seen = cond_stack[-1]
                # Take else only if no prior branch was taken and enclosing is active
                taking = (not seen) and all(inc for inc, _ in cond_stack[:-1])
                cond_stack[-1] = (taking, True)
            output_lines.append('')
            continue

        if stripped == '#endif':
            if cond_stack:
                cond_stack.pop()
            output_lines.append('')
            continue

        # All other lines: only process if currently active
        if not _is_active():
            output_lines.append('')
            continue

        # Function-like macro: #define NAME(params) body
        m = re.match(r'#define\s+(\w+)\(([^)]*)\)\s*(.*)', stripped)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(',') if p.strip()]
            body = m.group(3).strip()
            # Strip trailing // comment from body so it doesn't infect expansions.
            body = re.sub(r'\s*//.*$', '', body)
            func_defines[name] = (params, body)
            output_lines.append('')  # preserve line numbers
            continue

        # Object-like macro: #define NAME VALUE
        m = re.match(r'#define\s+(\w+)\s+(.*)', stripped)
        if m:
            name, value = m.group(1), m.group(2).strip()
            # Strip trailing // comment from value.
            value = re.sub(r'\s*//.*$', '', value).strip()
            obj_defines[name] = value
            output_lines.append('')
            continue

        # #define NAME (no value)
        m = re.match(r'#define\s+(\w+)\s*$', stripped)
        if m:
            obj_defines[m.group(1)] = '1'
            output_lines.append('')
            continue

        # #undef NAME
        m = re.match(r'#undef\s+(\w+)', stripped)
        if m:
            name = m.group(1)
            obj_defines.pop(name, None)
            func_defines.pop(name, None)
            output_lines.append('')
            continue

        # #include — skip (not supported)
        if stripped.startswith('#include'):
            output_lines.append('')
            continue

        # #pragma — skip
        if stripped.startswith('#pragma'):
            output_lines.append('')
            continue

        # Skip other preprocessor directives
        if stripped.startswith('#'):
            output_lines.append('')
            continue

        # Apply function-like macro expansions first (before object-like)
        result = line
        if func_defines:
            # Repeatedly expand until no more changes (handle nested macros)
            for _ in range(8):  # max expansion depth
                changed = False
                for name, (params, body) in sorted(func_defines.items(),
                                                    key=lambda x: -len(x[0])):
                    pattern = r'\b' + re.escape(name) + r'\s*\('
                    while True:
                        match = re.search(pattern, result)
                        if not match:
                            break
                        call_start = match.end() - 1  # position of '('
                        parsed = _find_macro_call_args(result, call_start)
                        if parsed is None:
                            break
                        args, call_end = parsed
                        # Expand: substitute params into body
                        expanded = _expand_func_macro(body, params, args)
                        result = result[:match.start()] + expanded + result[call_end:]
                        changed = True
                if not changed:
                    break

        # Apply object-like substitutions (whole-word only).
        # Repeat until stable to handle macros whose values reference other macros.
        for _ in range(8):
            prev = result
            for name, value in sorted(obj_defines.items(), key=lambda x: -len(x[0])):
                result = re.sub(r'\b' + re.escape(name) + r'\b', value, result)
            if result == prev:
                break

        output_lines.append(result)

    return '\n'.join(output_lines)
