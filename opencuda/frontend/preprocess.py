"""
Minimal C preprocessor for OpenCUDA.

Handles:
  #define NAME VALUE          — object-like text substitution
  #define NAME(params) body   — function-like macro expansion
  #define NAME                — define without value (for #ifdef)
  // and /* */ comments — already handled by lexer
"""

from __future__ import annotations
import re


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


def preprocess(source: str) -> str:
    """Apply #define substitutions to source code."""
    # object-like: name → replacement string
    obj_defines: dict[str, str] = {}
    # function-like: name → (params list, body string)
    func_defines: dict[str, tuple[list[str], str]] = {}
    output_lines = []

    for line in source.split('\n'):
        stripped = line.strip()

        # Function-like macro: #define NAME(params) body
        m = re.match(r'#define\s+(\w+)\(([^)]*)\)\s*(.*)', stripped)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(',') if p.strip()]
            body = m.group(3).strip()
            func_defines[name] = (params, body)
            output_lines.append('')  # preserve line numbers
            continue

        # Object-like macro: #define NAME VALUE
        m = re.match(r'#define\s+(\w+)\s+(.*)', stripped)
        if m:
            name, value = m.group(1), m.group(2).strip()
            obj_defines[name] = value
            output_lines.append('')
            continue

        # #define NAME (no value)
        m = re.match(r'#define\s+(\w+)\s*$', stripped)
        if m:
            obj_defines[m.group(1)] = '1'
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

        # Apply object-like substitutions (whole-word only)
        for name, value in sorted(obj_defines.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(name) + r'\b', value, result)

        output_lines.append(result)

    return '\n'.join(output_lines)
