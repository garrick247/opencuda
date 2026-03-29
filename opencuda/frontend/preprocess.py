"""
Minimal C preprocessor for OpenCUDA.

Handles:
  #define NAME VALUE          — object-like text substitution
  #define NAME(params) body   — function-like macro expansion
  #define NAME                — define without value (for #ifdef)
  #undef NAME                 — remove a define
  #ifdef / #ifndef / #else / #elif / #endif — conditional compilation
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
    """Apply #define substitutions and conditional compilation to source code."""
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

    for line in source.split('\n'):
        stripped = line.strip()

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
            # Simplified: treat #if 0 as false, anything else as true
            rest = stripped[3:].strip()
            taking = (rest != '0') and _is_active()
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
                    taking = (rest != '0') and _is_active()
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

        # Apply object-like substitutions (whole-word only)
        for name, value in sorted(obj_defines.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(name) + r'\b', value, result)

        output_lines.append(result)

    return '\n'.join(output_lines)
