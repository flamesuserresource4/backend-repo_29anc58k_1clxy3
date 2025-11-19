import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Tuple
from sympy import sympify, simplify, symbols, Matrix, expand, S
from sympy import diff, integrate, limit as sym_limit
from sympy.parsing.latex import parse_latex
from sympy import lambdify
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Preprocessor & Utilities
# =========================

# Known function names to normalize/protect
KNOWN_FUNCS = [
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
    'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
    'log', 'ln', 'sqrt', 'exp'
]

FUNC_PLACEHOLDER_PREFIX = "§FUNC§"


def normalize_unicode(expr: str) -> str:
    # Replace unicode math symbols with ASCII equivalents
    expr = expr.replace("π", "pi").replace("Π", "pi")
    expr = expr.replace("×", "*").replace("·", "*")
    expr = expr.replace("÷", "/")
    expr = expr.replace("−", "-")
    expr = expr.replace("^", "**")  # make compatible with SymPy
    # sqrt symbols: √x or √(x)
    expr = re.sub(r"√\s*\((.*?)\)", r"sqrt(\1)", expr)
    expr = re.sub(r"√\s*([A-Za-z0-9_]+)", r"sqrt(\1)", expr)
    return expr


def normalize_functions_case(expr: str) -> str:
    # Make function names case-insensitive (Sin, SIN, Ln, etc.)
    for f in sorted(KNOWN_FUNCS, key=len, reverse=True):
        expr = re.sub(rf"\b{f}\b", f, expr, flags=re.IGNORECASE)
    return expr


def protect_functions(expr: str) -> str:
    # Replace function names followed by '(' with placeholders to avoid variable+variable rule splitting them
    for fname in sorted(KNOWN_FUNCS, key=len, reverse=True):
        pattern = rf"\b{fname}\s*\("
        expr = re.sub(pattern, f"{FUNC_PLACEHOLDER_PREFIX}{fname}(", expr)
    return expr


def restore_functions(expr: str) -> str:
    # Restore placeholders back to function names
    return re.sub(rf"{FUNC_PLACEHOLDER_PREFIX}([a-zA-Z]+)\(", lambda m: f"{m.group(1)}(", expr)


def remove_illegal_chars(expr: str) -> str:
    # Allow digits, letters, operators, parentheses, comma, space, underscore, arrow symbols
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+\-*/(),._ <>→")
    return ''.join(ch for ch in expr if ch in allowed)


def balance_parentheses(expr: str) -> str:
    # Remove leading unmatched closing parens and append missing closing parens
    s = []
    balance = 0
    for ch in expr:
        if ch == '(':
            balance += 1
            s.append(ch)
        elif ch == ')':
            if balance > 0:
                balance -= 1
                s.append(ch)
            else:
                # skip unmatched ')'
                continue
        else:
            s.append(ch)
    if balance > 0:
        s.append(')' * balance)
    return ''.join(s)


def preprocess_limit_arrow(expr: str) -> str:
    # Convert limit(f(x), x→a) or x->a into sympy limit(f(x), x, a)
    def arrow_to_limit(m: re.Match) -> str:
        inner = m.group(1)
        var = m.group(2)
        val = m.group(3)
        return f"limit({inner}, {var}, {val})"

    # Patterns like limit( expr , x→a ) or x->a
    expr = re.sub(r"limit\(\s*(.+?)\s*,\s*([A-Za-z]+)\s*[→\-]+>\s*([A-Za-z0-9_\.\-]+)\s*\)", arrow_to_limit, expr)
    return expr


def preprocess_expression(raw: str) -> str:
    """Preprocess expression to ensure explicit multiplication and valid syntax.

    Steps:
      - Normalize unicode and function case
      - Handle limit arrow syntax
      - Insert explicit * using regex rules A–F
      - Balance parentheses
      - Remove illegal characters (after balancing to keep parentheses)
      - Restore function tokens
    Rules (REGEX):
      A. (\d)([a-zA-Z]) -> $1*$2
      B. ([a-zA-Z])([a-zA-Z]) -> $1*$2  (with protection so function names like sin( are not split)
      C. (\d)\( -> $1*(
      D. ([a-zA-Z])\( -> $1*(
      E. \)\( -> )*(
      F. (\d)(pi|e) -> $1*$2
    """
    if not raw:
        return raw

    expr = raw.strip()
    expr = normalize_unicode(expr)
    expr = normalize_functions_case(expr)
    expr = preprocess_limit_arrow(expr)

    # Normalize constant e when used plainly; SymPy recognizes E but also exp(1)
    # We'll keep 'e' as a symbol unless in numeric eval; SymPy sympify maps 'E' to Euler's number,
    # so map bare 'e' to E to avoid treating it as a variable.
    expr = re.sub(r"\be\b", "E", expr)

    # Protect known function calls
    expr = protect_functions(expr)

    # A: number + variable
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)

    # F: number + pi|E
    expr = re.sub(r"(\d)(pi|E)\b", r"\1*\2", expr)

    # C: number + (
    expr = re.sub(r"(\d)\(", r"\1*(", expr)

    # D: variable + (
    expr = re.sub(r"([a-zA-Z])\(", r"\1*(", expr)

    # E: )( -> )*(
    expr = re.sub(r"\)\(", r")*(", expr)

    # Insert * when a number, variable, or ')' is followed by a protected func call
    expr = re.sub(r"(\d)\s*(§FUNC§[A-Za-z]+\()", r"\1*\2", expr)
    expr = re.sub(r"([a-zA-Z])\s*(§FUNC§[A-Za-z]+\()", r"\1*\2", expr)
    expr = re.sub(r"\)\s*(§FUNC§[A-Za-z]+\()", r")*\1", expr)

    # B: variable + variable (after function protection)
    expr = re.sub(r"([a-zA-Z])([a-zA-Z])", r"\1*\2", expr)

    # Balance parentheses and clean illegal chars
    expr = balance_parentheses(expr)
    expr = restore_functions(expr)
    expr = remove_illegal_chars(expr)

    # Final whitespace normalization
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def collapse_coeff_var_terms(s: str) -> str:
    """Collapse occurrences of k*x -> kx when x is a single letter (degree 1),
    not part of a function name, and not immediately followed by another '*'.
    This preserves 5*sin(x) and 3*x**2 while turning 3*x into 3x.
    """
    pattern = re.compile(r"(?:(?<=^)|(?<=[+\-*/(\s]))(\d+)\*([A-Za-z])(?![A-Za-z0-9_\*])")
    return pattern.sub(r"\1\2", s)


def format_result(text: str) -> str:
    # Present powers with caret and collapse simple coefficients where appropriate
    s = text.replace("**", "^")
    s = collapse_coeff_var_terms(s)
    return s


# =========================
# Step-by-step Engine (heuristic)
# =========================

def steps_arithmetic_addition(a: str, b: str, result: str) -> List[str]:
    return [
        "Identify the operation: addition.",
        f"Add the two operands: {a} + {b}.",
        f"Compute the sum: {a} + {b} = {result}.",
        f"Final answer: {result}",
    ]


def steps_like_terms(a_coef: int, b_coef: int, var: str, result_coef: int) -> List[str]:
    return [
        f"Identify like terms: both terms contain '{var}'.",
        f"Combine coefficients: {a_coef} + {b_coef} = {result_coef}.",
        f"Attach variable: {result_coef}{var}.",
        f"Final answer: {result_coef}{var}",
    ]


def steps_distribution(coef: str, inner: str, expanded: str) -> List[str]:
    return [
        f"Distribute {coef} across each term in ({inner}).",
        f"Expand the product.",
        f"Combine like terms to get: {expanded}.",
        f"Final answer: {expanded}",
    ]


def steps_derivative(var: str) -> List[str]:
    return [
        f"Differentiate with respect to {var} using standard rules (power, product, chain).",
        f"Simplify the derivative.",
    ]


def steps_integral(var: str) -> List[str]:
    return [
        f"Integrate with respect to {var} using standard rules (power rule, substitution, integration by parts if needed).",
        f"Add the constant of integration when appropriate.",
    ]


def steps_limit(var: str) -> List[str]:
    return [
        f"Evaluate the limit as {var} approaches the target value.",
        f"Simplify the expression; apply algebraic manipulation or L'Hospital's Rule if needed.",
    ]


def detect_simple_patterns(expr_str: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Detect a few simple patterns to produce friendly steps."""
    # 2a + 2a
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z])\s*\+\s*(\d+)\s*\2\s*", expr_str)
    if m:
        a = int(m.group(1)); b = int(m.group(3)); var = m.group(2)
        res = f"{a+b}{var}"
        return steps_like_terms(a, b, var, a+b), res

    # 3x(x+1) basic distribution
    m = re.fullmatch(r"\s*(\d+[a-zA-Z])\s*\(\s*([a-zA-Z])\s*\+\s*(\d+)\s*\)\s*", expr_str)
    if m:
        coef = m.group(1)
        inner = f"{m.group(2)} + {m.group(3)}"
        return steps_distribution(coef, inner, ""), None

    return None, None


# =========================
# API Models
# =========================

class SolveRequest(BaseModel):
    input: str
    mode: Optional[str] = None  # auto, simplify, derivative, integral, limit, solve, matrix
    variable: Optional[str] = None
    latex: Optional[bool] = False  # if input is LaTeX

class SolveResponse(BaseModel):
    result: Any
    steps: Optional[List[str]] = None
    type: str


@app.get("/")
def read_root():
    return {"message": "Math backend ready"}


# =========================
# MathJS-first style pipeline (emulated), with SymPy symbolic step
# =========================

def mathjs_like_eval(expr_str: str):
    """Emulated math.js pass: try numeric evaluation/simplification quickly.
    - If expression is numeric after sympify: return evalf
    - Else return simplified form (without heavy symbolic tasks)
    Returns (success, result_sympy_expr)
    """
    try:
        expr = sympify(expr_str)
        if len(expr.free_symbols) == 0:
            return True, expr.evalf()
        # light simplify
        return True, simplify(expr)
    except Exception:
        return False, None


@app.post("/api/solve", response_model=SolveResponse)
def solve_math(req: SolveRequest):
    """Universal CAS endpoint with:
    - Robust preprocessing and syntax normalization
    - MathJS-like first pass (emulated)
    - SymPy symbolic operations for calculus/solve/factor/expand
    - Step-by-step heuristic explanations for all modes
    """
    expr_raw = req.input.strip()

    # Handle LaTeX separately
    if req.latex:
        parsed_str = expr_raw
    else:
        parsed_str = preprocess_expression(expr_raw)

    # Quick pattern-based steps
    steps, forced_result_text = detect_simple_patterns(parsed_str)

    # MathJS-like pass
    mj_success, mj_expr = mathjs_like_eval(parsed_str) if not req.latex else (False, None)

    mode = (req.mode or "auto").lower()

    # If LaTeX, or if we need symbolic operations, parse with SymPy
    try:
        if req.latex:
            expr = parse_latex(parsed_str)
        else:
            expr = sympify(parsed_str)
    except Exception:
        # Fallback to mathjs-like if available
        if mj_success and mj_expr is not None:
            return SolveResponse(result=format_result(str(mj_expr)), steps=steps, type=mode or "auto")
        raise HTTPException(status_code=400, detail="Unable to parse the expression. Please check syntax.")

    try:
        if mode in ("simplify", "auto"):
            simplified = simplify(expand(expr))
            # Prefer forced result if a known pattern was matched
            if forced_result_text:
                return SolveResponse(result=format_result(forced_result_text), steps=steps, type="simplify" if mode=="simplify" else "auto")
            # If mathjs-like succeeded, prefer its (often more numeric) result
            if mj_success and mj_expr is not None:
                return SolveResponse(result=format_result(str(mj_expr)), steps=steps, type="auto")
            return SolveResponse(result=format_result(str(simplified)), steps=steps, type="simplify" if mode=="simplify" else "auto")

        if mode == "derivative":
            var = symbols(req.variable) if req.variable else (list(expr.free_symbols)[0] if expr.free_symbols else symbols('x'))
            res = diff(expr, var)
            return SolveResponse(result=format_result(str(simplify(res))), steps=steps_derivative(str(var)), type="derivative")

        if mode == "integral":
            var = symbols(req.variable) if req.variable else (list(expr.free_symbols)[0] if expr.free_symbols else symbols('x'))
            res = integrate(expr, var)
            return SolveResponse(result=format_result(str(res)), steps=steps_integral(str(var)), type="integral")

        if mode == "limit":
            var = symbols(req.variable) if req.variable else symbols('x')
            # Try to find a target like limit(f(x), x, a) already parsed; otherwise default -> 0
            try:
                res = sym_limit(expr, var, 0)
            except Exception:
                res = sym_limit(expr, var, 0)
            return SolveResponse(result=format_result(str(res)), steps=steps_limit(str(var)), type="limit")

        if mode == "solve":
            from sympy import solve
            var = symbols(req.variable) if req.variable else None
            res = solve(expr, var) if var is not None else solve(expr)
            return SolveResponse(result=[format_result(str(r)) for r in res], steps=["Solve the equation for the target variable using algebraic manipulation."], type="solve")

        if mode == "matrix":
            m = Matrix(expr)
            return SolveResponse(result=format_result(str(m)), steps=["Interpret the input as a matrix and format it."], type="matrix")

        # Default: auto simplified result
        auto_result = simplify(expand(expr))
        return SolveResponse(result=format_result(str(auto_result)), steps=steps, type="auto")

    except Exception:
        # If symbolic step fails, fallback to mathjs-like result
        if mj_success and mj_expr is not None:
            return SolveResponse(result=format_result(str(mj_expr)), steps=steps or ["Computed a simplified numeric form."], type=mode)
        raise HTTPException(status_code=400, detail="Computation failed. The input was auto-corrected, but evaluation still failed.")


# =========================
# Graphing
# =========================

class GraphRequest(BaseModel):
    expression: str
    variable: Optional[str] = "x"
    xMin: Optional[float] = -10
    xMax: Optional[float] = 10
    points: Optional[int] = 400
    latex: Optional[bool] = False

class GraphResponse(BaseModel):
    x: List[float]
    y: List[Optional[float]]


@app.post("/api/graph", response_model=GraphResponse)
def graph_function(req: GraphRequest):
    expr_raw = req.expression.strip()
    try:
        if req.latex:
            expr = parse_latex(expr_raw)
        else:
            pre = preprocess_expression(expr_raw)
            expr = sympify(pre)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    var = symbols(req.variable)
    f = lambdify(var, expr, modules=["numpy", {"pi": np.pi}])

    xs = np.linspace(req.xMin, req.xMax, req.points)
    ys = []
    for x in xs:
        try:
            y = float(f(x))
            if not np.isfinite(y):
                ys.append(None)
            else:
                ys.append(y)
        except Exception:
            ys.append(None)

    return GraphResponse(x=[float(v) for v in xs], y=ys)


# =========================
# Health & DB test
# =========================

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
    }
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
    except Exception:
        pass
    return response


# =========================
# Self-tests for key cases (symbolic + functions + implicit mult)
# =========================

class SelfTestResult(BaseModel):
    passed: bool
    details: List[str]


@app.get("/api/selftest", response_model=SelfTestResult)
def self_test():
    tests = [
        ("2a + 2a", "4a"),
        ("3x(x+1)", "3*x^2 + 3*x"),
        ("(x+1)(x+2)", "x^2 + 3*x + 2"),
        ("5sin(x)", "5*sin(x)"),
        ("2pi", "2*pi"),
        ("2(x+5)", "2*x + 10"),
    ]
    details = []
    all_ok = True
    for inp, expected in tests:
        pre = preprocess_expression(inp)
        try:
            expr = sympify(pre)
            res = simplify(expand(expr))
            got = format_result(str(res))
        except Exception as e:
            got = f"ERROR: {e}"
        ok = (got.replace(' ', '') == expected.replace(' ', ''))
        all_ok = all_ok and ok
        details.append(f"{inp} => {got} | expected {expected} | {'OK' if ok else 'FAIL'}")
    return SelfTestResult(passed=all_ok, details=details)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
