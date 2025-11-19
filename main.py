import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
from sympy import sympify, simplify, symbols, Matrix, expand
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


# Known function names to protect during implicit multiplication insertion
KNOWN_FUNCS = [
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
    'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
    'log', 'ln', 'sqrt', 'exp',
]

FUNC_PLACEHOLDER_PREFIX = "§FUNC§"


def protect_functions(expr: str) -> str:
    # Replace function names followed by '(' with placeholders to avoid variable+variable rule splitting them
    for fname in sorted(KNOWN_FUNCS, key=len, reverse=True):
        pattern = rf"\b{fname}\s*\("
        expr = re.sub(pattern, f"{FUNC_PLACEHOLDER_PREFIX}{fname}(", expr)
    return expr


def restore_functions(expr: str) -> str:
    # Restore placeholders back to function names
    return re.sub(rf"{FUNC_PLACEHOLDER_PREFIX}([a-zA-Z]+)\(", lambda m: f"{m.group(1)}(", expr)


def preprocess_expression(raw: str) -> str:
    """Preprocess expression to insert explicit multiplication using regex rules.

    Rules (applied with REGEX in this order):
      A. (\d)([a-zA-Z]) -> $1*$2
      B. ([a-zA-Z])([a-zA-Z]) -> $1*$2  (with protection so function names like sin( are not split)
      C. (\d)\( -> $1*(
      D. ([a-zA-Z])\( -> $1*(
      E. \)\( -> )*(
      F. (\d)(pi) -> $1*pi

    Plus: handle adjacency before protected function placeholders so cases like 5sin(x) become 5*sin(x).
    """
    if not raw:
        return raw

    # Normalize common unicode constants and spaces
    expr = raw.strip()
    expr = expr.replace("π", "pi")

    # Protect known function calls to avoid splitting their names
    expr = protect_functions(expr)

    # A: number + variable
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)

    # F: number + pi
    expr = re.sub(r"(\d)(pi)\b", r"\1*\2", expr)

    # C: number + (
    expr = re.sub(r"(\d)\(", r"\1*(", expr)

    # D: variable + (
    expr = re.sub(r"([a-zA-Z])\(", r"\1*(", expr)

    # E: )( -> )*(
    expr = re.sub(r"\)\(", r")*(", expr)

    # Insert * when a number, variable, or ')' is followed immediately by a protected function call
    expr = re.sub(r"(\d)\s*(§FUNC§[A-Za-z]+\()", r"\1*\2", expr)
    expr = re.sub(r"([a-zA-Z])\s*(§FUNC§[A-Za-z]+\()", r"\1*\2", expr)
    expr = re.sub(r"\)\s*(§FUNC§[A-Za-z]+\()", r")*\1", expr)

    # B: variable + variable (after function protection so we don't split function names)
    expr = re.sub(r"([a-zA-Z])([a-zA-Z])", r"\1*\2", expr)

    # Restore protected functions
    expr = restore_functions(expr)

    return expr


def collapse_coeff_var_terms(s: str) -> str:
    """Collapse occurrences of k*x -> kx when x is a single letter (degree 1),
    not part of a function name, and not immediately followed by another '*'.
    This preserves 5*sin(x) and 3*x**2 while turning 3*x into 3x.
    """
    pattern = re.compile(r"(?:(?<=^)|(?<=[+\-*/(\s]))(\d+)\*([A-Za-z])(?=\b(?!\s*\*) )?")
    # The above pattern is messy due to escaping in Python strings; build a clearer one below.
    pattern = re.compile(r"(?:(?<=^)|(?<=[+\-*/(\s]))(\d+)\*([A-Za-z])(?![A-Za-z0-9_\*])")
    return pattern.sub(r"\1\2", s)


def format_result(text: str) -> str:
    # Present powers with caret and collapse simple coefficients where appropriate
    s = text.replace("**", "^")
    s = collapse_coeff_var_terms(s)
    return s


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


@app.post("/api/solve", response_model=SolveResponse)
def solve_math(req: SolveRequest):
    """Universal math solve endpoint with mandatory preprocessing.

    - Preprocess input to handle implicit multiplication via REGEX.
    - Parse with SymPy (LaTeX supported if latex=True).
    - For simplify/auto, expand then simplify for intuitive polynomial forms.
    """
    expr_raw = req.input.strip()

    if req.latex:
        parsed_str = expr_raw
    else:
        parsed_str = preprocess_expression(expr_raw)

    # Try to parse either as LaTeX or plain text
    try:
        if req.latex:
            expr = parse_latex(parsed_str)
        else:
            expr = sympify(parsed_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    mode = (req.mode or "auto").lower()

    try:
        if mode == "simplify" or mode == "auto":
            simplified = simplify(expand(expr))
            if mode == "simplify":
                return SolveResponse(result=format_result(str(simplified)), steps=None, type="simplify")
            auto_result = simplified

        if mode == "derivative":
            var = symbols(req.variable) if req.variable else list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            res = diff(expr, var)
            return SolveResponse(result=format_result(str(simplify(res))), steps=None, type="derivative")

        if mode == "integral":
            var = symbols(req.variable) if req.variable else list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            res = integrate(expr, var)
            return SolveResponse(result=format_result(str(res)), steps=None, type="integral")

        if mode == "limit":
            var = symbols(req.variable) if req.variable else symbols('x')
            res = sym_limit(expr, var, 0)
            return SolveResponse(result=format_result(str(res)), steps=None, type="limit")

        if mode == "solve":
            from sympy import solve
            var = symbols(req.variable) if req.variable else None
            res = solve(expr, var) if var is not None else solve(expr)
            return SolveResponse(result=[format_result(str(r)) for r in res], steps=None, type="solve")

        if mode == "matrix":
            m = Matrix(expr)
            return SolveResponse(result=format_result(str(m)), steps=None, type="matrix")

        # Default: auto simplified result
        return SolveResponse(result=format_result(str(auto_result)), steps=None, type="auto")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Computation error: {str(e)}")


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
    f = lambdify(var, expr, modules=["numpy", {"pi": np.pi, "e": np.e}])

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
