import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
from sympy import sympify, simplify, symbols, Eq, Matrix
from sympy import diff, integrate, limit as sym_limit
from sympy.parsing.latex import parse_latex
from sympy.plotting import plot
from sympy.core.sympify import SympifyError
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
    """Universal math solve endpoint using SymPy. Auto-detects expression type and returns simplified results.
    Supports LaTeX input if latex=True.
    """
    expr_str = req.input.strip()

    # Try to parse either as LaTeX or plain text
    try:
        if req.latex:
            expr = parse_latex(expr_str)
        else:
            expr = sympify(expr_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    mode = (req.mode or "auto").lower()

    try:
        if mode == "simplify" or mode == "auto":
            simplified = simplify(expr)
            # If explicit mode requested, return immediately
            if mode == "simplify":
                return SolveResponse(result=str(simplified), steps=None, type="simplify")
            # In auto mode, we keep simplified for default result
            auto_result = simplified

        if mode == "derivative":
            var = symbols(req.variable) if req.variable else list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            res = diff(expr, var)
            return SolveResponse(result=str(simplify(res)), steps=None, type="derivative")

        if mode == "integral":
            var = symbols(req.variable) if req.variable else list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            res = integrate(expr, var)
            return SolveResponse(result=str(res), steps=None, type="integral")

        if mode == "limit":
            # Expect input like "f(x); x->a"
            # If variable not supplied, try x->0
            var = symbols(req.variable) if req.variable else symbols('x')
            res = sym_limit(expr, var, 0)
            return SolveResponse(result=str(res), steps=None, type="limit")

        if mode == "solve":
            from sympy import solve
            var = symbols(req.variable) if req.variable else None
            res = solve(expr, var) if var is not None else solve(expr)
            return SolveResponse(result=[str(r) for r in res], steps=None, type="solve")

        if mode == "matrix":
            # Attempt to interpret expression as Matrix, like [[1,2],[3,4]]
            m = Matrix(expr)
            return SolveResponse(result=str(m), steps=None, type="matrix")

        # Default: auto simplified result
        return SolveResponse(result=str(auto_result), steps=None, type="auto")

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
    expr_str = req.expression.strip()
    try:
        expr = parse_latex(expr_str) if req.latex else sympify(expr_str)
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


# History endpoints and premium flags could be implemented with DB, but keeping MVP stateless for now.

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
