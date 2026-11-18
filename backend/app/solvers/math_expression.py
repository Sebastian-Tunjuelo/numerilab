import math
import re

import sympy as sp


MAX_EXPRESSION_LENGTH = 300
_NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_NAME = r"[A-Za-z][A-Za-z0-9]*"
_TOKEN = re.compile(rf"\s+|{_NUMBER}|{_NAME}|\*\*|[+*/^()=,-]")
_CONSTANTS = {"e": sp.E, "E": sp.E, "pi": sp.pi}
_FUNCTIONS = {
    "sin": sp.sin, "sen": sp.sin, "cos": sp.cos, "tan": sp.tan, "tg": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
}


def parse_math_expression(text: str, variables: list[str]) -> tuple[sp.Expr, list[sp.Symbol]]:
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"La expresión supera {MAX_EXPRESSION_LENGTH} caracteres.")
    if len(set(variables)) != len(variables) or any(
        not re.fullmatch(_NAME, name) or name in _CONSTANTS or name in _FUNCTIONS
        for name in variables
    ):
        raise ValueError("Usa nombres de variables distintos y evita nombres de funciones o constantes.")

    symbols = [sp.Symbol(name, real=True) for name in variables]
    names = _CONSTANTS | dict(zip(variables, symbols))
    tokens = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"Símbolo no admitido en la posición {position + 1}.")
        token = match[0]
        position = match.end()
        if token.isspace():
            continue
        if token.isalpha() and token not in names and token not in _FUNCTIONS:
            if all(letter in variables for letter in token):
                tokens.extend(token)
                continue
        tokens.append(token)

    if len(tokens) > 120:
        raise ValueError("La expresión contiene demasiados términos.")
    parser = _ExpressionParser(tokens, names)
    expression = parser.expression()
    if parser.peek() == "=":
        parser.take()
        right = parser.expression()
        if right != 0:
            expression = sp.Add(expression, -right, evaluate=False)
    if parser.peek():
        leftover = parser.peek()
        if leftover == "=":
            raise ValueError("La expresión solo admite una igualdad con dos miembros completos.")
        if leftover == ")":
            raise ValueError("Hay un paréntesis de cierre sin abrir.")
        if leftover == ",":
            raise ValueError("Coma inesperada fuera de un logaritmo de dos argumentos.")
        raise ValueError(f"Token inesperado '{leftover}' al final de la expresión.")
    return expression, symbols


class _ExpressionParser:
    def __init__(self, tokens: list[str], names: dict[str, sp.Expr]):
        self.tokens = tokens
        self.names = names
        self.position = 0

    def peek(self) -> str:
        return self.tokens[self.position] if self.position < len(self.tokens) else ""

    def take(self) -> str:
        token = self.peek()
        self.position += 1
        return token

    def expect_close(self) -> None:
        if self.take() != ")":
            raise ValueError("Falta cerrar un paréntesis.")

    def expression(self, precedence: int = 0, depth: int = 0) -> sp.Expr:
        if depth > 24:
            raise ValueError("La expresión contiene demasiados niveles anidados.")
        token = self.take()
        if token in ("+", "-"):
            left = self.expression(3, depth + 1)
            if token == "-":
                left = self.combine("*", sp.Integer(-1), left)
        elif token == "(":
            left = self.expression(0, depth + 1)
            self.expect_close()
        elif token in _FUNCTIONS:
            if self.take() != "(":
                raise ValueError(f"Escribe {token}(...) con paréntesis.")
            if self.peek() == ")":
                raise ValueError(f"La función '{token}' requiere al menos un argumento.")
            args = [self.expression(0, depth + 1)]
            if token in ("log", "ln") and self.peek() == ",":
                self.take()
                base = self.expression(0, depth + 1)
                if not base.free_symbols:
                    simplified = sp.simplify(base)
                    if simplified == 0 or simplified == 1:
                        raise ValueError("La base del logaritmo debe ser positiva y distinta de 1.")
                    if simplified.is_finite is False or simplified in (sp.nan, sp.zoo, sp.oo, -sp.oo):
                        raise ValueError("La base del logaritmo debe ser finita.")
                    evaluated = simplified.evalf(32)
                    re_part, im_part = evaluated.as_real_imag()
                    try:
                        re_val = float(re_part)
                        im_val = float(im_part)
                    except (TypeError, OverflowError) as exc:
                        raise ValueError("La base del logaritmo debe ser un número real finito.") from exc
                    if not math.isfinite(re_val) or not math.isfinite(im_val) or abs(im_val) > 1e-30:
                        raise ValueError("La base del logaritmo debe ser un número real.")
                    if re_val <= 0.0 or re_val == 1.0:
                        raise ValueError("La base del logaritmo debe ser positiva y distinta de 1.")
                args.append(base)
            if self.peek() == ",":
                raise ValueError(f"Número de argumentos inválido para la función '{token}'.")
            self.expect_close()
            try:
                left = _FUNCTIONS[token](*args)
            except TypeError as exc:
                raise ValueError(f"Argumentos inválidos para la función '{token}'.") from exc
        elif token in self.names:
            left = self.names[token]
        elif re.fullmatch(_NUMBER, token):
            exp_match = re.search(r"[eE]([+-]?\d+)$", token)
            if exp_match and abs(int(exp_match.group(1))) > 325:
                raise ValueError("El número queda fuera del rango admitido.")
            if not math.isfinite(float(token)):
                raise ValueError("El número queda fuera del rango admitido.")
            left = sp.Integer(token) if token.isdecimal() else sp.Float(token)
        else:
            raise ValueError(f"Se esperaba un número, una variable o una función ({token or 'fin de expresión'}).")

        while (token := self.peek()) and token not in (")", ",", "="):
            implicit = token == "(" or bool(re.fullmatch(_NAME, token))
            level = 2 if implicit else {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "**": 3}.get(token)
            if level is None:
                raise ValueError("Falta un operador entre términos.")
            if level < precedence:
                break
            operator = "*" if implicit else self.take()
            right = self.expression(level if level == 3 else level + 1, depth + 1)
            left = self.combine(operator, left, right)
        return left

    @staticmethod
    def combine(operator: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
        if operator == "+":
            return sp.Add(left, right, evaluate=False)
        if operator == "-":
            return sp.Add(left, sp.Mul(-1, right, evaluate=False), evaluate=False)
        if operator == "*":
            return sp.Mul(left, right, evaluate=False)
        if operator == "/":
            return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
        if right.is_number:
            # Acota las potencias numéricas antes de construirlas o diferenciarlas.
            try:
                exponent = float(right)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("El exponente constante queda fuera del rango admitido.") from exc
            if not math.isfinite(exponent) or abs(exponent) > 1024:
                raise ValueError("El exponente constante queda fuera del rango admitido.")
            if left.is_number:
                try:
                    magnitude = abs(float(left)) ** exponent
                except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                    raise ValueError("La potencia numérica queda fuera del rango admitido.") from exc
                if not math.isfinite(magnitude):
                    raise ValueError("La potencia numérica queda fuera del rango admitido.")
        return sp.Pow(left, right, evaluate=False)
