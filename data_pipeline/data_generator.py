import random
import sympy as sp
import json
import os

class SLaNgConverter:
    """Converts SymPy expressions and operations to SLaNg format and vice versa."""
    
    @staticmethod
    def to_slang(expr, op_type=None, var=None, limits=None):
        if op_type == "diff":
            return f"DIFF[{expr}, {var}]"
        elif op_type == "int":
            if limits:
                return f"INT[{expr}, {var}, {limits[0]}, {limits[1]}]"
            return f"INT[{expr}, {var}]"
        elif op_type == "lim":
            return f"LIM[{expr}, {var}, {limits}]"
        return str(expr).replace("**", "^")

    @staticmethod
    def from_slang(slang_str):
        # Basic cleanup for sympy parsing
        return slang_str.replace("^", "**")

class CalculusDatasetGenerator:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.functions = [
            self.x, self.x**2, self.x**3, sp.sin(self.x), sp.cos(self.x),
            sp.exp(self.x), sp.log(self.x), sp.tan(self.x), sp.sqrt(self.x)
        ]
        self.converter = SLaNgConverter()

    def generate_problem(self):
        op = random.choice(["diff", "int", "lim"])
        f1 = random.choice(self.functions)
        f2 = random.choice(self.functions)
        
        # Create a combined expression
        expr = f1 * random.choice([1, 2, 3]) + f2
        
        if op == "diff":
            solution = sp.diff(expr, self.x)
            problem_slang = self.converter.to_slang(expr, "diff", "x")
            solution_slang = self.converter.to_slang(solution)
        elif op == "int":
            # Keep it simple for integration to ensure we get a solution
            expr = random.choice(self.functions)
            solution = sp.integrate(expr, self.x)
            problem_slang = self.converter.to_slang(expr, "int", "x")
            solution_slang = self.converter.to_slang(solution)
        else: # lim
            point = random.choice([0, 1, sp.oo])
            try:
                solution = sp.limit(expr, self.x, point)
                problem_slang = self.converter.to_slang(expr, "lim", "x", point)
                solution_slang = self.converter.to_slang(solution)
            except:
                return self.generate_problem() # Retry if limit fails
        
        return {"input": problem_slang, "output": solution_slang}

    def generate_dataset(self, size=1000):
        dataset = []
        for _ in range(size):
            dataset.append(self.generate_problem())
        return dataset

if __name__ == "__main__":
    generator = CalculusDatasetGenerator()
    data = generator.generate_dataset(size=10000)
    
    os.makedirs("CalculusSolver/data", exist_ok=True)
    with open("CalculusSolver/data/dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(data)} problems in SLaNg format.")
