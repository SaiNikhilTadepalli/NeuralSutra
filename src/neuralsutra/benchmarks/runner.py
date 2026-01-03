import multiprocessing
import time

from neuralsutra.verification import verify_integration


def sympy_worker(expr, var, queue):
    """Isolates SymPy so it can be timed out safely."""
    from sympy import integrate

    try:
        res = integrate(expr, var)
        queue.put(res)
    except Exception:
        queue.put(None)


class BenchmarkRunner:
    def __init__(self, compiler):
        self.compiler = compiler

    def run_case(self, name, data, var):
        print(f"\n[ CASE ] {name}")
        expr = data["expr"]

        # SymPy baseline
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=sympy_worker, args=(expr, var, queue))

        start_sympy = time.perf_counter()
        process.start()
        process.join(timeout=30)

        if process.is_alive():
            process.terminate()
            t_sympy = 30.0
            print(f"  - SymPy Baseline     : TIMEOUT (30s)")
        else:
            t_sympy = time.perf_counter() - start_sympy
            print(f"  - SymPy Baseline     : {t_sympy:.4f}s")

        # Directly call NeuralSutra to maintain model warm-state
        start_ns = time.perf_counter()
        try:
            v_res = self.compiler.compile(expr, var)
            t_ns = time.perf_counter() - start_ns
            print(f"  - NeuralSutra Engine : {t_ns:.4f}s")
        except Exception as e:
            print(f"  - NeuralSutra Engine : ERROR ({e})")
            t_ns = 999
            v_res = None

        # Display validation and performance metrics
        is_correct = verify_integration(expr, v_res, var)
        speedup = t_sympy / t_ns if t_ns > 0 else 0

        print(f"  - Verification       : {'PASSED' if is_correct else 'FAILED'}")
        print(f"  - System Speedup     : {speedup:.2f}x")
        print("-" * 50)


def run_benchmark_suite(compiler, cases=None):
    """
    Execute the full test battery.
    """
    from sympy import Symbol

    from neuralsutra.benchmarks.cases import get_benchmark_cases

    x = Symbol("x")
    runner = BenchmarkRunner(compiler)
    test_cases = cases or get_benchmark_cases()

    print("NEURALSUTRA: PERFORMANCE BENCHMARK")
    for name, data in test_cases.items():
        runner.run_case(name, data, x)
