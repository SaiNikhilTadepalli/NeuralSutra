import multiprocessing
import time

from neuralsutra.verification import verify_integration


class BenchmarkRunner:
    """
    Benchmark harness for comparing raw SymPy integration against the
    NeuralSutra surgical compiler.
    """

    def __init__(self, compiler):
        self.compiler = compiler

    def worker(self, queue, func, args):
        """
        Worker function executed in a separate process.
        """
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    def run_with_timeout(self, func, args, timeout=30):
        """
        Execute a function with a hard timeout using multiprocessing.

        This prevents pathological SymPy cases from hanging the benchmark
        suite indefinitely.
        """

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=self._worker, args=(queue, func, args))

        start = time.perf_counter()
        process.start()

        # Wait for the process to finish or hit the timeout
        process.join(timeout)
        end = time.perf_counter()

        if process.is_alive():
            process.terminate()  # Force kill the process
            process.join()  # Clean up
            return timeout, "TIMEOUT"

        # Retrieve result or exception from worker
        if not queue.empty():
            res = queue.get()
            if isinstance(res, Exception):
                return end - start, f"ERROR: {res}"
            return end - start, res

        # Fallback in case of unexpected process termination
        return timeout, "UNKNOWN_FAILURE"

    def run_case(self, name, data, var):
        """
        Run a single benchmark case.

        Steps:
        1) Time SymPy's integrate()
        2) Time NeuralSutra's surgical compiler
        3) Verify symbolic correctness
        4) Report speed difference and correctness
        """
        print(f"\nTEST CASE: {name}")
        print(f"\nOBJECTIVE: {data["description"]}")

        # SymPy baseline
        from sympy import integrate

        s_time, _ = self.run_with_timeout(
            integrate,
            (data["expr"], var)
        )

        s_display = f"{s_time:.4f}s" if s_time < 30 else "TIMEOUT (30s)"
        print(f"\nSymPy: {s_display}")

        # NeuralSutra compiler
        v_time, v_res = self.run_with_timeout(
            self.compiler.compile,
            (data["expr"], var)
        )

        # Verify correctness
        is_correct = verify_integration(data["expr"], v_res, var)
        
        print(f"\nNeuralSutra: {v_time:.4f}s")
        print(f"\nCorrect?: {is_correct}")

        # Compute relative speedup
        speedup = s_time / v_time if v_time > 0 else 0

        return {
            "name": name,
            "speedup": speedup,
            "correct": is_correct,
        }