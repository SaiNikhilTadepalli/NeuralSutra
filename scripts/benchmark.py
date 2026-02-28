from neuralsutra.benchmarks.runner import run_benchmark_suite
from neuralsutra.compiler import Compiler


def main() -> None:
    # Load the compiler
    try:
        compiler = Compiler("models/router.pth", "models/vocab.json")
    except FileNotFoundError:
        return print("Error: model files missing from 'models/'.")

    # Run the suite
    run_benchmark_suite(compiler)


if __name__ == "__main__":
    main()
