from transformers import Gemma3Model

from hf_model_pipeline import HFModelPipeline, HFModelDownloader
import sys

Gemma3_Model_ID = "google/gemma-3-1b-it"

if __name__ == "__main__":
    # get the C/C++ code to analyze from cli input (path)
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_cpp_file>")
        sys.exit(1)

    cpp_file_path = sys.argv[1]
    try:
        with open(cpp_file_path, 'r') as file:
            cpp_code = file.read()
    except FileNotFoundError:
        print(f"File not found: {cpp_file_path}")
        sys.exit(1)

    model_path = HFModelDownloader.get_local_model_path(Gemma3_Model_ID)
    pipeline = HFModelPipeline(model_path)

    prompt = f"""
    <start_of_turn>user
    You are a C/C++ security expert. Follow these steps:
    1. Identify vulnerabilities by line.
    2. Reply using exactly this template (one per vulnerability):
    line <line_number>: vulnerability: <desc>; fix suggestion: <fix>
    No extra text or commentary.

    Now analyze this code:
    <<CODE>>
    {cpp_code}
    <<END CODE>>
    <end_of_turn>
    <start_of_turn>model
    """

    result = pipeline.get_model_response(prompt)
    print(f"Analyzer output:\n\n{result}")

    file.close()