""" 
    Outout parsers for response formates.
    1. CoT output parser
    2. TIR output parser
"""
import re
import io
import contextlib
from llm_sandbox import SandboxSession

def parser_cot_response(response):
    """
        Parse the Chain-of-Thought (CoT) response to extract the final answer.
    """
    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, response)
    if match:
        answer = match.group(1).strip()
        return answer
    else:
        return "No valid answer found in the response."
    
def parser_tir_response(response):
    """
        Parsing python code block from the response and need to execute it.
    """
    pattern = r"```python(.*?)```"
    match = re.search(pattern, response, re.DOTALL)

    # get the output pattern 
    pattern_output = r"```output(.*?)```"
    match_output = re.search(pattern_output, response, re.DOTALL)
    if all([match, match_output]):
        code_block = match.group(1).strip()
        output_block = match_output.group(1).strip()
        return code_block, output_block
    elif match:
        code_block = match.group(1).strip()
        return code_block, None
    else:
        return None, None
    

# Code execution with no safety guarantees
def run_code_non_safe(code: str) -> str:
    buf = io.StringIO()
    ns = {}  # isolated namespace
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(code, ns)
    return buf.getvalue()


# Executing code in a sandboxed environment
def run_code_safely_sandbox(code: str) -> str:
    with SandboxSession(lang="python") as session:
        output = session.run(code)
    return output.stdout




# sanity checking
if __name__ == "__main__":
    tir_response = """
        Let's break down the problem step-by-step and use Python code to ensure our calculations are accurate.

        1. Kevin starts at 0 and wants to get to 1.
        2. On his first hop, he hops $\frac{1}{3}$ of the total distance (which is 1), so he hops $\frac{1}{3}$.
        3. After the first hop, the remaining distance is $1 - \frac{1}{3} = \frac{2}{3}$.
        4. On his second hop, he hops $\frac{1}{3}$ of the remaining distance, which is $\frac{1}{3} \times \frac{2}{3} = \frac{2}{9}$.
        5. After the second hop, the remaining distance is $\frac{2}{3} - \frac{2}{9} = \frac{6}{9} - \frac{2}{9} = \frac{4}{9}$.
        6. On his third hop, he hops $\frac{1}{3}$ of the remaining distance, which is $\frac{1}{3} \times \frac{4}{9} = \frac{4}{27}$.
        7. After the third hop, the remaining distance is $\frac{4}{9} - \frac{4}{27} = \frac{12}{27} - \frac{4}{27} = \frac{8}{27}$.
        8. On his fourth hop, he hops $\frac{1}{3}$ of the remaining distance, which is $\frac{1}{3} \times \frac{8}{27} = \frac{8}{81}$.
        9. After the fourth hop, the remaining distance is $\frac{8}{27} - \frac{8}{81} = \frac{24}{81} - \frac{8}{81} = \frac{16}{81}$.
        10. On his fifth hop, he hops $\frac{1}{3}$ of the remaining distance, which is $\frac{1}{3} \times \frac{16}{81} = \frac{16}{243}$.
        11. After the fifth hop, the remaining distance is $\frac{16}{81} - \frac{16}{243} = \frac{48}{243} - \frac{16}{243} = \frac{32}{243}$.

        The total distance Kevin has hopped after five hops is the sum of the distances of each hop:
        \[
        \frac{1}{3} + \frac{2}{9} + \frac{4}{27} + \frac{8}{81} + \frac{16}{243}
        \]

        Let's calculate this sum using Python to ensure accuracy.
        ```python
from fractions import Fraction

# Calculate the distance for each hop
first_hop = Fraction(1, 3)
second_hop = Fraction(1, 3) * Fraction(2, 3)
third_hop = Fraction(1, 3) * second_hop
fourth_hop = Fraction(1, 3) * third_hop
fifth_hop = Fraction(1, 3) * fourth_hop

# Sum the distances of each hop
total_distance = first_hop + second_hop + third_hop + fourth_hop + fifth_hop
print(total_distance)
        ```
        ```output
        121/243
        ```
        After calculating the total distance Kevin has hopped after five hops, we find that the total distance is $\frac{121}{243}$.

        Thus, the final answer is:
        \[
        \boxed{\frac{121}{243}}
        \]
    """
    

    # print("Parsed CoT Response:", parser_cot_response(cot_response))
    code_block, output_block = parser_tir_response(tir_response)

    print("Code Block:\n", code_block)

    code_block = "print('Hello, World!')"
    if code_block:
        print("Extracted Code Block:\n", code_block)
        execution_output = run_code_safely_sandbox(code_block)
        print("Execution Output:\n", execution_output)
    else:
        print("No code block found in the response.")
    # print("Parsed TIR Response:", parser_tir_response(tir_response))
    