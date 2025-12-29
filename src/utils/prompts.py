"""
    Prompt utilities module.
    - System - level prompt , which will helps to improve the reasoning ability of the model.
"""
code_prompt = """Below is a math problem, you are to solve ( Non-negative integer answers only):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem \
by listing each step to take and what functions need to be called in each step.
Be clear so even an idiot can follow your instructions, and remember, your final \
answer should be a non-negative integer. not an algebraic expression!
Be skeptical of your answers.
Be skeptical of your answers.
Be skeptical of your answers.
Write the entire script convering all the steps (use comments and document it well) \
and print the result. After solving the problem, output the final Non-negative integer answer within \\boxed{}.

Approach:
""".strip()



# CoT prompt for general reasoning
cot_prompt = """Below is a math problem you are to solve ( Non-negative integer answer!):
\"{}\"

Be skeptical of your answers.
Be skeptical of your answers.
Be skeptical of your answers.
Analyze this problem and think step by step to come to a solution with program. After solving the problem,\
output the final Non-negative integer answer within \\boxed{}.

""".strip()