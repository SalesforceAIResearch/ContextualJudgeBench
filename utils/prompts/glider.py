PROMPT_PAIRWISE = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
<CONTEXT> 
{context}
</CONTEXT>

<USER INPUT>
{question}
</USER INPUT>

<MODEL OUTPUT 1>
{response_a}
</MODEL OUTPUT 1>

<MODEL OUTPUT 2>
{response_b}
</MODEL OUTPUT 2>

Pass Criteria:
Compare the MODEL OUTPUT 1 and MODEL OUTPUT 2 and select the response that better accomplishes the task of answering the user's request given the provided context.
Prioritize evaluating whether the response is faithful to the context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria} - if both seem equal, you must choose the one with even slightly better qualities.

Rubric:
0: The MODEL OUTPUT 1 is better than MODEL OUTPUT 2. 
1: The MODEL OUTPUT 2 is better than MODEL OUTPUT 1.

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""

PROMPT_PAIRWISE_CONDITIONAL = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
<CONTEXT> 
{context}
</CONTEXT>

<USER INPUT>
{question}
</USER INPUT>

<MODEL OUTPUT 1>
{response_a}
</MODEL OUTPUT 1>

<MODEL OUTPUT 2>
{response_b}
</MODEL OUTPUT 2>

Pass Criteria:
Compare the MODEL OUTPUT 1 and MODEL OUTPUT 2 and select the response that better accomplishes the task of answering the user's request given the provided context.
Prioritize evaluating whether the response is faithful to the context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
First, determine if MODEL OUTPUT 1 is faithful to the context. Provide reasoning for your decision, then write your response in the <reasoning> block as MODEL OUTPUT 1 Reasoning: <reasoning for MODEL OUTPUT 1 faithfulness> MODEL OUTPUT 1 Faithfulness: <yes/no>
Second, determine if MODEL OUTPUT 2 is faithful to the context. Provide reasoning for your decision, then write your response in the <reasoning> block as MODEL OUTPUT 2 Reasoning: <reasoning for MODEL OUTPUT 2 faithfulness> MODEL OUTPUT 2 Faithfulness: <yes/no>
If one response is faithful while the other response is not, select the faithful response. If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria}.

Rubric:
0: The MODEL OUTPUT 1 is better than MODEL OUTPUT 2. 
1: The MODEL OUTPUT 2 is better than MODEL OUTPUT 1.

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""

PROMPT_PAIRWISE_GENERIC = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
<CONTEXT> 
{context}
</CONTEXT>

<USER INPUT>
{question}
</USER INPUT>

<MODEL OUTPUT 1>
{response_a}
</MODEL OUTPUT 1>

<MODEL OUTPUT 2>
{response_b}
</MODEL OUTPUT 2>

Pass Criteria:
Compare the MODEL OUTPUT 1 and MODEL OUTPUT 2 and select the response that better accomplishes the task of answering the user's request given the provided context based on {criteria}.

Rubric:
0: The MODEL OUTPUT 1 is better than MODEL OUTPUT 2. 
1: The MODEL OUTPUT 2 is better than MODEL OUTPUT 1.

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""

def pairwise_parsing(judge_output, flip=False):
    start_tag = '<score>'
    end_tag = '</score>'
    
    start_index = judge_output.find(start_tag) + len(start_tag)
    end_index = judge_output.find(end_tag)
    
    if start_index == -1 or end_index == -1:
        return -1
    
    score = judge_output[start_index:end_index].strip()
    
    try:
        judgment = float(score)
        if not flip:
            if judgment == 0:
                return 1
            elif judgment == 1:
                return 2
            else:
                return -1
        else:
            if judgment == 0:
                return 2
            elif judgment == 1:
                return 1
            else:
                return -1

    except ValueError:
        return -1

pairwise_parsing_conditional = pairwise_parsing