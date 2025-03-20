PROMPT_PAIRWISE="""
Please act as an impartial contextual judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below given the context. You should choose the assistant that answers the user\'s question better and remains more faithful to the given context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
If both responses are equally faithful, your evaluation should consider {criteria}. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

[The Start of Context]
{context}
[The End of Context]
""".strip()

PROMPT_PAIRWISE_CONDITIONAL="""
Please act as an impartial contextual judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below given the context. You should choose the assistant that answers the user\'s question better and remains more faithful to the given context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
First, determine if Response A is faithful to the context. Second, determine if Response B is faithful to the context. If one response is faithful while the other response is not, select the faithful response. If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria}.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "yes/no" for Output (a) faithfulness, then "yes/no" for Output (b) faithfulness then "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

[The Start of Context]
{context}
[The End of Context]
""".strip()

PROMPT_PAIRWISE_GENERIC="""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question and context displayed below. You should choose the assistant answers the user\'s question better. 
Your evaluation should consider factors such as {criteria} Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

[The Start of Context]
{context}
[The End of Context]
"""

def pairwise_parsing(judge_output, flip=False):
    if not flip:
        if '[[A]]' in judge_output:
            return 1
        elif '[[B]]' in judge_output:
            return 2
        else:
            return -1
    else:
        if '[[A]]' in judge_output:
            return 2
        elif '[[B]]' in judge_output:
            return 1
        else:
            return -1

pairwise_parsing_conditional = pairwise_parsing # no changes to conditional parsing