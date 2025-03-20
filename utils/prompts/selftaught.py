PROMPT_PAIRWISE_SYSTEM="""
Please act as an impartial contextual judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below given the context. You should choose the assistant that answers the user\'s question better and remains more faithful to the given context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer. If both responses are equally faithful, your evaluation should consider {criteria}. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.
""".strip()

PROMPT_PAIRWISE_SYSTEM_CONDITIONAL="""
Please act as an impartial contextual judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below given the context. You should choose the assistant that answers the user\'s question better and remains more faithful to the given context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer. First, determine if Response A is faithful to the context. Write your response in your feedback as "Response A faithfulness reasoning:  <reasoning for response A faithfulness> Response A faithfulness: <yes/no>". Second, determine if Response B is faithful to the context. Write your response in your feedback as "Response B faithfulness reasoning:  <reasoning for response B faithfulness> Response B faithfulness: <yes/no>". If one response is faithful while the other response is not, select the faithful response. If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria}. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.
""".strip()

PROMPT_PAIRWISE_SYSTEM_GENERIC="""
'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that answers the user\'s question better. Your evaluation should consider factors such as {criteria} Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.'
""".strip()


PROMPT_PAIRWISE="""
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


pairwise_parsing_conditional = pairwise_parsing