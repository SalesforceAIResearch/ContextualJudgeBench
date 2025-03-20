PROMPT_PAIRWISE="""
###Task Description:
A question, a context, and two responses to evaluate are given.
1. Write a detailed feedback that assess the quality of the responses based on whether the response is faithful to the context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
2. If one response is faithful while the other response is not, select the faithful response. If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria}.
3. After writing a feedback, choose a better response between Response A and Response B. You should refer to the context and choose the response that is more faithful or based on the criteria if equally faithful.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{question}

###Response A:
{response_a}

###Response B:
{response_b}

###Context:
{context}

###Feedback: 
""".strip()

PROMPT_PAIRWISE_CONDITIONAL="""
###Task Description:
A question, a context, and two responses to evaluate are given.
1. Write a detailed feedback that assess the quality of the responses based on whether the response is faithful to the context. A response is faithful to the context if all of the factual information in the response is attributable to the context. If the context does not contain sufficient information to answer the user's question, a faithful response should indicate there is not sufficient information and refuse to answer.
2. First, determine if Response A is faithful to the context. Provide reasoning for your decision, then write your response in your feedback as "Response A faithfulness reasoning: <reasoning for response A faithfulness> Response A faithfulness: <yes/no>"
3. Second, determine if Response B is faithful to the context. Provide reasoning for your decision, then write your response in your feedback as "Response B faithfulness reasoning: <reasoning for response B faithfulness> Response B faithfulness: <yes/no>"
4. If one response is faithful while the other response is not, select the faithful response. If both responses are equally faithful to the context, prioritize evaluating responses based on {criteria}.
5. After writing a feedback, choose a better response between Response A and Response B. You should refer to the context and choose the response that is more faithful or based on the criteria if equally faithful.
6. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
7. Please do not generate any other opening, closing, and explanations.

###Instruction:
{question}

###Response A:
{response_a}

###Response B:
{response_b}

###Context:
{context}

###Feedback: 
""".strip()

PROMPT_PAIRWISE_GENERIC="""
###Task Description:
A question, a context, and two responses to evaluate are given.
1. Write a detailed feedback that assess the quality of the responses based {criteria}
3. After writing a feedback, choose a better response between Response A and Response B.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{question}

###Response A:
{response_a}

###Response B:
{response_b}

###Context:
{context}

###Feedback: 
""".strip()


def pairwise_parsing(judge_output, flip=False):
    critique_judgement = judge_output.split('[RESULT]')
    judgement = critique_judgement[-1].strip()
    
    if not flip:
        if judgement == 'A':
            return 1
        elif judgement == 'B':
            return 2
        else:
            return -1
    else:
        if judgement == 'A':
            return 2
        elif judgement == 'B':
            return 1
        else:
            return -1

pairwise_parsing_conditional = pairwise_parsing
