CRITIQUE_EVAL_PROMPT="""
You are given an <evaluation explanation>, a <evaluation outcome>, and a set of <criteria>. 
Another large language model conducted a pairwise evaluation between two responses, Response A and Response B. 
Based on the content of the <evaluation explanation>, your task is to decide if the <evaluation outcome> was decided based on <criteria>.
The <evaluation explanation> is allowed to mention criteria other than <criteria>. But it must use <criteria> as the primary criteria in its decision.

<evaluation explanation>: {critique}
<evaluation outcome>: {judgment}
<criteria>: {criteria}

Please give a short explanation, then respond with Yes or No. Use the format
<explanation>: your explanation
<decision>: Yes or No
""".strip()

REFUSAL_CRITERIA_CRITIQUE_EVAL="Refusal validity / faithfulness: The chosen response either correctly refuses to answer or correctly does not refuse and answers. This makes the chosen response appropriately faithful."
FAITHFULNESS_CRITERIA_CRITIQUE_EVAL="Faithfulness: The chosen response is more faithful, factual, or truthful"
CONCISENESS_CRITERIA_CRITIQUE_EVAL="Conciseness: The chosen response is more concise or less wordy or verbose"
COMPLETENESS_CRITERIA_CRITIQUE_EVAL="Completeness: The chosen response is more complete, thorough, or comprehensive"

criteria_dict_critique_eval = {
    'faithfulness': FAITHFULNESS_CRITERIA_CRITIQUE_EVAL,
    'conciseness': CONCISENESS_CRITERIA_CRITIQUE_EVAL,
    'completeness': COMPLETENESS_CRITERIA_CRITIQUE_EVAL,
    'refusal': REFUSAL_CRITERIA_CRITIQUE_EVAL,
}