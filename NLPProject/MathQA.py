import tensorflow_hub as hub
import json
import numpy as np
import math
import operator
import random
from collections import Counter

# Import svm model for approach 2
from sklearn import svm

"""
Idea behind approach: What if there is a pattern in which answers are set among the choices for similar questions?
"""

"""
Defining Functions
"""


# Function to retrieve and split questions and answers from input
def administer_questions(questions):
    # Initializing a dictionary to store question and its answer
    QA = []
    for question in questions:
        # guess = student.solve(question)
        answer = {'ques': question['question'], 'ans': question['answer'], 'id': question['id']}
        QA.append(answer)
    return QA


# Function to calculate cosine similarity:
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Function to write results on file
def write_answers_to_file(answers, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(answers, indent=4))
    #files.download(filename)


"""
Main Function
"""

# Reading train dataset json file
with open('sat.train.json') as f:
    questions = json.load(f)

# Loading tensorflow model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

# Calling function to split both questions and answers and return a dictionary from train dataset
QA = administer_questions(questions)

# Access elements in list of dictionaries and store in separate lists
question = []
an = []
for dict in QA:
    question.append(dict["ques"])
    an.append(dict["ans"])

# Creating sentence embeddings using the model
sentence_embeddings = model(question)

""""
APPROACH 1: USING USE MODEL TO CATEGORIZE QUESTIONS SIMILAR QUESTIONS AND DECALRING ANSWERS WITH MOST FREQUENT CHOICE IN SIMILAR QUESTIONS
"""

# Testing model with Test file questions


# Loading test file
with open('sat.dev.json') as f:
    test_questions = json.load(f)

test_dict = administer_questions(test_questions)

# Access questions in list of dictionaries from test dataset and store in separate lists
test_question = []
test_id = []
for dict in test_dict:
    test_question.append(dict["ques"])
    test_id.append(dict["id"])

i = 0  # iterator for test id index list
outputs = []

# Testing model with test questions
for query in test_question:
    # Convert the test question into vector
    query_vec = model([query])[0]

    # Calculating Similarity between test query and the list of questions
    quest_count = 0
    sim_count = 0
    rule1_ans = []
    for ind, quest in enumerate(question):
        sim = cosine(query_vec, model([quest])[0])
        # Selecting 0.6 as threshold because it was observed manually that this actually covers most of the similar questions
        if (sim > 0.6):
            sim_count += 1
            rule1_ans.append(an[ind])
            # print("Question = ", quest, "; similarity = ", sim)
        quest_count += 1
        # If no similar question is found, there could be two ways to deal with the situation:
        # 1. Randomly select a choice among answers
        # 2. Check for sim < 0.5
        # In our case, option 1 works (considering the time complexity which may further be increased due to runnning the entire loop on similarity < 0.6 which is not worth the efforrt)
        if (sim_count == 0 and ind == len(question) - 1):
            answ = random.choice(an)
            rule1_ans = answ
    # Calculate the number of times each choice has been selected
    rule1_count = Counter(rule1_ans)
    # If there are similar questions then only check the max ocurrence of choice
    if rule1_count != 0:
        # Finding the most frequent answer for the category
        answ = max(rule1_count.items(), key=operator.itemgetter(1))[0]

    # Printing results
    print("Total questions are: ", quest_count)
    print("Similar questions are: ", sim_count)
    print("Rule 1 answer counts", rule1_count)
    print("Rule 1 answer is ", answ)

    # Store question id and answer in a dictionary
    output = {'id': test_id[i], 'answer': answ}
    i += 1
    outputs.append(output)

# Write answer to the output file
write_answers_to_file(outputs, 'scoring_program\input\\res\output.json')

# ACCURACY achieved from Approach 1: 18.77 % (vs 17% for Baseline)

""""
APPROACH 2: USING SENTENCE EMBEDDINGS FROM U.S.E TO FEED SUPPORT VECTOR MACHINE LIKE AiFU DID WITH INFERSENT
"""
# Reference of code: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(sentence_embeddings, an)

# Predict the response for test dataset
sentence_embeddings_test = model(test_question)
y_pred = clf.predict(sentence_embeddings_test)

# Store question id and answer in a dictionary
outputs = []
for i in range(len(y_pred)):
    output = {'id': test_id[i], 'answer': y_pred[i]}
    outputs.append(output)

# Write answer to the output file
write_answers_to_file(outputs, 'scoring_program\input\\res\output2.json')

# ACCURACY achieved from Approach 2: 20.216 % with linear kernel, 17.328 % with rbf kernel, 17.14% with polynomial kernel (vs 17% for Baseline)