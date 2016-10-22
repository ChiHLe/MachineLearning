import math
__author__ = 'Chi Le'
# Naive Bayes Algorithm for document classification
# Accuracy for given test set: 94.4%

# Bayes Theorem:
# P(RED | Doc) = P(RED) * P(Doc|RED) / P(Doc)
# = P(RED) * P(word1|RED)*P(word2|RED)*... / P(Doc)

# Compare P(RED | Doc) to P(BLUE | Doc):
# P(RED) * P(word1|RED)*P(word2|RED)*...
# P(BLUE) * P(word1|BLUE)*P(word2|BLUE)*...

# Get training data
red_train = []
blue_train = []
#Counter for number of documents of each category
red = 0
blue = 0

#Open training file
#Each line is a document
with open('train.txt') as file:
    for line in file:
        #Strip each line into tokens
        tokens = line.rstrip().split()
        #If the document is RED, add tokens to red_train and increment red counter
        if tokens[0] == 'RED':
            red_train += tokens[1:]
            red += 1
        else:
            blue_train += tokens[1:]
            blue += 1

print(red, blue)
# Probability if a document is red or blue
p_red = float(red)/(blue+red)
p_blue = 1 - p_red

vocab = set(red_train + blue_train)
# Initialize the values of tokens in each dictionary to 0
red_dict = {key: 1./(len(red_train) + len(vocab)) for key in vocab}
blue_dict = {key: 1./(len(blue_train) + len(vocab)) for key in vocab}

# Go through the training data and update probability
for token in red_train:
    red_dict[token] += 1./(len(red_train) + len(vocab))

for token in blue_train:
    blue_dict[token] += 1./(len(blue_train) + len(vocab))



# Predict test data
print("Predict:\tActual:")
error = 0
count = 0

# Read test file
# Log(Product of prob) = Sum of log prob

# Calculate error in prediction
with open('test.txt') as test_file:
    for line in test_file:
        tokens = line.rstrip().split()
        p_test_blue = math.log(p_blue)
        p_test_red = math.log(p_red)
        #Read each document, ignore first token with classification
        for token in tokens[1:]:
            #Only consider tokens that have been trained
            if token in vocab:
                p_test_blue += math.log(blue_dict[token])
                p_test_red += math.log(red_dict[token])
        #Compare values for prediction
        if p_test_red > p_test_blue:
            pred = 'RED'
        else:
            pred = 'BLUE'
        print(pred + '\t' + tokens[0])
        count += 1
        if pred != tokens[0]:
            error += 1.

    print("Classification Accuracy: " + str(1 - error/count))
