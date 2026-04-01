"""
    user will give text(detailed), we will generate two things from this document as follow:
        (i) Notes of text,
        (ii) Quiz
    Output will be Notes and quiz of topic, Thus it required the Parallel Chain.
"""

" Note : We are using multiple models(1,2,3..), but we can do same thing using single model only."

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGroq(model = "llama-3.3-70b-versatile" )
model2 = ChatGroq(model = "llama-3.3-70b-versatile" )
model3 = ChatGroq(model = "llama-3.3-70b-versatile" )

template1 = PromptTemplate(
    template= 'Imagine you are teacher who make imp notes of topic for students , so make a notes of: \n {text}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Imagine you are expert exam paper designer, make a quiz of 5 questions of following: \n {text}',
    input_variables= ['text']
)

template3 = PromptTemplate(
    template= 'Merge the provided notes and quiz into a single document. \n notes -> {notes} and quiz -> {quiz}',
    input_variables= ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model1 | parser,
    'quiz' : template2 | model2 | parser
})

merged_chain = template3 | model3 | parser

chain = parallel_chain | merged_chain

text = '''Logistic regression is a supervised learning algorithm used to predict the probability of a binary outcome (e.g., Yes/No, 0/1) by fitting data to an S-shaped sigmoid function. It maps linear input features to probabilities between 0 and 1, typically classifying outcomes based on a 0.5 threshold. [1, 2, 3, 4, 5]  
This video provides a quick overview of logistic regression: 
Key Aspects of Logistic Regression: 

• Functionality: Unlike linear regression, which predicts continuous numbers, logistic regression predicts the likelihood of categorical events. 
• The Sigmoid Function: The core mechanism is the formula $\sigma(z) = \frac{1}{1+e^{-z}}$, which converts linear predictions ($z$) into probabilities. 
• Applications: Commonly used for binary classification, such as spam detection, disease diagnosis, loan default prediction, and marketing conversions. 
• Model Training: The model finds the best-fitting weights for input features by maximizing the likelihood of observing the data, often using methods like maximum likelihood estimation. 
• Key Limitations: Requires that data points be independent and assumes little to no multicollinearity among independent variables. [1, 3, 5, 6, 7, 8, 9, 10]  

Key Concepts and Types: 

• Binary Logistic Regression: The most common form, used when the target variable has only two possible outcomes (e.g., true/false). 
• Multinomial Logistic Regression: Used when the output has three or more unordered classes (e.g., predicting "spam", "urgent", or "normal" email). 
• Ordinal Logistic Regression: Used when the output has three or more ordered categories (e.g., "low", "medium", "high" risk). [2, 5, 7, 11, 12]  

Key Metrics and Techniques: 

• Maximum Likelihood Estimation (MLE): The optimization technique used to determine the best parameters for the model. 
• Log-Odds (Logit): The model predicts the log of the odds for the target variable. 
• Regularization: Techniques like L1 (Lasso) and L2 (Ridge) are often applied to prevent overfitting. [8, 10, 13, 14, 15]  

For implementation, libraries such as scikit-learn offer built-in logistic regression tools that support various solvers and regularization methods.'''

result = chain.invoke({'text' : text})

print(result)