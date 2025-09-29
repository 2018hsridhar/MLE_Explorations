'''
How LangChain OpenAI ( and wrapper ) works?
How LangChain deals with memory?

Limit lengths of prompts and responses :
- Max tokens for prompts: 4096
- Max tokens for responses: 1024
- Total max tokens: 4096
- Control costs and avoid excessive token usage

Question Test Cases :
What's the capital of India? -> New Dehli
What's the largest city in the world? -> Tokyo

Test :
I am Hari. I am a senior software engineer. I live in San Francisco!
Questions:
1. What is your name?
2. Where do you live?
3. What is your profession?



Asking questions : related or unrelated?
Do we give context or not?

Yes, providing context can help the model generate more accurate and relevant responses.
OpenAI Key = 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



Resources = https://www.youtube.com/watch?v=UXGNpuHgT5g

'''

# pip3 install langchain_openai


from langchain_openai import ChatOpenAI

# export OPENAI_API_KEY="sk-..."  # Replace with your actual API key


api_key= "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"
targetModel = "gpt-3.5-turbo"
targetTemp = 0
max_tokens = 1024
model = ChatOpenAI(model=targetModel, temperature=targetTemp, openai_api_key=api_key, max_tokens=max_tokens)

# Function to truncate prompt if too long
# Assuming max prompt length is 500 characters for this example
def truncate_prompt(prompt, max_length=500):
    """Truncate prompt to max_length characters."""
    return prompt[:max_length]

# Interactive loop to ask questions
while True:
    user_question = input("Enter your question: ")
    exitResponses = ['exit', 'quit', 'q', 'close', 'stop']
    if user_question.lower() in exitResponses:
        print(f"Exiting the program.")
        break
    # Truncate prompt if too long
    user_question = truncate_prompt(user_question)
    response = model.invoke(user_question)
    print("Answer:", response.content)

def testLangChain():
    test_prompt = "What's the capital of India?"
    response = model.invoke(test_prompt)
    print("Test Answer:", response.content)