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



Resources = https://www.youtube.com/watch?v=UXGNpuHgT5g

'''

# pip3 install langchain_openai


from langchain_openai import ChatOpenAI

def create_langchain_model():
    api_key = ""
    targetModel = "gpt-3.5-turbo"
    targetTemp = 0
    max_tokens = 1024
    langchainModel = ChatOpenAI(model=targetModel, temperature=targetTemp, openai_api_key=api_key, max_tokens=max_tokens)
    return langchainModel

# Function to truncate prompt if too long
# Assuming max prompt length is 500 characters for this example
def truncate_prompt(prompt, max_length=500):
    """Truncate prompt to max_length characters."""
    return prompt[:max_length]

# Interactive loop to ask questions
def ask_questions():
    langchainModel = create_langchain_model()
    while(True):
        user_question = input("Enter your question: ")
        exitResponses = ['exit', 'quit', 'q', 'close', 'stop']
        if user_question.lower() in exitResponses:
            print(f"Exiting the program.")
            break
        # Truncate prompt if too long
        user_question = truncate_prompt(user_question)
        response = langchainModel.invoke(user_question)
        print("Answer:", response.content)

def testLangChain():
    langchainModel = create_langchain_model()
    test_prompts_and_answers = [
        ("What's the capital of India?", "New Delhi"),
        ("What's the largest city in the world?", "Tokyo"),
        ("I am Hari. I am a senior software engineer. I live in San Francisco!\nQuestions:\n1. What is your name?\n2. Where do you live?\n3. What is your profession?", 
         "1. My name is Hari.\n2. I live in San Francisco.\n3. I am a senior software engineer.")
    ]
    for test_prompt, expected_answer in test_prompts_and_answers:
        response = langchainModel.invoke(test_prompt)
        print("Test Answer:", response.content)
        print("Expected Answer:", expected_answer)
        print("Test Passed!" if response.content == expected_answer else "Test Failed!")

def main():
    testLangChain() # Run test

if __name__ == "__main__":
    main()