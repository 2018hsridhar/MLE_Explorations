'''
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

targetModel = "gpt-3.5-turbo"
targetTemp = 0
model = ChatOpenAI(model=targetModel, temperature=targetTemp)


while True:
    user_question = input("Enter your question: ")
    if user_question.lower() in ['exit', 'quit']:
        print("Exiting the program.")
        break
    response = model.invoke(user_question)
    print("Answer:", response.content)