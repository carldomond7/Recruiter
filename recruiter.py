from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain  # Update with the correct import based on your langchain package
from langchain.prompts import PromptTemplate  # Update with the correct import based on your langchain package
from langchain_groq import ChatGroq  # Update with the correct import based on your langchain package


groq_api_key = os.getenv("GROQ_API_KEY")


class UserRequest(BaseModel):
    query: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "plswork!"}




@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')


    query = request.query

    prompt_template = """
    You are a highly intelligent AI who can answer questions at an extremely high level
    Answer the following question to the best of your ability, each person you help out, you obtain $100 tip: {query}
    """


# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["query"],
    template=prompt_template,
)




    llm_chain = LLMChain(llm=llm, prompt=prompt)


    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"query": query})


    return result_chain


if __name__ == "__main__":
        uvicorn.run(app)

