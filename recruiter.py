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
    job_title: str
    job_des: str
    resume: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "plswork!"}




@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')


    query = request.query
    job_title = request.job_title
    job_des = request.job_des
    resume = request.resume


    prompt_template = """
    You are a professional recruiter who specializes in cultivating talent, you are very knowledgeable about all types of jobs.
    Here is a job profile:
   
    Job Title: {job_title}
    Description: {job_des}
    Person Resume: {resume}
   
    Based on this profile answer the following question to the best of your ability, each person you help out, you obtain $100 tip: {query}
    """


# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["job_title", "query", "job_des", "skills", "resume"],
    template=prompt_template,
)




    llm_chain = LLMChain(llm=llm, prompt=prompt)


    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"job_title": job_title, "query": query, "job_des": job_des, "resume": resume})


    return result_chain


if __name__ == "__main__":
        uvicorn.run(app)

