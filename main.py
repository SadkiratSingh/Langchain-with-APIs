import yaml
import os
import tiktoken

from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits.openapi import planner
from langchain.requests import RequestsWrapper
from langchain.llms.openai import OpenAI

#reduce spec
raw_petstore_api_spec = None
with open("./openapi-specs/cricket_openapi.yaml") as f:
    raw_petstore_api_spec = yaml.load(f, Loader=yaml.Loader)
petstore_api_spec = reduce_openapi_spec(raw_petstore_api_spec)
requests_wrapper = RequestsWrapper()

#encode spec
spec_encoding_model = tiktoken.encoding_for_model('text-davinci-003')
encoding_tokens = spec_encoding_model.encode(yaml.dump(raw_petstore_api_spec))
print(len(encoding_tokens))

llm = OpenAI(model_name="gpt-4", temperature=0.0)

petstore_agent = planner.create_openapi_agent(petstore_api_spec, requests_wrapper, llm)
user_query = '''Provides a summary of all matches played on 2021-09-01 in json format with value for locale as 'en'. Use api_key=s5qa2vwhwkjghux235hteaxh as a query parameter
             '''
# user_query = '''Get information for the pet with id 10.
#              '''
petstore_agent.run(user_query)

# endpoints = [
#     (route, operation)
#     for route, operations in raw_petstore_api_spec["paths"].items()
#     for operation in operations
#     if operation in ["get", "post"]
# ]
# print(len(endpoints))

