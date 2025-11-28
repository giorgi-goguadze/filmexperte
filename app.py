#%% pakete
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

 
#%%
#%% Parser
class MovieOutput(BaseModel):
    title: str
    director: str
    role_names: list[str] = Field(description="The role name formatted as e.g. 'Chris Adams' should be returned as 'Adams, Chris'")
    actor_names: list[str]
    release_year: int
    revenue: str = Field(description="The worldwide revenue of the film in million US dollars", examples=["150M USD"])
    confidence_score: float = Field(description="This refers to the confidence score of the model in predicting the information. It is a number between 0 and 1.")
 
class MoviesOutput(BaseModel):
    movies: list[MovieOutput]
 
parser = PydanticOutputParser(pydantic_object=MoviesOutput)
# parser.get_format_instructions()
 
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Die bist ein Filmexperte und lieferst Informationen zu Filmen.
    Gib 5 Filme zur Beschreibung in absteigender Reihenfolge. Die Reihenfolge ergibt sich aus dem kommerziellen Erfolg.
    Verwende das vorgegebene Schema {format_instructions}
    """),
    ("user", "Filmbeschreibung: <beschreibung>{beschreibung}</beschreibung>")
]).partial(format_instructions=parser.get_format_instructions())
 
# %% Modellinstanz erstellen
model = ChatGroq(model="openai/gpt-oss-120b", model_kwargs={"seed":42}, temperature=0)
 
#%% Chain erstellen
chain = prompt_template | model | parser
 
 
st.title("Unser Filmexperte")
prompt = st.chat_input("Beschreibe, was du vom Film noch weißt.")
if prompt:
    res_movies = chain.invoke({"beschreibung": prompt})
    for res in res_movies.movies:
        st.markdown(f"***Titel***: {res.title}")
        st.markdown(f"***Regisseur***: {res.director}")
        st.markdown(f"***Rollennamen***: {res.role_names}")
        st.markdown(f"***Schauspielernamen***: {res.actor_names}")
        st.markdown(f"***Erscheinungsjahr***: {res.release_year}")
        st.markdown(f"***Einnahmen [Mio. $]***: {res.revenue}")
        st.markdown("-"*20)
   
   