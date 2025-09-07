from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Setup
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model="ollama/crewai-researcher-summarizer",
    base_url="http://localhost:11434/v1"
)

# Agents
researcher = Agent(
    role="Researcher",
    goal="Gather key insights, data, and references on {topic}",
    backstory="An expert researcher who finds trustworthy, up-to-date information.",
    llm=llm, verbose=True
)

summarizer = Agent(
    role="Summarizer",
    goal="Condense research findings into a short, clear report",
    backstory="A skilled analyst who converts raw research into digestible summaries.",
    llm=llm, verbose=True
)

presenter = Agent(
    role="Presenter",
    goal="Turn summaries into a presentation-style markdown (headings, bullets, highlights)",
    backstory="A communication specialist who formats information for easy consumption.",
    llm=llm, verbose=True
)

# Tasks
research_task = Task(
    description="Collect the latest insights, statistics, and references on {topic}.",
    expected_output="Detailed research notes with sources.",
    agent=researcher
)

summarize_task = Task(
    description="Summarize research notes into a concise, structured report.",
    expected_output="2–3 paragraph summary with key takeaways.",
    agent=summarizer
)

presentation_task = Task(
    description="Format the summary into a markdown presentation with headings and bullet points.",
    expected_output="Presentation-style markdown document.",
    agent=presenter
)

# Crew
crew = Crew(
    agents=[researcher, summarizer, presenter],
    tasks=[research_task, summarize_task, presentation_task],
    verbose=True
)

if __name__ == "__main__":
    topic = input("Enter the topic: ")
    result = crew.kickoff(inputs={"topic": topic})
    print(result)
