from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Setup API (replace with your actual key if needed)
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model="ollama/mistral",
    base_url="http://localhost:11434/v1"
)

# Agents
planner = Agent(
    role="Planner",
    goal="Create an outline for a blog on {topic}",
    backstory="You prepare structured outlines and research for Medium blogs.",
    llm=llm, verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a blog post on {topic} based on the planner’s outline",
    backstory="You turn the outline into a polished, factual, and engaging blog post.",
    llm=llm, verbose=True
)

editor = Agent(
    role="Editor",
    goal="Polish and proofread the blog post for Medium’s style",
    backstory="You ensure clarity, grammar, tone, and overall readability.",
    llm=llm, verbose=True
)

# Tasks
plan = Task(
    description="Create a blog outline on {topic} with intro, key points, conclusion, SEO keywords.",
    expected_output="Structured content plan with outline, keywords, and sources.",
    agent=planner
)

write = Task(
    description="Write a blog post on {topic} using the planner’s outline.",
    expected_output="Markdown blog post, structured with sections and 2–3 paragraphs each.",
    agent=writer
)

edit = Task(
    description="Proofread and refine the blog for grammar, tone, and readability.",
    expected_output="Final blog post in markdown, ready to publish.",
    agent=editor
)

# Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

if __name__ == "__main__":
    topic = input("Enter the blog topic: ")
    result = crew.kickoff(inputs={"topic": topic})
    print(result)
