from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .tools_loader import get_tools

@CrewBase
class RadiologyCrew:
    """Multi-agent radiology report generation crew"""

    def __init__(self):
        # Load tools once
        self.tools = get_tools()
        
        # Initialize LLMs with optimal settings
        self.vision_llm = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)
        self.text_llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.1)
        self.draft_llm = LLM(model="groq/deepseek-r1-distill-llama-70b", temperature=0.3)
        self.critic_llm = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)

    @agent
    def vision_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['vision_agent'],
            tools=[self.tools["vision_tool"]],
            llm=self.vision_llm,
            allow_delegation=False,
            verbose=False
        )

    @agent
    def pubmed_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['pubmed_agent'],
            tools=[self.tools["pubmed_tool"]],
            llm=self.text_llm,
            verbose=False
        )

    @agent
    def iu_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['iu_agent'],
            tools=[self.tools["iu_tool"]],
            llm=self.text_llm,
            verbose=False
        )

    @agent
    def draft_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['draft_agent'],
            llm=self.draft_llm,
            verbose=False
        )

    @agent
    def critic_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['critic_agent'],
            llm=self.critic_llm,
            verbose=True
        )

    @task
    def caption_task(self) -> Task:
        return Task(config=self.tasks_config['caption_task'])

    @task
    def pubmed_task(self) -> Task:
        return Task(
            config=self.tasks_config['pubmed_task'],
            context=[self.caption_task()]
        )

    @task
    def iu_task(self) -> Task:
        return Task(config=self.tasks_config['iu_task'])

    @task
    def draft_task(self) -> Task:
        return Task(
            config=self.tasks_config['draft_task'],
            context=[self.caption_task(), self.iu_task(), self.pubmed_task()]
        )

    @task
    def critic_task(self) -> Task:
        return Task(
            config=self.tasks_config['critic_task'],
            context=[self.caption_task(), self.iu_task(), self.draft_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False
        )