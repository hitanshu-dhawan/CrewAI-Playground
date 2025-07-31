#!/usr/bin/env python3
"""
CrewAI Example: Conflict Detection in Text using Gemini API
"""

import os
from crewai import Agent, Task, Crew, LLM


def main():
    
    api_key = os.getenv('GOOGLE_API_KEY')

    input_text = "I'm going to office to rest."

    llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=api_key,
        temperature=0.1
    )

    agent = Agent(
        role='Critical Thinker',
        goal='Analyse the text and identify if any conflicting information within',
        backstory="""You are an expert critical thinker with exceptional analytical skills.
        Your specialty is identifying contradictions, inconsistencies, and conflicting 
        statements within text. You have years of experience in logic, reasoning, 
        and fact-checking. You approach every text with a methodical mindset, 
        carefully examining each statement for potential conflicts with other 
        statements in the same text.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=f"""Find if there are any conflicting statement / information in text.
        
        Text to analyze:
        {input_text}
        
        Instructions:
        1. Read the entire text carefully
        2. Identify all factual statements and claims
        3. Look for contradictions, inconsistencies, or conflicting information
        4. Determine if any statements contradict each other
        5. Provide your final answer as either 'conflict' or 'no conflict'
        """,
        expected_output="Respond with 'conflict' / 'no conflict'",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    # Execute the task
    result = crew.kickoff()

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
