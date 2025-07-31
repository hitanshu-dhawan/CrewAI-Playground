#!/usr/bin/env python3
"""
CrewAI Podcast Script Writer

This script demonstrates a two-agent system for creating podcast scripts:
1. Research Agent - Gathers information from the web using DuckDuckGo
2. Podcast Script Agent - Converts research into an engaging, humorous podcast script

The agents work sequentially to produce a complete podcast script from a given topic.
"""

import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# Create a custom search tool using the @tool decorator
@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web using DuckDuckGo for the given query."""
    search_engine = DuckDuckGoSearchRun()
    return search_engine.run(query)


def setup_llm():
    """
    Initialize and configure the language model.
    
    Returns:
        LLM: Configured language model instance (Gemini or OpenAI)
    
    Raises:
        ValueError: If no API key is found for either service
    """
    # Try Gemini first
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if gemini_api_key:
        print("🔧 Using Gemini LLM...")
        return LLM(
            model="gemini/gemini-1.5-flash",
            api_key=gemini_api_key,
            temperature=0.7  # Higher temperature for more creative output
        )
    
    # If no API keys are found
    raise ValueError(
        "No API key found. Please set either GOOGLE_API_KEY or OPENAI_API_KEY "
        "environment variable."
    )


def create_research_agent(llm):
    """
    Create a research agent that gathers information from the web.
    
    Args:
        llm (LLM): Configured language model instance
    
    Returns:
        Agent: Research agent with DuckDuckGo search capabilities
    """
    agent = Agent(
        role='Research Specialist',
        goal='Gather comprehensive and accurate information about the given topic from web sources',
        backstory="""
        You are an experienced research specialist with expertise in finding reliable 
        information quickly and efficiently. You excel at synthesizing multiple sources 
        into coherent summaries that capture the most important and interesting aspects 
        of any topic. Your research forms the foundation for engaging content creation.
        """,
        tools=[duckduckgo_search],  # Use the properly wrapped tool
        verbose=True,               # Enable detailed output
        allow_delegation=False,     # Prevent delegation to other agents
        llm=llm                     # Assign the configured LLM
    )
    
    return agent


def create_podcast_script_agent(llm):
    """
    Create a podcast script agent that converts research into engaging scripts.
    
    Args:
        llm (LLM): Configured language model instance
    
    Returns:
        Agent: Podcast script writer agent
    """
    agent = Agent(
        role='Podcast Script Writer',
        goal='Transform research content into an engaging, humorous, and conversational podcast script',
        backstory="""
        You are a talented podcast script writer known for your wit, humor, and ability 
        to make any topic entertaining. You excel at creating conversational content that 
        feels like a friend telling you an interesting story over coffee. Your scripts 
        are filled with personality, occasional jokes, fun observations, and engaging 
        storytelling techniques that keep listeners hooked from start to finish.
        """,
        verbose=True,               # Enable detailed output
        allow_delegation=False,     # Prevent delegation to other agents
        llm=llm                     # Assign the configured LLM
    )
    
    return agent


def create_research_task(research_agent, topic):
    """
    Create a research task for gathering information about the topic.
    
    Args:
        research_agent (Agent): The research agent
        topic (str): The topic to research
    
    Returns:
        Task: Research task configuration
    """
    task = Task(
        description=f"""
        Research the topic "{topic}" thoroughly using web search capabilities.
        
        Your research should include:
        1. Key facts and background information
        2. Interesting stories, anecdotes, or case studies
        3. Recent developments or news
        4. Fun facts or surprising details
        5. Different perspectives or controversies (if any)
        
        Compile your findings into a comprehensive research summary that provides 
        rich material for creating an engaging podcast script.
        
        Topic to research: {topic}
        """,
        expected_output="""
        A well-structured research summary containing:
        - Key facts and background information
        - Interesting stories and anecdotes
        - Recent developments
        - Fun facts and surprising details
        - Multiple perspectives on the topic
        """,
        agent=research_agent
    )
    
    return task


def create_script_writing_task(script_agent, research_task):
    """
    Create a script writing task that depends on the research task.
    
    Args:
        script_agent (Agent): The podcast script writer agent
        research_task (Task): The completed research task
    
    Returns:
        Task: Script writing task configuration
    """
    task = Task(
        description="""
        Using the research provided, create an engaging and humorous podcast script.
        
        The script should:
        1. Start with an attention-grabbing hook
        2. Present information in a conversational, storytelling format
        3. Include humor, wit, and entertaining commentary
        4. Use a friendly, accessible tone (like talking to a friend)
        5. Add transitions and personality throughout
        6. Include occasional jokes, observations, or funny comparisons
        7. End with a memorable conclusion
        
        Style guidelines:
        - Write as if you're a charismatic podcast host
        - Use "we", "you", and conversational language
        - Add parenthetical stage directions for emphasis: (dramatic pause), (chuckles), etc.
        - Include rhetorical questions to engage the audience
        - Make it feel spontaneous and natural, not scripted
        
        Length: Aim for a 5-10 minute podcast segment (approximately 750-1500 words).
        """,
        expected_output="""
        A complete podcast script formatted with:
        - Clear intro hook
        - Conversational narrative flow
        - Humorous commentary and observations  
        - Engaging storytelling elements
        - Natural transitions between topics
        - Memorable conclusion
        - Stage directions in parentheses
        """,
        agent=script_agent,
        context=[research_task]  # This task depends on the research task output
    )
    
    return task


def create_podcast_script(topic):
    """
    Orchestrate the creation of a podcast script for the given topic.
    
    Args:
        topic (str): The topic for the podcast script
    
    Returns:
        str: The completed podcast script
    
    Raises:
        Exception: If any error occurs during the script creation process
    """
    try:
        # Step 1: Initialize the language model
        print("🔧 Setting up language model...")
        llm = setup_llm()
        
        # Step 2: Create the research agent with search capabilities
        print("🔍 Creating Research Specialist agent...")
        research_agent = create_research_agent(llm)
        
        # Step 3: Create the podcast script writer agent
        print("🎙️ Creating Podcast Script Writer agent...")
        script_agent = create_podcast_script_agent(llm)
        
        # Step 4: Create the research task
        print("📋 Setting up research task...")
        research_task = create_research_task(research_agent, topic)
        
        # Step 5: Create the script writing task (depends on research)
        print("📝 Setting up script writing task...")
        script_task = create_script_writing_task(script_agent, research_task)
        
        # Step 6: Create and configure the crew
        print("👥 Assembling the podcast production crew...")
        crew = Crew(
            agents=[research_agent, script_agent],  # Both agents in the crew
            tasks=[research_task, script_task],     # Sequential tasks
            verbose=True                            # Enable detailed execution logging
        )
        
        # Step 7: Execute the podcast script creation process
        print(f"🚀 Starting podcast script creation for topic: '{topic}'...")
        print("=" * 80)
        
        result = crew.kickoff()
        
        return result
        
    except Exception as error:
        print(f"❌ Error during script creation: {str(error)}")
        raise


def main():
    """
    Main execution function - creates a podcast script for a user-specified topic.
    """
    print("=" * 80)
    print("🎙️ CrewAI Podcast Script Writer")
    print("=" * 80)
    print("Welcome to the AI-powered podcast script generator!")
    print("This tool will research any topic and create an engaging podcast script.")
    print("-" * 80)
    
    # Get topic from user input
    topic = input("\n📝 Enter the topic for your podcast script: ").strip()
    
    if not topic:
        print("❌ No topic provided. Please run the script again with a valid topic.")
        return
    
    print(f"\n🎯 Topic selected: '{topic}'")
    print("🔄 Starting the script creation process...")
    print("=" * 80)
    
    try:
        # Create the podcast script
        script = create_podcast_script(topic)
        
        # Display the final result
        print("\n" + "=" * 80)
        print("🎉 PODCAST SCRIPT COMPLETE!")
        print("=" * 80)
        print("\n" + "🎙️" * 5 + " FINAL PODCAST SCRIPT " + "🎙️" * 5)
        print("-" * 80)
        print(script)
        print("-" * 80)
        print("🎉 Script generation complete! Ready for recording!")
        
    except Exception as error:
        print(f"\n❌ Script creation failed: {str(error)}")


if __name__ == "__main__":
    main()
