#!/usr/bin/env python3
"""
CrewAI Conflict Detection Example

This script demonstrates how to use CrewAI with Google's Gemini LLM
to detect conflicting information in text. The agent analyzes text
and returns either 'conflict' or 'no conflict' based on its findings.
"""

import os
from crewai import Agent, Task, Crew, LLM


def setup_llm():
    """
    Initialize and configure the Gemini LLM.
    
    Returns:
        LLM: Configured Gemini language model instance
    
    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
    """
    # Retrieve Google API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # Validate that API key is available
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Please set it with your Google API key."
        )
    
    # Configure and return the Gemini LLM with optimal settings for conflict detection
    llm = LLM(
        model="gemini/gemini-1.5-flash",  # Use Gemini 1.5 Flash for fast, accurate responses
        api_key=api_key,                  # Authentication key
        temperature=0.1                   # Low temperature for consistent, focused responses
    )
    
    return llm


def create_conflict_detection_agent(llm):
    """
    Create a specialized agent for detecting conflicts in text.
    
    Args:
        llm (LLM): Configured language model instance
    
    Returns:
        Agent: CrewAI agent specialized in critical thinking and conflict detection
    """
    agent = Agent(
        role='Critical Thinker',
        goal='Analyse the text and identify if any conflicting information within',
        backstory="""
        You are an expert critical thinker with exceptional analytical skills.
        Your specialty is identifying contradictions, inconsistencies, and conflicting 
        statements within text. You have years of experience in logic, reasoning, 
        and fact-checking. You approach every text with a methodical mindset, 
        carefully examining each statement for potential conflicts with other 
        statements in the same text.
        """,
        verbose=True,           # Enable detailed output for debugging
        allow_delegation=False, # Prevent delegation to other agents
        llm=llm                 # Assign the configured LLM
    )
    
    return agent


def create_analysis_task(agent, text_to_analyze):
    """
    Create a task for the agent to analyze text for conflicts.
    
    Args:
        agent (Agent): The conflict detection agent
        text_to_analyze (str): The text to be analyzed for conflicts
    
    Returns:
        Task: CrewAI task configured for conflict detection
    """
    task = Task(
        description=f"""Find if there are any conflicting statement / information in text.
        
        Text to analyze:
        {text_to_analyze}
        
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
    
    return task


def analyze_text_for_conflicts(text):
    """
    Main function to analyze text for conflicting information.
    
    Args:
        text (str): The text to be analyzed
    
    Returns:
        str: Analysis result from the CrewAI crew execution
    
    Raises:
        Exception: If any error occurs during the analysis process
    """
    try:
        # Step 1: Initialize the language model
        print("🔧 Setting up Gemini LLM...")
        llm = setup_llm()
        
        # Step 2: Create the specialized conflict detection agent
        print("🤖 Creating Critical Thinker agent...")
        agent = create_conflict_detection_agent(llm)
        
        # Step 3: Create the analysis task
        print("📋 Setting up conflict detection task...")
        task = create_analysis_task(agent, text)
        
        # Step 4: Create and configure the crew
        print("👥 Assembling crew...")
        crew = Crew(
            agents=[agent],     # Single agent crew
            tasks=[task],       # Single task to execute
            verbose=True        # Enable detailed execution logging
        )
        
        # Step 5: Execute the analysis
        print("🚀 Starting conflict analysis...")
        result = crew.kickoff()
        
        return result
        
    except Exception as error:
        print(f"❌ Error during analysis: {str(error)}")
        raise


def main():
    """
    Main execution function - demonstrates conflict detection on sample text.
    """
    print("=" * 60)
    print("🔍 CrewAI Conflict Detection Example")
    print("=" * 60)
    
    # Sample text that contains a potential conceptual conflict
    sample_text = "It's a sunny day, let me take an umbrella to the office."  # Example with conflict
    
    print(f"\n📝 Analyzing text: '{sample_text}'")
    print("-" * 60)
    
    try:
        # Perform the conflict analysis
        result = analyze_text_for_conflicts(sample_text)
        
        # Display the final result
        print("\n" + "=" * 60)
        print(f"🎯 Final Result: {result}")
        print("=" * 60)
        
    except Exception as error:
        print(f"\n❌ Analysis failed: {str(error)}")
        print("Please check your GOOGLE_API_KEY and try again.")


if __name__ == "__main__":
    main()
