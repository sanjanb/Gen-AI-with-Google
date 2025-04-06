# Prompt Engineering Overview

## Table of Contents
1. [Introduction](#introduction)
2. [Prompt Engineering](#prompt-engineering)
3. [LLM Output Configuration](#llm-output-configuration)
4. [Prompting Techniques](#prompting-techniques)
5. [Best Practices](#best-practices)
6. [Summary](#summary)
7. [Endnotes](#endnotes)
8. [Acknowledgements](#acknowledgements)

## Introduction
A text prompt is the input a large language model (LLM) uses to predict a specific output. Crafting effective prompts is crucial for obtaining accurate and meaningful responses. This whitepaper focuses on writing prompts for the Gemini model within Vertex AI, exploring various techniques, best practices, and challenges in prompt engineering.

## Prompt Engineering
Prompt engineering involves designing high-quality prompts to guide LLMs in producing accurate outputs. It is an iterative process that considers the model's training data, configurations, word choice, style, tone, structure, and context. Effective prompt engineering optimizes prompt length and evaluates writing style and structure in relation to the task.

### Key Aspects
- **LLM as a Prediction Engine**: LLMs predict the next token based on previous tokens and training data.
- **Prompt Types**: Prompts can be used for text summarization, information extraction, question answering, text classification, language or code translation, code generation, and code documentation or reasoning.
- **Model Selection**: Prompts may need optimization for specific models like Gemini, GPT, Claude, Gemma, or LLaMA.

## LLM Output Configuration
Configuring LLM outputs involves setting parameters like output length, sampling controls, temperature, top-K, and top-P to optimize responses for specific tasks.

### Output Length
- **Token Generation**: Controls the number of tokens generated in a response. More tokens mean higher computation, slower response times, and higher costs.
- **Output Length Restriction**: Important for techniques like ReAct to prevent useless token emission after the desired response.

### Sampling Controls
- **Temperature**: Controls the degree of randomness in token selection. Lower temperatures are more deterministic, while higher temperatures lead to more diverse outputs.
- **Top-K and Top-P**: Restrict the next token to come from the top predicted probabilities, controlling randomness and diversity.

### Putting It All Together
- **Configuration Settings**: Choosing between temperature, top-K, top-P, and the number of tokens depends on the specific application and desired outcome. Experimentation is key to finding the right balance.

## Prompting Techniques
Various prompting techniques leverage how LLMs are trained and work to obtain relevant results.

### General Prompting / Zero-Shot
- **Zero-Shot Prompt**: Provides only a task description and some starting text. Useful for tasks like classifying movie reviews.

### One-Shot & Few-Shot
- **One-Shot Prompt**: Provides a single example for the model to imitate.
- **Few-Shot Prompt**: Provides multiple examples to show the model a pattern to follow. Useful for guiding the model to a specific output structure or pattern.

### System, Contextual, and Role Prompting
- **System Prompting**: Sets the overall context and purpose, defining the model's fundamental capabilities and overarching purpose.
- **Contextual Prompting**: Provides specific details or background information relevant to the current conversation or task.
- **Role Prompting**: Assigns a specific character or identity for the model to adopt, generating responses consistent with that role's knowledge and behavior.

### Step-Back Prompting
- **Step-Back Prompting**: Encourages the LLM to consider a general question related to the specific task, activating relevant background knowledge and reasoning.

### Chain of Thought (CoT)
- **CoT Prompting**: Generates intermediate reasoning steps, leading to more accurate answers for complex tasks. Can be combined with few-shot prompting for better results.

### Self-Consistency
- **Self-Consistency**: Combines sampling and majority voting to generate diverse reasoning paths and select the most consistent answer, improving accuracy and coherence.

### Tree of Thoughts (ToT)
- **ToT**: Allows LLMs to explore multiple reasoning paths simultaneously, maintaining a tree of thoughts where each thought is an intermediate step.

### ReAct (Reason & Act)
- **ReAct Prompting**: Enables LLMs to solve complex tasks using natural language reasoning combined with external tools, mimicking human behavior by combining verbal reasoning and taking actions.

### Automatic Prompt Engineering
- **Automatic Prompt Engineering**: Automates prompt creation by prompting a model to generate more prompts, evaluating them, altering good ones, and repeating.

### Code Prompting
- **Code Prompting**: Involves writing prompts for returning code, explaining code, translating code, and debugging and reviewing code.

## Best Practices
Effective prompt engineering requires experimentation and adherence to best practices.

### Key Best Practices
- **Provide Examples**: Include one-shot or few-shot examples within a prompt to act as a teaching tool.
- **Design with Simplicity**: Prompts should be concise, clear, and easy to understand.
- **Be Specific About the Output**: Provide specific details in the prompt to help the model focus on what’s relevant.
- **Use Instructions Over Constraints**: Instructions are generally more effective than constraints.
- **Control the Max Token Length**: Set a max token limit in the configuration or explicitly request a specific length in your prompt.
- **Use Variables in Prompts**: Make prompts more dynamic by using variables that can be changed for different inputs.
- **Experiment with Input Formats and Writing Styles**: Different models, configurations, prompt formats, word choices, and submits can yield different results.
- **Mix Up Classes for Classification Tasks**: For few-shot prompting with classification tasks, mix up the possible response classes in the examples.
- **Adapt to Model Updates**: Stay informed about model architecture changes and adjust prompts to leverage new features.
- **Experiment with Output Formats**: Try having the output returned in a structured format like JSON or XML for non-creative tasks.
- **Document Prompt Attempts**: Document your prompt attempts in full detail to learn over time what works.

## Summary
This whitepaper discussed prompt engineering and various prompting techniques, including zero-shot, few-shot, system, role, contextual, step-back, chain of thought, self-consistency, Tree of Thoughts, and ReAct. It also explored ways to automate prompts and concluded with best practices for becoming a better prompt engineer.

## Endnotes
Provides a list of references and sources used in the whitepaper, including links to Google resources, research papers on various prompting techniques, and Google Cloud Platform GitHub repositories with examples.

## Acknowledgements
Lists the content contributors, curators and editors, technical writer, and designer who contributed to the whitepaper.

---

This structured report provides a comprehensive overview of prompt engineering, highlighting key techniques, best practices, and considerations for optimizing LLM outputs.
