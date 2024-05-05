# Financial Analysis App
## Overview
The Financial Analysis App is a web application built to analyze financial data of companies retrieved from the SEC EDGAR database using ```sec-api```. It provides functionalities to fetch, analyze, and visualize financial statements, enabling users to gain insights into a company's performance over time.

## Features
1. **Company Overview:** This option provides a concise summary of the company based on its ticker name. It includes details such as founding information, headquarters location, key products, revenue sources, competitors, market capitalization, CEO, focus areas, and future plans.
- This overview is beneficial for users who want a quick understanding of the company's background and current position in the market.
  
2. **Product-Based Revenue Insights:** This feature analyzes the company's revenue breakdown by product categories over time. By visualizing revenue trends and distributions across different product lines, users can gain insights into the performance of individual products and identify which ones contribute most significantly to the company's overall revenue.
- This information is valuable for strategic decision-making, product planning, and assessing market demand.

3. **Region-Based Revenue Insights:** This option examines the company's revenue distribution across geographical regions or markets. By visualizing revenue trends by region, users can assess the company's global presence, identify growth opportunities in specific regions, and evaluate the impact of regional factors on revenue performance.
- Understanding regional revenue dynamics can help businesses optimize their market expansion strategies, allocate resources effectively, and mitigate geographical risks.
## How to Run the Application
Access the application in your web browser at https://financial-analysis-app-dkshjn.streamlit.app/

**OR**

1. Install the required Python dependencies listed in requirements.txt using pip install -r requirements.txt.
2. Run the Streamlit application by executing streamlit run app.py in the terminal.
3. Access the application in your web browser at http://localhost:8501.

Check out the app demo video [here](https://drive.google.com/file/d/16YK9DCXpC6NrzuN_YA2jI8jj39HhrTO8/view?usp=sharing).
## Tech Stack
- **Python:** Used as the primary programming language for backend development due to its simplicity, extensive libraries, and ecosystem support for financial analysis tasks.
- **Streamlit:** Used for building the web application's user interface due to its ease of use, rapid prototyping capabilities, and ability to turn data scripts into interactive web apps.


## Dependencies
- **sec-api:** The ```sec-api``` package is a Python library that provides access to the U.S. Securities and Exchange Commission (SEC) Electronic Data Gathering, Analysis, and Retrieval (EDGAR) database. It allows users to programmatically query and retrieve various types of financial data, including filings, reports, and disclosures submitted by publicly traded companies. For more details visit [sec-api](https://sec-api.io/).
- **Pandas:** Leveraged for data manipulation and analysis tasks, as it provides powerful data structures and functions to work with structured data, making it suitable for handling financial datasets.
- **Matplotlib:** Employed for data visualization purposes, as it is a widely-used plotting library in Python that offers flexibility and customization options for creating various types of charts and graphs.
- **DSPy:** DSPy is a Python library that facilitates interaction various LLMs. It offers a streamlined interface for accessing and utilizing these powerful language models for various natural language processing tasks. For more details, checkout the [DSPy github repository](https://github.com/stanfordnlp/dspy).
- **Ollama**: The LLM (Large Language Model) utilized in the Financial Analysis App is Llama2, a variant of Ollama, implemented through DSPy. Llama2 is an instance of Ollama, a model designed for generating human-like text based on input prompts. For more details on the LLM model Llama2, please visit the [Ollama website](https://ollama.com/library/llama2).


