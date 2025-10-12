# AI Rules and Tech Stack Guidelines

This document outlines the core technologies and libraries used in the AlpacaTradingAgent application, along with clear rules for their usage to ensure consistency, maintainability, and best practices.

## Tech Stack Overview

The AlpacaTradingAgent is a sophisticated multi-agent financial trading framework built on a diverse set of technologies:

*   **Frontend:** A responsive web interface built with Dash (Python), styled using Tailwind CSS, and leveraging `dash-bootstrap-components` for UI elements and Plotly for interactive charts.
*   **Agent Framework:** The core multi-agent system is orchestrated using Python, with `LangChain` and `LangGraph` providing the foundational framework for agent creation, communication, and workflow management.
*   **Data & APIs:** Extensive integration with various external APIs for real-time financial and news data, including OpenAI, Alpaca, Finnhub, FRED, DeFi Llama, CryptoCompare, Reddit, and Google News.
*   **Data Processing:** `Pandas` is the standard for all data manipulation, `Stockstats` for calculating technical indicators, and `ChromaDB` for managing agent memories and reflections.
*   **LLM Integration:** Utilizes OpenAI's language models for advanced analysis, reasoning, and debate among agents.
*   **CLI & UI:** A robust command-line interface (CLI) is developed using `Typer`, `Rich`, and `Questionary` for interactive user input and rich output, complementing the web UI.
*   **Styling & Components:** Tailwind CSS is the primary utility-first CSS framework, enhanced by pre-built components from `shadcn/ui` and `Radix UI` for accessibility and consistency.
*   **Icons:** `lucide-react` is used for all iconography across the application.
*   **Containerization:** The application is designed for deployment using Docker, ensuring portability and consistent environments.

## Library Usage Rules

To maintain a clean, efficient, and scalable codebase, adhere to the following library usage rules:

*   **Frontend Styling:**
    *   Always use **Tailwind CSS** for styling components. Prioritize utility classes for layout, spacing, colors, and other design aspects.
    *   For accessible and pre-styled UI components, use **shadcn/ui** and **Radix UI**. Do **NOT** modify the source files of `shadcn/ui` components directly; create new components if customization is required.
*   **Icons:** All icons throughout the application must be sourced from **lucide-react**.
*   **Routing:** For client-side navigation, **React Router** should be used, with all primary routes defined within `src/App.tsx`.
*   **Agent Orchestration:** The multi-agent system's core logic, including agent definitions, chains, and graphs, must be built using **LangChain** and **LangGraph**.
*   **LLM Interaction:** All interactions with OpenAI's language models should be performed using the **langchain-openai** library or the native **OpenAI Python client**.
*   **Financial Market Data & Trading:**
    *   Use **alpaca-py** (via `AlpacaUtils`) for fetching real-time and historical stock and cryptocurrency market data, as well as for executing trading orders.
    *   For stock-specific news, insider sentiment, and fundamental data, utilize **finnhub-python** (via `FinnhubUtils`).
    *   General web scraping and API calls for economic data (FRED), DeFi metrics (DeFi Llama), crypto news (CryptoCompare), social sentiment (Reddit), and global news (Google News) should use the **requests** library, often wrapped in utility functions.
    *   While `yfinance` is available, `alpaca-py` is the preferred source for market data due to its direct trading integration.
*   **Data Manipulation:** **Pandas** is the mandatory library for all data manipulation, analysis, and tabular data handling.
*   **Technical Indicators:** **Stockstats** is the designated library for calculating all technical indicators on financial time-series data.
*   **Vector Database/Memory:** **ChromaDB** is used for implementing the agent's long-term memory, storing and retrieving past financial situations and recommendations.
*   **CLI Development:**
    *   **Typer** is the framework for building command-line interfaces.
    *   **Rich** is used for creating rich text, tables, progress bars, and overall enhanced terminal output.
    *   **Questionary** is used for interactive command-line prompts and selections.
*   **Web UI Framework:** The web user interface is built using **Dash** (Python), with **dash-bootstrap-components** for Bootstrap-themed UI elements and **Plotly** for all interactive charting.
*   **Environment Variables:** **python-dotenv** is used to manage and load environment variables (e.g., API keys, secrets) from `.env` files.