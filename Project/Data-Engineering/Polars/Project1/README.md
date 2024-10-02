# Cari Dokter

Cari Dokter is a voice-based application that helps users find suitable doctors in Indonesia. The project combines web scraping, data processing, and voice interaction to provide a seamless experience for users seeking medical professionals.

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project consists of two main components:

1. **Data Engineering Pipeline**
   - Scrapes data from hospital websites in Indonesia
   - Manually preprocesses the scraped data
   - Uses Airflow DAGs for daily updates to a PostgreSQL database

2. **Voice-Based User Interface**
   - Utilizes speech-to-text and text-to-speech functionality
   - Implements natural language processing with LangChain and OpenAI
   - Provides a user-friendly interface using Streamlit

## Tech Stack

### Data Engineering
- Apache Airflow
- PostgreSQL
- Polars
- Web scraping: Beautiful Soup (bs4) and Selenium

### User Interface
- Streamlit
- LangChain
- OpenAI
- Speech-to-text: SpeechRecognition library
- Text-to-speech: gTTS (Google Text-to-Speech)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FTDS-assignment-bay/p2-final-project-ftds-019-hck-group-002.git
   cd into the directory
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, use the following command in your terminal:

```bash
streamlit run huggingface.py
```

## Data Pipeline

The Airflow DAG system follows these steps:

1. **Check Folder**: The DAG checks a specified folder for new CSV files.
2. **Validate Columns**: It verifies if the column names in the CSV files match the expected schema.
3. **Transform Data**: If the columns match, the DAG performs necessary transformations on the data.
4. **Database Update**: The processed data is then appended to the PostgreSQL database.

This pipeline ensures that our doctor data is regularly updated with the latest information from various sources.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact


- Alif Nurhadi =  https://www.linkedin.com/in/alifnurhadi/
- Ghaffar Farros 
- Clara Linggaputri 

