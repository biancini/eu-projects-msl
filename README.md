# EU Projects MSL

## Overview

**EU Projects MSL** is a Python-based application that processes PDF documents pertaining to EU-funded projects. It extracts textual content from these PDFs and facilitates querying the information through natural language inputs. This tool is particularly beneficial for researchers, project managers, and policymakers who need to analyze and retrieve specific information from extensive EU project documentation.

## Features

- **PDF Text Extraction:** Efficiently parses and extracts text from PDF files.
- **Natural Language Querying:** Allows users to input queries in natural language to retrieve relevant information from the processed documents.
- **Modular Architecture:** Organized codebase with separate modules for PDF reading, data modeling, configuration, and query handling.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/biancini/eu-projects-msl.git
   cd eu-projects-msl
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your PDF Files:**

   Place all the PDF documents you wish to process in a designated directory.

2. **Configure the Application:**

   Review and modify `confs.py` to set the appropriate paths and configurations for your environment.

3. **Run the Application:**

   ```bash
   python app.py
   ```

   The application will process the PDFs, extract their content, and initialize the query interface.

4. **Query the Documents:**

   Once the setup is complete, you can input natural language queries to retrieve information from the processed documents.

## Project Structure

- `app.py`: Streamlit UI for interacting with the bot.
- `readpdfs.py`: Handles the extraction of text from PDF files.
- `chain.py`: Manages the processing chain for querying.
- `datamodels.py`: Defines data models used within the application.
- `confs.py`: Contains configuration settings.
- `requirements.txt`: Lists all Python dependencies.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This tool was developed to streamline the analysis of EU-funded project documentation, enhancing accessibility and efficiency in information retrieval.
