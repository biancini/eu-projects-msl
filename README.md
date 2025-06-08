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

   You can also run the application in console mode by executing:

   ```bash
   python main.py
   ```

4. **Query the Documents:**

   Once the setup is complete, you can input natural language queries to retrieve information from the processed documents.

## Testing with OpenAI Evals

This module supports evaluation using the OpenAI Evals framework. Test cases are defined in JSONL format within the test_evals/data/ directory.

### Running the Evaluation

To execute the tests, use the following command from the root directory of your project:

```bash
PYTHONPATH=".:test_evals" oaieval eu_rag euprojects-eval --registry-path test_evals
```

Explanation:
-	PYTHONPATH=".:test_evals": Adds the current directory and test_evals to the Python path, ensuring that custom modules are discoverable.
-	oaieval eu_rag euprojects-eval: Runs the evaluation using the eu_rag completion function and the euprojects-eval evaluation configuration.
-	--registry-path test_evals: Specifies the path to the registry containing evaluation configurations and data.

### Defining Test Cases

Each test case in the tests.jsonl file should be a JSON object with the following structure:
```json
{
  "test_id": 1,
  "input": "how many KPIs do SPECTRO have?",
  "ideal": "SPECTRO has a total of 22 Key Performance Indicators (KPIs).",
  "pages": "Proposal 98-99\nGrant Agreement 81-82"
}
```

Explanation:
-  test_id: A unique identifier for the test case.
-	in input: The prompt or question to be evaluated.
-	ideal: The expected correct response from the model.
-	pages: The pages from which the information is extracted, formatted as "Proposal page numbers\nGrant Agreement page numbers".

Ensure that each test case is on a separate line in the JSONL file.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENCE](LICENCE) file for details.

## Acknowledgments

This tool was developed to streamline the analysis of EU-funded project documentation, enhancing accessibility and efficiency in information retrieval.
