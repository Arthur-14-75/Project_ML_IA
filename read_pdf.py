import PyPDF2
import pandas as pd

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the number of pages
        num_pages = len(pdf_reader.pages)
        print(f"Number of pages: {num_pages}")
        
        # Read all pages
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text

# Read the project PDF
project_text = read_pdf("ML and AI project.pdf")
print("\nProject PDF content:")
print(project_text[:1000])  # Print first 1000 characters

# Read the extra features PDF
extra_features_text = read_pdf("project_extra_features.pdf")
print("\nExtra Features PDF content:")
print(extra_features_text[:1000])  # Print first 1000 characters

# Read the Excel file with class labels
labels_df = pd.read_excel('train/classif.xlsx')

# Count the number of each type
print('Number of each insect type in the dataset:')
print(labels_df['bug type'].value_counts()) 