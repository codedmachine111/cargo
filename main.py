import os
from process_input import processInput

def main():
    file_path = input("Enter the path to your PDF or ZIP file: ")

    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    with open(file_path, 'rb') as file:
        text_chunks = processInput(file)

    if text_chunks:
        print("Processing done successfully!")
        print(text_chunks)
    else:
        print("No data extracted from the file.")

if __name__ == '__main__':
    main()
