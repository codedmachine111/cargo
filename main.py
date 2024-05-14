import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from process_input import processInput

# SIDEBAR
with st.sidebar:
    st.title('Abridge')
    st.markdown('''
        ## About
        Chat with your all your pdf files with ease!
    ''')
    add_vertical_space(5)
    

def main():
    st.header("Drop your PDF files here")

    # Take input files
    uploaded_file = st.file_uploader(type=['pdf', 'zip'], label="Click below or drag and drop your files to upload!")

    # Process the input files accordingly
    if(uploaded_file):   
        # Extract text and tables
        with st.status("Please wait as we processing your input files...", expanded=True) as status:
            st.write("Extracting data from Documents...")
            text_chunks = processInput(uploaded_file)
            status.update(label="Processing done successfully!", state="complete", expanded=False)

        st.write(text_chunks)
    else:
        st.write('''No pdf uploaded''')

if __name__ == '__main__':
    main()