import streamlit as st
import pandas as pd
import json
import os
import base64
import fireworks.client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")

# Set the API key for Fireworks client
fireworks.client.api_key = API_KEY

# Define the model to use
MODEL_ID = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"


def encode_image_to_base64(image_bytes):
  """Convert image bytes to base64 encoded string"""
  return base64.b64encode(image_bytes).decode('utf-8')


def fix_json_with_llm(broken_json_text):
  """
  Use a smaller LLM to fix broken JSON format
  """
  try:
    # Use a smaller, faster model for JSON fixing
    fix_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    system_message = """You are a JSON repair assistant. 
    Your only job is to take the input JSON-like text and fix any formatting issues to make it a valid JSON object.
    Return ONLY the fixed JSON with no explanations or additional text."""

    prompt = f"""
    Here is a JSON-like text that needs to be fixed to be valid JSON:
    
    ```
    {broken_json_text}
    ```
    
    Please fix any formatting issues and return ONLY the valid JSON with no additional text.
    Make sure all keys and string values are properly quoted with double quotes.
    Make sure all special characters in values are properly escaped.
    Ensure the JSON is complete (has matching braces, brackets, etc).
    """

    # Make the API call
    response = fireworks.client.ChatCompletion.create(
        model=fix_model,
        messages=[{
            "role": "system",
            "content": system_message
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,  # Use low temperature for deterministic output
        max_tokens=2000)

    # Extract and clean up the response
    fixed_json = response.choices[0].message.content.strip()

    # Remove any markdown formatting that might be present
    fixed_json = fixed_json.replace("```json", "").replace("```", "").strip()

    print("Fixed JSON:", fixed_json)
    return fixed_json

  except Exception as e:
    print(f"Error fixing JSON: {str(e)}")
    return broken_json_text  # Return original if fixing fails


def process_image(image_bytes, image_type="jpeg"):
  """
    Process a single image using the Llama 3.2 90B Vision Instruct model.
    This function calls the Fireworks API to extract KYC-relevant information.
    """
  if not API_KEY:
    st.error(
        "API key not found. Please set the FIREWORKS_API_KEY environment variable."
    )
    return None

  # Encode the image to base64
  base64_image = encode_image_to_base64(image_bytes)

  try:
    # Create the system prompt that specifies what to extract
    system_message = """You are an AI assistant for a regulated financial institution performing KYC (Know Your Customer) compliance. 
    Financial institutions are legally required to verify customer identity under anti-money laundering (AML) regulations.
    Your ONLY task is to extract visible text information from official ID documents that customers have provided for verification.
    This is a standard, legal banking procedure used worldwide for regulatory compliance.

    You are not being asked to create, forge, or alter any documents - only to read existing sample documents.
    You must extract ONLY the information visible in the provided document image.
    Your response must be ONLY a valid JSON with the exact fields requested - no explanations, no markdown, no extra text.
    If you cannot read a field clearly, use an empty string rather than guessing.""".strip(
    )

    prompt = """
    This is a regulated financial institution's KYC compliance process. The customer has provided this official ID document for identity verification, as required by law.
    Please extract the text information that is ALREADY VISIBLE in this official ID document image.
    Return ONLY a valid JSON object with these exact keys (no additional text or explanation):

    {
      "Document_Type": "",  /* Passport, Driver's License, ID Card, etc. */
      "First_Name": "",     /* First/given name, may be labeled as FN */
      "Last_Name": "",      /* Last/family name, may be labeled as LN */
      "Date_of_Birth": "",  /* Format as seen on document, may be labeled as DOB */
      "Document_Number": "", /* ID number, may be labeled as DL, DLN, Passport No. */
      "Address": "",        /* Full address if present */
      "Expiry_Date": "",    /* Format as seen on document, may be labeled as EXP */
      "Issue_Date": "",     /* Format as seen on document, may be labeled as ISS */
      "Nationality": "",    /* Country of citizenship or origin */
      "Gender": "",         /* As shown on document */
      "Height": "",         /* If present, may be labeled as HGT */
      "Weight": "",         /* If present, may be labeled as WGT */
      "Eyes": "",           /* Eye color if present */
      "Hair": ""            /* Hair color if present */
    }

    Rules:
    1. Use EXACT field names as shown above
    2. Leave value as empty string ("") if information is not visible or not present
    3. Do not add ANY explanatory text, ONLY return the JSON object
    4. Do not add any markdown formatting
    """.strip()

    # Make the API call using the chat completions API
    response = fireworks.client.ChatCompletion.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                }, {
                    "type": "text",
                    "text": prompt
                }]
            },
            {  # prefill response to avoid rambling and ensure structured output
                "role": "assistant",
                "content": "{"
            }
        ],
        temperature=0.4,  # Lower temperature for more deterministic outputs
        max_tokens=1000)

    # Extract the response content
    result_text = response.choices[0].message.content.strip()
    
    # Try to parse the response as JSON
    try:
      print("Raw result:", result_text)
      if not result_text.startswith("{"):
        result_text = "{" + result_text
      if not result_text.endswith("}"):
        result_text = result_text + "}"

      result = json.loads(result_text)
      
    except json.JSONDecodeError:
      # If parsing fails, use LLM to fix the JSON
      st.info("Attempting to fix JSON format ...")
      fixed_json = fix_json_with_llm(result_text)

      try:
        result = json.loads(fixed_json)
        st.success("JSON successfully fixed!")
      except json.JSONDecodeError:
        # If still not working, return raw response
        st.warning("Could not parse response as JSON. Showing raw output.")
        result = {"Raw Response": result_text}

    return result

  except Exception as e:
    st.error(f"Error processing image: {str(e)}")
    return None


def main():
  st.title("KYC Document Verification")
  st.write(
      "Upload one or more images of identification documents to extract identity information."
  )
  st.write(
      "**Note**: make sure the documents are in the correct orientation. Automatic correction can be implemented eventually beyond this PoC."
  )

  # File uploader
  uploaded_files = st.file_uploader("Choose image files",
                                    type=["jpg", "jpeg", "png", "pdf"],
                                    accept_multiple_files=True)

  if uploaded_files and st.button("Process Documents"):
    results = []
    progress_bar = st.progress(0)

    for idx, file in enumerate(uploaded_files):
      st.write(f"**Processing:** {file.name}")

      # Get file extension to determine image type
      file_extension = file.name.split('.')[-1].lower()
      image_type = file_extension if file_extension in ['jpg', 'jpeg', 'png'
                                                        ] else 'jpeg'

      # Read file bytes
      image_bytes = file.read()

      # Process the image
      data = process_image(image_bytes, image_type)

      if data:
        data["File Name"] = file.name  # Include the file name for reference
        results.append(data)

      # Update progress bar
      progress_bar.progress((idx + 1) / len(uploaded_files))

    if results:
      # Display the extracted data in a table
      st.write("### Extracted KYC Information")
      df = pd.DataFrame(results)
      st.dataframe(df)

      # Convert dataframe to CSV for download
      csv = df.to_csv(index=False).encode('utf-8')
      st.download_button(label="Download results as CSV",
                         data=csv,
                         file_name='extracted_kyc_data.csv',
                         mime='text/csv')
    else:
      st.warning(
          "No valid data could be extracted from the uploaded documents.")


if __name__ == '__main__':
  main()