import os
from datetime import datetime, timedelta
import re
import tempfile
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai
import json
import pymongo
import requests
import sib_api_v3_sdk
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from PyPDF2 import PdfReader
import pytz
from bson import json_util
from sib_api_v3_sdk.rest import ApiException
import textwrap

load_dotenv()

app = FastAPI(
    title="SeraAI API",
    description="API for fetching, summarizing, and scheduling newsletters from Supreme Court Judgements.",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("APP_URL")],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Configure the Generative AI model
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
fallback_model = genai.GenerativeModel(
    "gemini-1.5-pro"
)  # Fallback model
# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
# Constants for turbo processing
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_CHUNK_SIZE = 8000  # characters
# Configure Brevo Mail API
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key["api-key"] = os.getenv("BREVO_API_KEY")
# Create instances of the API classes
template_api = sib_api_v3_sdk.TransactionalEmailsApi(
    sib_api_v3_sdk.ApiClient(configuration)
)
campaign_api = sib_api_v3_sdk.EmailCampaignsApi(
    sib_api_v3_sdk.ApiClient(configuration)
)
contacts_api_instance = sib_api_v3_sdk.ContactsApi(
    sib_api_v3_sdk.ApiClient(configuration)
)

# Pydantic Models
class PDFData(BaseModel):
    filename: str
    file_content: str

class Summary(BaseModel):
    date: str
    CA: str
    title: str
    Respondent: str
    background: str
    chronology: List[str]
    key_points: List[str]
    conclusion: List[str]
    Judgment_By: List[str]

class NewsletterData(BaseModel):
    body: str

class ScheduleNewsletterData(BaseModel):
    body: str
    list_id: int
    scheduled_time: str
    period_of_time: str
    sender_name: str


# Helper functions (same as Streamlit app):
def sanitize_filename(filename):
    return re.sub(r"[<>:\"/\\|?*]", "_", filename)


def parse_date(date_str, input_format="%d-%b-%Y"):
    try:
        return datetime.strptime(date_str, input_format).date()
    except ValueError:
        return None


def fetch_from_website(user_date):
    url = "https://www.sci.gov.in/#1697446384453-9aeef8cc-5f35"
    response = requests.get(url)
    fetched_pdfs = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Locate the container with the PDF links in the updated HTML structure
        judgement_div = soup.find("div", {"id": "1697446384453-9aeef8cc-5f35"})

        if judgement_div:
            pdf_links = []
            for li in judgement_div.find_all("li"):
                a_tag = li.find("a", href=True)
                if a_tag and "href" in a_tag.attrs:
                    href = a_tag["href"]
                    # Check if the link ends with .pdf or has 'view-pdf' in the href
                    if href.endswith(".pdf") or "view-pdf" in href:
                        # Transform the URL from view-pdf to sci-get-pdf
                        href = href.replace("view-pdf", "sci-get-pdf")
                        upload_div = a_tag.find("div", style="color:#5959dd;")
                        if upload_div:
                            upload_date_str = re.search(
                                r"\d{2}-\d{2}-\d{4}", upload_div.text
                            ).group()
                            upload_date = datetime.strptime(
                                upload_date_str, "%d-%m-%Y"
                            ).date()
                            # Check if the upload date matches the user's date
                            if upload_date and upload_date == user_date:
                                pdf_links.append(
                                    (href, a_tag.text.strip(), upload_date)
                                )

            for idx, (link, description, upload_date) in enumerate(pdf_links):
                pdf_response = requests.get(link)
                if pdf_response.status_code == 200:
                    pdf_name = f"{idx+1}_{sanitize_filename(description[:100])}.pdf"
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_file:
                        temp_file.write(pdf_response.content)
                        fetched_pdfs.append((pdf_name, temp_file.name))
    return fetched_pdfs


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return None


def generate_summary(text, filename):
    prompt = f"""
    Summarize the following Supreme Court judgment in the format specified below:
    
    {{
      "date": "YYYY-MM-DD",
      "CA": "CIVIL APPEAL NO./ Case Number/ Petition Number of case",
      "title": "A professional tone title summarizing the case",
      "Respondent": "Name of APPELLANT VS Name of RESPONDENTS",
      "background": "Brief background of the case explaining the dispute",
      "chronology": [
        "Month YYYY: Details of events in a single line.",
        "Month YYYY: Details of events in a single line." 
      ],
      "key_points": [
        "A key legal point discussed in the judgment.",
        "Another key legal point discussed in the judgment."
      ],
      "conclusion": [
        "A crisped and precise summary of the judgment's outcome."
      ],
      "Judgment_By": [
        "Name of the judge who have given the judgement with proper HON'BLE MR./MRS. JUSTICE"
      ]
    }}

    Judgment text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip()
        if not (cleaned_response.startswith("{") and cleaned_response.endswith("}")):
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}") + 1
            if start != -1 and end != 0:
                cleaned_response = cleaned_response[start:end]
            else:
                raise ValueError("Could not extract valid JSON from the response")

        return json.loads(cleaned_response), None
    except json.JSONDecodeError as e:
        return None, cleaned_response
    except Exception as e:
        return None, str(e)


def insert_all_to_db(merged_data):
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["myDatabase"]
    collection = db["SeraAI"]
    try:
        result = collection.insert_many(merged_data)
        return f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB."
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error inserting data into MongoDB: {str(e)}"
        )


def generate_newsletter(case_summaries):
    prompt = f"""
    You are a technical writing expert and newsletter guru, specializing in crafting highly engaging content with captivating headlines and subject lines. Analyze the following list of Supreme Court case summaries and create a newsletter post that compels readers to gains insights, the article must target professionals so the tone of newsletter generated is also needs to be professional so that it properly target reader.

    Case Summaries:
    {json.dumps(case_summaries, indent=2)}

    Generate a single newsletter post in HTML format with the following structure including all the judgments in the list above:
    Use appropriate HTML tags for formatting, including <h1>, <h2>, <h3>, <p>, <ul>, <li>, etc.
    To generate the read more reference links use this simple strategy - https://seraphicadvisors.com/sera-ai/YYYY-MM-DD/id , just replace the date in the link and id with $oid value
    Here is sample example for you to first learn how to do it:
        <html>
        <head>
            <title> Featured Case: Title of the most interesting case </title>
        </head>
        <body>
            <h1 style="text-align: left; color:black;"> Featured Case: Title of the most interesting case </h1>
            <p style="text-align: left; color:black;">A little idea about that case</p>
            <p style="text-align: left;"><strong><span style="color:red;">Read the Full Article: </span></strong><a href="https://seraphicadvisors.com/sera-ai/2024-07-16/66974ecd4167f217f43965a1" tabindex="-1" style="color: blue;"><span style="color:blue;">Here</span></a><br></p>
            <p style="text-align:left;">[Name of APPELLANT VS Name of RESPONDENTS, decided on DD-MM-YYYY]</p>
            <br/>
            <h2 style="text-align: left; color:black;">Other Supreme Court Decisions You Might Find Interesting:</h2>
            <br/>
            <h3 style="text-align: left; color:black;">[Case 2 Title]</h3>
            <p style="text-align: left; color:black;">[Brief and insights of the case]</p>
            <p style="text-align: left;"><strong><span style="color:red;">Read the Full Article: </span></strong><a href="https://seraphicadvisors.com/sera-ai/YYYY-MM-DD/id" style="color: blue;" tabindex="-1"><span style="color:blue;">Here</span></a><br/></p>
            <p style="text-align:left;">[Name of APPELLANT VS Name of RESPONDENTS, decided on DD-MM-YYYY]</p>
            <br/>
            <h3 style="text-align: left; color:black;">[Case 3 Title]</h3>
            <p style="text-align: left; color:black;">[Brief and insights of the case]</p>
            <p style="text-align: left;"><strong><span style="color:red;">Read the Full Article: </span></strong><a href="https://seraphicadvisors.com/sera-ai/YYYY-MM-DD/id" style="color: blue;" tabindex="-1"><span style="color:blue;">Here</span></a><br/></p>
            <p style="text-align:left;">[Name of APPELLANT VS Name of RESPONDENTS, decided on DD-MM-YYYY]</p>
            <br/>
            <h3 style="text-align: left; color:black;">[Case 4 Title]</h3>
            <p style="text-align: left; color:black;">[Brief and insights of the case]</p>
            <p style="text-align: left;"><strong><span style="color:red;">Read the Full Article: </span></strong><a href="https://seraphicadvisors.com/sera-ai/YYYY-MM-DD/id" style="color: blue;" tabindex="-1"><span style="color:blue;">Here</span></a><br/></p>
            <p style="text-align:left;">[Name of APPELLANT VS Name of RESPONDENTS, decided on DD-MM-YYYY]</p>
            </body>
        </body>
        </html>
    """

    response = model.generate_content(prompt)
    return response.text


def extract_title_and_body(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    body = str(soup.body) if soup.body else "No body content found"
    return title, body


def fetch_articles_from_mongodb(date):
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["myDatabase"]
    collection = db["SeraAI"]

    date_str = date.strftime("%Y-%m-%d")
    articles = collection.find({"date": date_str})

    # Convert BSON to JSON-serializable dictionaries
    return json.loads(json_util.dumps(list(articles)))


def get_templates():
    try:
        api_response = template_api.get_smtp_templates(
            limit=10, offset=0, sort="desc"
        )
        templates = [(template.id, template.name) for template in api_response.templates]
        return templates
    except ApiException as e:
        raise HTTPException(
            status_code=500, detail=f"Exception when calling get_smtp_templates: {e}"
        )


def preview_template(template_id):
    try:
        api_response = template_api.get_smtp_template(template_id)
        return api_response.html_content
    except ApiException as e:
        raise HTTPException(
            status_code=500, detail=f"Exception when calling get_smtp_template: {e}"
        )


def create_campaign_draft(
    subject, sender_name, sender_email, modified_html, list_id, scheduled_utc_time
):
    try:
        email_campaigns = {
            "name": subject,
            "subject": subject,
            "sender": {"name": sender_name, "email": sender_email},
            "htmlContent": modified_html,
            "recipients": {"listIds": [list_id]},
            "scheduledAt": scheduled_utc_time,
        }
        # Create the campaign draft
        api_response = campaign_api.create_email_campaign(email_campaigns)
        return f"Campaign scheduled successfully. ID: {api_response.id}"
    except ApiException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Exception when calling create_email_campaign: {e}",
        )
    except Exception as ex:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {ex}"
        )


def insert_custom_content(template_html, custom_html):
    placeholder = '<p style="margin: 0; background-font-weight: normal;">!-- CUSTOM_CONTENT --!</p>'
    if placeholder in template_html:
        return template_html.replace(placeholder, custom_html)
    else:
        return template_html.replace("</body>", f"{custom_html}</body>")


def convert_to_utc(ist_date_str, ist_time_str, am_pm):
    # Parse date and time from input
    ist_datetime_str = f"{ist_date_str} {ist_time_str} {am_pm}"
    ist_datetime = datetime.strptime(ist_datetime_str, "%Y-%m-%d %I:%M %p")

    # Set IST timezone
    ist_tz = pytz.timezone("Asia/Kolkata")
    ist_datetime = ist_tz.localize(ist_datetime)

    # Convert to UTC
    utc_datetime = ist_datetime.astimezone(pytz.utc)
    return utc_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def generate_subject():
    current_date = datetime.now().strftime("%d %B %Y")
    return f"Seraphic Advisors - Newsletter || {current_date}"

# API Endpoints
@app.post("/api/fetch_pdfs", response_model=List[PDFData], summary="Fetches PDFs from a website based on the input date, extracts text and returns in desired format")
async def fetch_pdfs(user_date: str = Form(...)):
    try:
        parsed_date = parse_date(user_date, "%Y-%m-%d")
        if not parsed_date:
            raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD format.")
        fetched_pdfs = fetch_from_website(parsed_date)
        pdf_data_list = []
        for pdf_name, pdf_path in fetched_pdfs:
            text = extract_text_from_pdf(pdf_path)
            if text:
               pdf_data_list.append(PDFData(filename=pdf_name, file_content=text))
            else:
               raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract text from PDF: {pdf_name}",
               )
        return pdf_data_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_summaries", response_model=List[Summary], summary="Generates summaries using the Gemini model from the received extracted data")
async def generate_summaries(pdf_data_list: List[PDFData]):
    summaries = []
    for pdf_data in pdf_data_list:
        summary, error_text = generate_summary(pdf_data.file_content, pdf_data.filename)
        if summary:
           summaries.append(Summary(**summary))
        else:
            raise HTTPException(status_code=500, detail=f"Failed to generate summary for: {pdf_data.filename} Error: {error_text}")
    return summaries


@app.post("/api/store_summaries", summary="Store generated summaries to MongoDB", response_model=str)
async def store_summaries_to_db(summaries: List[Summary]):
    try:
       summary_dicts = [summary.dict() for summary in summaries]
       result = insert_all_to_db(summary_dicts)
       return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_newsletter", response_model=NewsletterData, summary="Generates a newsletter from the fetched summaries from MongoDB on the provided date")
async def generate_newsletter_endpoint(date: str = Form(...)):
    try:
        parsed_date = parse_date(date, "%Y-%m-%d")
        if not parsed_date:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Please use YYYY-MM-DD format."
            )
        articles = fetch_articles_from_mongodb(parsed_date)
        if not articles:
            raise HTTPException(
                status_code=404, detail="No articles found for the specified date."
            )
        newsletter_html = generate_newsletter(articles)
        _, newsletter_body = extract_title_and_body(newsletter_html)
        return NewsletterData(body=newsletter_body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule_newsletter", summary="Schedule newsletter delivery using the Brevo API.", response_model=str)
async def schedule_newsletter_endpoint(data: ScheduleNewsletterData):
    try:
        templates = get_templates()
        selected_template = templates[1]
        template_html = preview_template(selected_template[0])

        sender_email = "newsletter@seraphicadvisors.info"
        subject = generate_subject()

        scheduled_utc_time = convert_to_utc(
            datetime.now().strftime("%Y-%m-%d"), data.scheduled_time, data.period_of_time
        )  # Assuming it is in AM

        modified_html = insert_custom_content(template_html, data.body)

        result = create_campaign_draft(
            subject, data.sender_name, sender_email, modified_html, data.list_id, scheduled_utc_time
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to the SeraAI Agent FastAPI"}
    )