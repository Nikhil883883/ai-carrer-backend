from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import PyPDF2
import io
from typing import Dict, List
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
import json
from datetime import datetime


load_dotenv() 

app = FastAPI(
    title="AI Career Advisor API",
    description="API for career recommendations, chat assistance, and job trends",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CareerRequest(BaseModel):
    name: Optional[str] = None
    ageGroup: str
    qualification: str
    skills: str
    interests: str
    workStyle: str
    experience: str
    subjects: Optional[str] = None
    hobbies: Optional[str] = None
    learningStyle: str
    priority: str
    personalityType: Optional[str] = None
    preferredIndustries: Optional[str] = None
    longTermGoal: Optional[str] = None
    workingHours: Optional[str] = None
    travelWillingness: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama3-8b-8192"  
class ResumeVettingRequest(BaseModel):
    resume_text: Optional[str] = None
    resume_pdf: Optional[bytes] = None
    job_description: Optional[str] = None
    model: Optional[str] = "llama3-70b-8192"
    analysis_type: Optional[List[str]] = ["ats", "skills", "structure"] 
import PyPDF2
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip() if text else ""
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process PDF: {str(e)}"
        )

def handle_api_error(e: Exception, service_name: str) -> None:
    
    error_msg = f"{service_name} API Error: {str(e)}"
    if isinstance(e, requests.exceptions.RequestException):
        error_msg += f" | Status Code: {e.response.status_code if hasattr(e, 'response') else 'Unknown'}"
    raise HTTPException(status_code=500, detail=error_msg)

def call_groq_api(prompt: str, model: str = "llama3-8b-8192") -> str:
    
    groq_api_key = os.getenv("GROQ_API_KEY") or "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"
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional career counselor with 20 years of experience."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
@app.post("/chat/", summary="Chat with AI assistant")
async def chat_with_bot(payload: ChatRequest):
   
    try:
        response = call_groq_api(payload.message, payload.model)
        return {
            "response": response,
            "model": payload.model
        }
    except Exception as e:
        handle_api_error(e, "Groq Chat")

@app.post("/career/", summary="Get career recommendations")
async def recommend_career(
    resume_pdf: UploadFile = File(None),
    name: str = Form(None),
    ageGroup: str = Form(None),
    qualification: str = Form(None),
    skills: str = Form(None),
    interests: str = Form(None),
    workStyle: str = Form(None),
    experience: str = Form(None),
    subjects: str = Form(None),
    hobbies: str = Form(None),
    learningStyle: str = Form(None),
    priority: str = Form(None),
    personalityType: str = Form(None),
    preferredIndustries: str = Form(None),
    longTermGoal: str = Form(None),
    workingHours: str = Form(None),
    travelWillingness: str = Form(None)
):
    
    try:
       
        extracted_info = ""
        if resume_pdf:
            if resume_pdf.content_type != "application/pdf":
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Only PDF files are allowed"}
                )
            pdf_bytes = await resume_pdf.read()
            extracted_info = extract_text_from_pdf(pdf_bytes)
        
        
        form_fields = {
            "name": name,
            "ageGroup": ageGroup,
            "qualification": qualification,
            "skills": skills,
            "interests": interests,
            "workStyle": workStyle,
            "experience": experience,
            "subjects": subjects,
            "hobbies": hobbies,
            "learningStyle": learningStyle,
            "priority": priority,
            "personalityType": personalityType,
            "preferredIndustries": preferredIndustries,
            "longTermGoal": longTermGoal,
            "workingHours": workingHours,
            "travelWillingness": travelWillingness
        }

      
        prompt = f"""Analyze this career profile and provide detailed recommendations:
        
        ### Extracted from Resume:
        {extracted_info[:10000] if extracted_info else "No resume provided"}

        ### Provided Information:
        {json.dumps({k: v or "Not provided" for k, v in form_fields.items()}, indent=2)}

        ### Instructions:
        1. First extract key information from the resume if provided
        2. Combine with explicitly provided information
        3. Provide 3 detailed career recommendations with:
           - Career title
           - Why it's a good fit
           - Required qualifications/skills
           - Growth potential
           - Salary expectations
           - First steps to pursue
        4. Format with clear headings and bullet points
        5. Highlight which information came from resume vs form
        """

        
        recommendations = call_groq_api(prompt, "llama3-70b-8192")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "recommendations": recommendations,
                "source": "PDF + Form Fields" if extracted_info else "Form Fields Only"
            }
        )

    except Exception as e:
        print(f"Career recommendation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "detail": f"Career analysis failed: {str(e)}",
                "suggestion": "Try providing more information or a different resume"
            }
        )

@app.get("/jobs/")
async def groq_job_search(query: str = "software engineer", location: str = "remote"):
   
    try:
        prompt = f"""
        Act as a senior career advisor with access to global job market data.
        Provide detailed insights about {query} jobs in {location} including:
        
        1. Current market demand (High/Medium/Low)
        2. Typical salary ranges
        3. Top companies hiring for these roles
        4. Required skills/qualifications
        5. 5 specific job recommendations with:
           - Hypothetical job titles
           - Company names
           - Key requirements
           - Growth potential
           - Recommended application strategy
        
        Format the response with clear sections and bullet points.
        Include realistic but generic examples since we're not connecting to actual job boards.
        """
        
        recommendations = call_groq_api(prompt)
        
        return {
            "source": "AI-generated market analysis",
            "query": query,
            "location": location,
            "analysis": recommendations,
            "note": "These are AI-generated insights based on general market knowledge"
        }
        
    except Exception as e:
        handle_api_error(e, "Groq Job Search")
@app.post("/vet-resume/")
async def vet_resume(
    resume_pdf: UploadFile = File(None),
    resume_text: str = Form(None),
    job_description: str = Form(None),
    analysis_type: List[str] = Form(["ats", "skills", "structure"]),
    nocache: str = Form(None)  
):
    try:
        
        resume_content = resume_text or ""
        if resume_pdf:
            if resume_pdf.content_type != "application/pdf":
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Only PDF files are allowed"}
                )
            pdf_bytes = await resume_pdf.read()
            resume_content = extract_text_from_pdf(pdf_bytes)
        
        if not resume_content.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "No resume content provided"}
            )

        
        prompt = f"""Analyze this resume for a {job_description or 'general'} position.
        Current date: {datetime.now().strftime("%Y-%m-%d")}
        Analysis ID: {nocache or "1"}
        
        Return ONLY valid JSON with these exact fields:
        {{
            "overall_score": (0-100, calculated based on content),
            "ats_score": (0-100, ATS compatibility),
            "skill_score": (0-100, skill match),
            "structure_score": (0-100, organization),
            "detailed_feedback": (markdown string),
            "missing_skills": [array of strings],
            "suggested_improvements": [array of strings]
        }}

        RESUME CONTENT:
        {resume_content[:15000]}

        JOB DESCRIPTION:
        {job_description or 'Not provided'}

        IMPORTANT:
        - Calculate fresh scores each time
        - Never return placeholder/default values
        - Ensure all scores are between 0-100
        - JSON must be parseable
        """

        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                groq_response = call_groq_api(prompt, "llama3-70b-8192")
                
                
                json_str = groq_response[groq_response.find('{'):groq_response.rfind('}')+1]
                analysis = json.loads(json_str)
                
                
                required_fields = ['overall_score', 'ats_score', 'skill_score']
                for field in required_fields:
                    if field not in analysis or not isinstance(analysis[field], int) or not 0 <= analysis[field] <= 100:
                        raise ValueError(f"Invalid {field} value")
                
                break  
            
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:  
                    print(f"Groq response parsing failed: {str(e)}")
                    print(f"Raw response: {groq_response}")
                    raise
                continue

        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": {
                    "overall_score": analysis["overall_score"],
                    "ats_score": analysis["ats_score"],
                    "skill_score": analysis["skill_score"],
                    "structure_score": analysis.get("structure_score", 0),
                    "detailed_feedback": analysis["detailed_feedback"],
                    "missing_skills": analysis["missing_skills"],
                    "suggested_improvements": analysis["suggested_improvements"],
                    "analysis_id": nocache  
                }
            }
        )

    except Exception as e:
        print(f"Error in vet_resume: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "detail": f"Analysis failed: {str(e)}",
                "retry_suggestion": "Please try again with a slightly modified resume text"
            }
        )
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": app.version}