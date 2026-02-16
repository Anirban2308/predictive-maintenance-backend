from fastapi import FastAPI
from crewai import Agent, Task, Crew
import os
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import json
import cohere
from langchain_cohere import ChatCohere
import os
from bson import ObjectId
from fastapi import HTTPException

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = "dummy"

# ------------------------------------------------
# FASTAPI INITIALIZATION
# ------------------------------------------------
app = FastAPI()
origins = [
    "https://predictive-maintenance-frontend.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# CONSTANT VEHICLE ID
# ------------------------------------------------
FIXED_VEHICLE_ID = "V101"

# ------------------------------------------------
# MONGODB SETUP
# ------------------------------------------------
MONGO_URL = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(MONGO_URL)

db = client["autopredict"]
workflow_collection = db["maintenance_workflow"]
appointment_collection = db["appointments"]

# ------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------
class TelemetryRequest(BaseModel):
    data: List[Dict]

class VoiceQuery(BaseModel):
    message: str

class Appointment(BaseModel):
    vehicle_id: str
    service_type: str
    date: str
    time: str
    status: str = "Pending"
    recommended_by_ai: bool = True

class AppointmentStatusUpdate(BaseModel):
    status: str  # Pending | In Progress | Resolved

# ------------------------------------------------
# OPENAI SETUP
# ------------------------------------------------


# Create Cohere LLM for CrewAI
llm = ChatCohere(
    model="command-r-plus",
    temperature=0.3,
    cohere_api_key=os.environ["COHERE_API_KEY"]
)
# ------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------
def calculate_brake_health(vibration=None):
    if vibration is None:
        return 75
    if vibration < 3:
        return 95
    elif vibration < 5:
        return 80
    elif vibration < 7:
        return 65
    else:
        return 40

# ------------------------------------------------
# AGENT DEFINITIONS
# ------------------------------------------------
master_agent = Agent(
    role="Master Agent",
    goal="Manage entire maintenance workflow.",
    backstory="Controls sub-agents and ensures correct processing.",
    llm=llm
)

data_agent = Agent(
    role="Telemetry Data Analyst",
    goal="Detect issues, severity, confidence and compute health score.",
    backstory="Identifies anomalies in telemetry.",
    llm=llm
)

diagnosis_agent = Agent(
    role="Technical Diagnosis Expert",
    goal="Provide root-cause analysis.",
    backstory="Automotive domain specialist.",
    llm=llm
)

customer_agent = Agent(
    role="Customer Assistant",
    goal="Explain issues to user in simple language.",
    backstory="Communicates complex issues simply.",
    llm=llm
)

scheduler_agent = Agent(
    role="Maintenance Scheduler",
    goal="Suggest repair appointment time.",
    backstory="Skilled in service slot planning.",
    llm=llm
)

feedback_agent = Agent(
    role="Feedback Summarizer",
    goal="Create short feedback summary.",
    backstory="Summarizes insights concisely.",
    llm=llm
)

# ------------------------------------------------
# BASIC ROOT ROUTE
# ------------------------------------------------
@app.get("/")
async def home():
    return {"message": "Predictive Maintenance AI Running 🚀"}

# ------------------------------------------------
# VOICE ENDPOINT
# ------------------------------------------------
@app.post("/voice-query")
async def voice_query(req: VoiceQuery):
    payload = TelemetryRequest(data=[{"user_message": req.message}])
    response = await run_maintenance(payload)
    return {"reply": response["customer_message"]}

# ------------------------------------------------
# MAIN AGENT WORKFLOW
# ------------------------------------------------
@app.post("/run-maintenance")
async def run_maintenance(request: TelemetryRequest):
    data = request.data

    # ------------------------------------------------
    # HANDLE NATURAL LANGUAGE QUERIES
    # ------------------------------------------------
    if "user_message" in data[0] and not any(
        k in data[0] for k in ["engineTemp", "oilPressure", "batteryVoltage", "tirePressure"]
    ):
        user_question = data[0]["user_message"].lower()
        doc = await workflow_collection.find_one({"vehicle_id": FIXED_VEHICLE_ID})

        if not doc:
            return {"customer_message": "Your vehicle is healthy. No issues detected."}

        if "health" in user_question:
            return {"customer_message": f"Your vehicle health score is {doc.get('overall_health_score', 100)}%"}

        if "issue" in user_question or "problem" in user_question:
            issues = doc.get("issues", [])
            if not issues:
                return {"customer_message": "There are no current issues."}
            issue_names = ", ".join(i["name"] for i in issues)
            return {"customer_message": f"Current issues: {issue_names}"}

        if "schedule" in user_question or "maintenance" in user_question:
            return {"customer_message": f"Your maintenance is scheduled for {doc.get('predicted_schedule', 'Not scheduled')}."}

        if "engine temperature" in user_question or "engine temp" in user_question:
            telemetry = doc.get("telemetry", {})
            temp = telemetry.get("engineTemp", "Not available")
            return {"customer_message": f"Your engine temperature is {temp}."}

        return {"customer_message": "Ask me about health, issues or schedule."}

    # ------------------------------------------------
    # TELEMETRY PROCESSING
    # ------------------------------------------------
    telemetry = data[0]
    telemetry["id"] = FIXED_VEHICLE_ID
    issues = []

    engine_temp = float(telemetry.get("engineTemp", "90").replace("°C",""))
    if engine_temp > 100:
        issues.append({
            "name": "Engine Temperature High",
            "severity": "high",
            "confidence": 0.9,
            "recommended_action": "Check cooling system"
        })

    if telemetry.get("oilPressure", 30) < 20:
        issues.append({
            "name": "Oil Pressure Low",
            "severity": "high",
            "confidence": 0.85,
            "recommended_action": "Check oil level"
        })

    if telemetry.get("batteryVoltage", 12.6) < 12:
        issues.append({
            "name": "Battery Voltage Low",
            "severity": "medium",
            "confidence": 0.75,
            "recommended_action": "Recharge or replace battery"
        })

    if telemetry.get("tirePressure", 32) < 30:
        issues.append({
            "name": "Tire Pressure Low",
            "severity": "medium",
            "confidence": 0.7,
            "recommended_action": "Inflate tires"
        })

    # Technician-friendly summary
    diagnosis_text = ", ".join([f"{i['name']} — {i['severity']}" for i in issues]) \
                     if issues else "No issues detected."

    # Customer-friendly message
    customer_message = ""
    if issues:
        for i in issues:
            customer_message += f"{i['name']} ({i['severity']}): {i['recommended_action']}. "
    else:
        customer_message = "Everything is normal."

    # ------------------------------------------------
    # COMPONENT HEALTH & OVERALL HEALTH
    # ------------------------------------------------
    component_health = {
        "engineHealth": max(0, 100 - (engine_temp - 90) * 2),
        "battery": int((float(telemetry.get("batteryVoltage", 12.6)) / 12.6) * 100),
        "brakes": calculate_brake_health(telemetry.get("vibration")),
        "tirePressure": "Low" if telemetry.get("tirePressure", 30) < 30 else "Normal"
    }

    overall_health = int(
        (component_health["engineHealth"] +
         component_health["battery"] +
         component_health["brakes"] +
         (100 if component_health["tirePressure"] == "Normal" else 70)) / 4
    )

    # ------------------------------------------------
    # SCHEDULE
    # ------------------------------------------------
    schedule_text = "Tomorrow 10 AM" if issues else "No appointment needed"
    timestamp_now = datetime.utcnow()

    # Load history
    existing = await workflow_collection.find_one({"vehicle_id": FIXED_VEHICLE_ID})
    history = existing.get("history", []) if existing else []

    history.append({
        "timestamp": timestamp_now,
        "telemetry": telemetry,
        "issues": issues
    })

    # ------------------------------------------------
    # SAVE TO WORKFLOW COLLECTION
    # ------------------------------------------------
    await workflow_collection.replace_one(
        {"vehicle_id": FIXED_VEHICLE_ID},
        {
            "vehicle_id": FIXED_VEHICLE_ID,
            "telemetry": telemetry,
            "issues": issues,
            "component_health": component_health,
            "overall_health_score": overall_health,
            "diagnosis_summary": diagnosis_text,
            "customer_friendly_message": customer_message,
            "predicted_schedule": schedule_text,
            "feedback": "Immediate maintenance recommended due to detected issues." if issues else "System operating normally.",
            "last_updated": timestamp_now,
            "history": history
        },
        upsert=True
    )

    # ------------------------------------------------
    # CREATE SINGLE MERGED APPOINTMENT
    # ------------------------------------------------
    if issues:
        service_name = ", ".join(i["name"] for i in issues)
        existing_appointment = await appointment_collection.find_one({
            "vehicle_id": FIXED_VEHICLE_ID,
            "date": schedule_text.split()[0],
            "time": "10:00 AM",
            "status": "Pending"
        })
        if not existing_appointment:
            await appointment_collection.insert_one({
                "vehicle_id": FIXED_VEHICLE_ID,
                "service_type": service_name,
                "date": schedule_text.split()[0],
                "time": "10:00 AM",
                "status": "Pending",
                "recommended_by_ai": True,
                "created_at": timestamp_now
            })

    return {
        "issues": issues,
        "anomalies": issues, 
        "health_score": overall_health,
        "diagnosis_summary": diagnosis_text,
        "customer_message": customer_message,
        "schedule": schedule_text,
        "feedback": "Immediate maintenance recommended due to detected issues." if issues else "System operating normally."
    }

# ------------------------------------------------
# DASHBOARD ENDPOINTS
# ------------------------------------------------
@app.get("/dashboard/stats")
async def get_dashboard_stats():
    total_runs = await workflow_collection.count_documents({})
    all_docs = workflow_collection.find({})
    issues_count = 0
    critical_count = 0
    async for doc in all_docs:
        for issue in doc.get("issues", []):
            issues_count += 1
            if issue.get("severity") == "high":
                critical_count += 1
    return {
        "total_runs": total_runs,
        "total_issues": issues_count,
        "critical_issues": critical_count
    }

@app.get("/dashboard/data")
async def get_dashboard_data():
    try:
        doc = await workflow_collection.find_one({"vehicle_id": FIXED_VEHICLE_ID})
        if not doc:
            # Return empty dashboard instead of crashing
            return {
                "telemetry": {},
                "health_score": 100,
                "component_health": {},
                "predicted_issues": [],
                "anomalies": [],
                "diagnosis": "",
                "customer_message": "No data available",
                "schedule": "No appointment needed",
                "feedback": "",
                "last_updated": None
            }

        return {
            "telemetry": doc.get("telemetry") or {},
            "health_score": doc.get("overall_health_score") or 100,
            "component_health": doc.get("component_health") or {},
            "predicted_issues": doc.get("issues") or [],
            "anomalies": doc.get("issues") or [],
            "diagnosis": doc.get("diagnosis_summary") or "",
            "customer_message": doc.get("customer_friendly_message") or "",
            "schedule": doc.get("predicted_schedule") or "No appointment needed",
            "feedback": doc.get("feedback") or "",
            "last_updated": doc.get("last_updated")
        }
    except Exception as e:
        # Log error on server, return safe response
        print("Error in /dashboard/data:", e)
        return {
            "telemetry": {},
            "health_score": 100,
            "component_health": {},
            "predicted_issues": [],
            "anomalies": [],
            "diagnosis": "",
            "customer_message": "Error fetching data",
            "schedule": "No appointment needed",
            "feedback": "",
            "last_updated": None
        }


# ------------------------------------------------
# APPOINTMENTS ENDPOINTS
# ------------------------------------------------
@app.get("/appointments")
async def get_active_appointments():
    appointments = []
    cursor = appointment_collection.find({
        "vehicle_id": FIXED_VEHICLE_ID,
        "status": {"$ne": "Resolved"}
    })
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        appointments.append(doc)
    return appointments

@app.get("/appointments/history")
async def get_appointment_history():
    history = []
    cursor = appointment_collection.find({
        "vehicle_id": FIXED_VEHICLE_ID,
        "status": "Resolved"
    })
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        history.append(doc)
    return history

@app.patch("/appointments/{appointment_id}/status")
async def update_appointment_status(appointment_id: str, update: AppointmentStatusUpdate):
    result = await appointment_collection.update_one(
        {"_id": ObjectId(appointment_id)},
        {"$set": {
            "status": update.status,
            "updated_at": datetime.utcnow()
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"message": "Appointment status updated successfully"}

@app.get("/analytics/full")
async def get_full_analytics():
    doc = await workflow_collection.find_one({"vehicle_id": FIXED_VEHICLE_ID})
    if not doc:
        return {
            "sensors": {},
            "violations": [],
            "timeline": []
        }

    history = doc.get("history", [])

    # ---------------- SENSOR TRENDS ----------------
    def build_series(key, safe_range):
        values = []
        for h in history[-24:]:
            if key in h["telemetry"]:
                values.append({
                    "time": h["timestamp"].strftime("%H:%M"),
                    "value": float(str(h["telemetry"][key]).replace("°C", ""))
                })
        return {
            "values": values,
            "safe_range": safe_range,
            "ai_insight": f"{key.replace('_', ' ').title()} trend indicates deviation from safe operating range."
        }

    sensors = {
        "engine_temperature": build_series("engineTemp", (70, 95)),
        "oil_pressure": build_series("oilPressure", (25, 65)),
        "battery_voltage": build_series("batteryVoltage", (12, 14)),
        "tire_pressure": build_series("tirePressure", (32, 36))
    }

    # ---------------- THRESHOLD VIOLATIONS ----------------
    violations = []
    for issue in doc.get("issues", []):
        violations.append({
            "parameter": issue["name"],
            "safe_range": "See specs",
            "current": "Out of range",
            "deviation": issue["severity"],
            "breach_duration": "—",
            "status": "Critical" if issue["severity"] == "high" else "Moderate"
        })

    # ---------------- FAILURE TIMELINE ----------------
    timeline = []
    for h in history[-6:]:
        if h["issues"]:
            timeline.append({
                "time": h["timestamp"].strftime("Day %d — %H:%M"),
                "stage": h["issues"][0]["name"],
                "description": h["issues"][0]["recommended_action"]
            })

    return {
        "sensors": sensors,
        "violations": violations,
        "timeline": timeline
    }
