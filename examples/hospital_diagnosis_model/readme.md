# 🧠 Medical Diagnosis ABM (Mesa + Mesa-LLM)

## 🔥 1. Core Idea

This project simulates a **real-world hospital diagnosis workflow** using Agent-Based Modeling (ABM):

- Patients develop symptoms and seek care
- Doctors diagnose, recommend tests, and prescribe treatment
- Hospitals manage resources (beds, doctors, labs)
- Optional: LLM-based reasoning enhances diagnosis decisions

This model captures:
- Patient flow
- Diagnosis uncertainty
- Resource constraints
- Waiting times
- Mortality and recovery dynamics

---

## 🧱 2. Agents

### 👤 2.1 PatientAgent (Core Entity)

#### Attributes
- `id`
- `age`
- `gender`
- `symptoms` (list)
- `disease` (hidden ground truth)
- `severity` (0–1)
- `health_state` (healthy / sick / critical / recovered / dead)
- `time_since_infection`
- `wealth`
- `location`

#### Behavior
- Decide when to visit hospital
- Choose hospital (distance vs waiting time)
- Respond to treatment
- Update health over time

---

### 👨‍⚕️ 2.2 DoctorAgent (LLM Agent 💡)

#### Attributes
- `specialization` (general / cardiology / etc.)
- `experience_level`
- `accuracy_rate`
- `fatigue`

#### Behavior
- Diagnose patients
- Recommend tests
- Prescribe treatment

#### 🧠 LLM Role
DoctorAgent uses LLM for reasoning:

**Input:**
- Symptoms
- Patient history
- Test results

**Output:**
- Diagnosis
- Confidence score
- Next step (tests / treatment)

---

### 🏥 2.3 HospitalAgent

#### Attributes
- `capacity` (beds)
- `doctors_available`
- `lab_capacity`
- `waiting_queue`
- `treatment_quality`

#### Behavior
- Admit patients
- Allocate doctors
- Manage queues
- Track outcomes

---

### 🧪 2.4 LabAgent (Optional)

#### Attributes
- `test_types`
- `accuracy`
- `processing_time`
- `cost`

#### Behavior
- Generate test results (with noise/uncertainty)

---

### 🧠 2.5 SpecialistAgent (Advanced)

- Cardiologist, Neurologist, etc.
- Invoked for complex cases

---

## 🌍 3. Environment

### Option 1: Grid (Mesa)
- City grid layout
- Hospitals at fixed positions
- Patients move across grid

### Option 2: GeoSpace (Mesa-Geo) ⭐
- Real hospital locations
- Real population density
- Spatial realism

---

## ⚙️ 4. Global Parameters

### 🏥 System Parameters
- `NUM_PATIENTS`
- `NUM_HOSPITALS`
- `NUM_DOCTORS`
- `HOSPITAL_CAPACITY`
- `LAB_CAPACITY`

### 🦠 Disease Parameters
- `DISEASE_LIST`
- `SYMPTOM_DISTRIBUTION`
- `SEVERITY_DISTRIBUTION`
- `RECOVERY_RATE`
- `MORTALITY_RATE`

### ⏱ Time Parameters
- `SIMULATION_STEPS`
- `TIME_PER_STEP` (hour/day)

### 🧠 LLM Parameters
- `LLM_TEMPERATURE`
- `LLM_ACCURACY_WEIGHT`
- `REASONING_DEPTH`
- `COST_PER_QUERY`

### 📊 Policy Parameters
- `TEST_COST`
- `TREATMENT_COST`
- `WAITING_TIME_THRESHOLD`

---

## 🔄 5. Interaction Flow

### 🧩 Simulation Steps

1. **Disease Onset**

patient.disease = random.choice(DISEASE_LIST)


2. **Decision Phase**

if severity > threshold:
go_to_hospital()


3. **Hospital Selection**
- Nearest OR least crowded

4. **Queue System**
- Waiting time increases

5. **Diagnosis**

DoctorAgent:
→ basic diagnosis OR
→ LLM reasoning
→ request tests


6. **Lab Testing**
- Adds uncertainty/noise

7. **Treatment Decision**
- Correct or incorrect

8. **Outcome**

if correct_treatment:
recover
else:
worsen / death


---

## 🧠 6. LLM Integration (USP 💥)

### 🔹 Basic Version
- LLM used as diagnosis function

### 🚀 Advanced Version (GSoC-Level)

Multi-agent reasoning system:


Patient → GP Agent → Specialist Agents → Consensus → Diagnosis


Features:
- Debate-based reasoning
- Reduced hallucination
- Collaborative diagnosis

---

## 📊 7. Metrics

- `diagnosis_accuracy`
- `average_waiting_time`
- `mortality_rate`
- `recovery_rate`
- `hospital_utilization`
- `llm_cost`

---

## 🧮 8. Key Formulas

### Diagnosis Accuracy
```bash
accuracy = correct_diagnosis / total_cases
Mortality Probability
P(death) = severity * (1 - treatment_effectiveness)
Waiting Time
waiting_time = queue_length / doctor_capacity
🧱 9. Architecture
MedicalDiagnosisModel
│
├── PatientAgent
├── DoctorAgent (LLM-enabled)
├── HospitalAgent
├── LabAgent
│
├── Scheduler (RandomActivation)
├── Space (Grid / Geo)
├── DataCollector
│
├── LLM Engine (OpenAI / Local Model)
🚀 10. Advanced Features (GSoC-Level Enhancements)
🔹 1. Misdiagnosis Learning
Doctors improve over time based on feedback
🔹 2. Adaptive Hospital Load
Overload increases mortality
🔹 3. Socio-economic Inequality
Poor patients delay treatment
🔹 4. Emergency System
Ambulance prioritization
🔹 5. Explainable AI
LLM outputs reasoning steps