import mesa
import random
import math
from enum import Enum
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

student_tool_manager = ToolManager()
school_tool_manager = ToolManager()


class Student_state(Enum):
    Enrolled = "ENROLLED"
    APPLIED = "APLIED"
    DROPOUT = "DROPOUT"
    GRADUATE = "GRADUATE"

# -----------------------------
# 🎓 STUDENT AGENT
# -----------------------------
class StudentAgent(LLMAgent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.pos = None

        # --- Core ---
        self.ses = random.random()
        self.achievement = random.uniform(-1, 1)
        self.grade = random.randint(1, 12)

        self.state = "ENROLLED"
        self.passed = True

        # --- Budget (BCᵢ) ---
        self.budget = max(
            0,
            model.beta_bc0
            + model.beta_bc1 * self.ses
            + random.normalvariate(0, model.sigma_bc),
        )

        # --- School ---
        self.current_school = None
        self.choice_set = []
        self.utility_scores = {}
        self.social_network = []
        self.memory = STLTMemory(
            model=self,
            llm_model="ollama/llama3.1:latest",
            display=True
        )
        self.tool_manager = student_tool_manager

    # -----------------------------
    # 📈 Achievement Update
    # -----------------------------
    def update_achievement(self):
        noise = random.normalvariate(0, self.model.sigma_ach)

        school_effect = 0
        if self.current_school:
            school_effect = self.current_school.value_added

        self.achievement = (
            self.model.alpha_ach
            + self.model.beta_ach * self.achievement
            + school_effect
            + noise
        )

        self.achievement = max(-4, min(4, self.achievement))

    # -----------------------------
    # 🚪 Dropout
    # -----------------------------
    def apply_dropout(self):

        if self.state == "DROPOUT":
            return

        if self.passed:
            p = self.model.alpha_pass
        else:
            p = self.model.alpha_fail * math.exp(
                self.model.beta_fail * self.grade
            )

        if random.random() < p:
            self.state = "DROPOUT"

    # -----------------------------
    # 🌐 Social Network
    # -----------------------------
    def build_social_network(self):

        self.social_network = []

        for other in self.model.students:

            if other == self:
                continue

            ses_diff = abs(self.ses - other.ses)

            p = math.exp(self.model.theta * ses_diff) / (
                1 + math.exp(self.model.theta * ses_diff)
            )

            if random.random() < p:
                self.social_network.append(other)

    # -----------------------------
    # 🔍 Choice Set (WITH BUDGET)
    # -----------------------------
    def build_choice_set(self):

        self.choice_set = []

        for school in self.model.schools:

            dx = self.pos[0] - school.pos[0]
            dy = self.pos[1] - school.pos[1]
            dist = math.sqrt(dx**2 + dy**2)

            # ❗ Budget constraint (IMPORTANT)
            if school.tuition > self.budget:
                continue

            if dist < self.model.distance_threshold:
                self.choice_set.append(school)
            elif random.random() < school.visibility_prob:
                self.choice_set.append(school)

    # -----------------------------
    # 🧠 Utility Function (FULL)
    # -----------------------------
    def compute_utility(self):

        self.utility_scores = {}

        for school in self.choice_set:

            dx = self.pos[0] - school.pos[0]
            dy = self.pos[1] - school.pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            social_signal = sum(
                1 for f in self.social_network if f.current_school == school
            )

            ses_diff = abs(self.ses - school.avg_ses)

            V = (
                self.model.beta_distance * distance
                + self.model.beta_quality * school.mean_achievement
                + self.model.beta_selectivity * int(school.selective)
                + self.model.beta_social * social_signal
                + self.model.beta_ses * ses_diff
            )

            self.utility_scores[school] = V

    # -----------------------------
    # 🎯 Softmax Choice
    # -----------------------------
    def choose_school(self):

        if not self.utility_scores:
            return None

        schools = list(self.utility_scores.keys())
        values = list(self.utility_scores.values())

        exp_vals = [math.exp(v) for v in values]
        total = sum(exp_vals)

        probs = [v / total for v in exp_vals]

        return random.choices(schools, weights=probs, k=1)[0]
    def step(self):
        prompt = f"""
        You are a student choosing schools in a competitive education system.

        You consider:
        - distance to school
        - achivement {self.achievement}
        - budget constraint {self.budget}
        - Utility score {self.utility_scores}
        - social network {self.social_network}
        - whether your friends attend the school
        - how similar the school is to your background (SES)

        You must:
        - choose a school from your available choice set
        - prefer better schools but consider cost and distance
        - update your decision based on past outcomes
        You can use tools like speak_to,move_one_step

        If you fail repeatedly, you may drop out.
        If you complete grade 12, you graduate.

        """
        observation = self.generate_obs()
        plan = self.reasoning.plan(
            obs = observation,
            selected_tools=None
        )
        self.aapply_plan(plan)



# -----------------------------
# 🏫 SCHOOL AGENT
# -----------------------------
class SchoolAgent(LLMAgent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.pos = None
        self.capacity = random.randint(20, 50)
        self.students = []

        self.selective = random.choice([True, False])
        self.value_added = random.uniform(0, 1)
        self.mean_achievement = random.uniform(0, 1)

        self.visibility_prob = 0.3
        self.avg_ses = random.random()

        # --- Tuition ---
        self.tuition = random.uniform(10, 50)

        self.memory = STLTMemory(
            model=self,
            llm_model="ollama/llama3.1:latest",
            display=True
        )
        self.tool_manager = school_tool_manager

    # -----------------------------
    # 💰 Tuition Update (FORMULA)
    # -----------------------------
    def update_tuition(self):

        noise = random.normalvariate(0, self.model.sigma_T)

        self.tuition = (
            self.model.alpha_T
            + self.model.beta_T * self.tuition
            + noise
        )

        self.tuition = max(0, self.tuition)

    # -----------------------------
    # 🎯 Selection
    # -----------------------------
    def select_students(self, applicants):

        if len(applicants) <= self.capacity:
            return applicants

        if self.selective:
            weights = [a.achievement + 4 for a in applicants]
            return random.choices(applicants, weights=weights,  k=self.capacity)

        return random.sample(applicants, self.capacity)
    def step(self):
        prompt = ""
        observation = self.generate_obs()
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[]
        )
    

