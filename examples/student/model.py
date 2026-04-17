from mesa import Model
from mesa.space import MultiGrid
import random

from examples.student.agent import SchoolAgent,StudentAgent


class SchoolModel(Model):

    def __init__(self, width=20, height=20, n_students=100, n_schools=10, seed=None):
        super().__init__(seed=seed)

        self.grid = MultiGrid(width, height, False)

        self.students = []
        self.schools = []

        # -----------------------------
        # 🧠 PARAMETERS (YOUR SET)
        # -----------------------------
        self.alpha_ach = 0.2
        self.beta_ach = 0.7
        self.sigma_ach = 0.5

        self.alpha_pass = 0.01
        self.alpha_fail = 0.05
        self.beta_fail = 0.1

        self.alpha_T = 5
        self.beta_T = 0.9
        self.sigma_T = 2

        self.beta_distance = -0.05
        self.beta_quality = 0.6
        self.beta_selectivity = 0.3
        self.beta_social = 0.4
        self.beta_ses = -0.2

        self.theta = -0.5

        self.beta_bc0 = 10
        self.beta_bc1 = 2
        self.sigma_bc = 5

        self.distance_threshold = 20

        self.pass_rate = 0.8

        # -----------------------------
        # 🏫 Schools
        # -----------------------------
        for i in range(n_schools):
            school = SchoolAgent(i, self)

            x = self.random.randrange(width)
            y = self.random.randrange(height)

            self.grid.place_agent(school, (x, y))
            school.pos = (x, y)

            self.schools.append(school)

        # -----------------------------
        # 🎓 Students
        # -----------------------------
        for i in range(n_students):
            student = StudentAgent(i + n_schools, self)

            x = self.random.randrange(width)
            y = self.random.randrange(height)

            self.grid.place_agent(student, (x, y))
            student.pos = (x, y)

            self.students.append(student)

    # -----------------------------
    # 🎯 PASS / FAIL
    # -----------------------------
    def pass_fail(self):

        grades = {}

        for s in self.students:
            grades.setdefault(s.grade, []).append(s)

        for group in grades.values():

            sorted_students = sorted(group, key=lambda s: s.achievement, reverse=True)

            cutoff = int(len(sorted_students) * self.pass_rate)

            for i, s in enumerate(sorted_students):
                s.passed = (i < cutoff)

    # -----------------------------
    # 🔁 MATCHING
    # -----------------------------
    def matching(self):

        for school in self.schools:
            school.students = []

        unassigned = [s for s in self.students if s.state != "DROPOUT"]

        while unassigned:

            applications = {school: [] for school in self.schools}

            for student in unassigned:
                school = student.choose_school()
                if school:
                    applications[school].append(student)

            new_unassigned = []

            for school, applicants in applications.items():

                selected = school.select_students(applicants)

                for s in selected:
                    s.current_school = school
                    school.students.append(s)

                rejected = [s for s in applicants if s not in selected]

                for s in rejected:
                    if school in s.choice_set:
                        s.choice_set.remove(school)
                    new_unassigned.append(s)

            if new_unassigned == unassigned:
                break

            unassigned = new_unassigned

    # -----------------------------
    # 🔁 STEP
    # -----------------------------
    def step(self):

        # 1. Tuition update (NEW)
        for school in self.schools:
            school.update_tuition()

        # 2. Achievement
        for s in self.students:
            s.update_achievement()

        # 3. Pass / Fail
        self.pass_fail()

        # 4. Dropout
        for s in self.students:
            s.apply_dropout()

        # 5. Social network
        for s in self.students:
            s.build_social_network()

        # 6. Choice set
        for s in self.students:
            s.build_choice_set()

        # 7. Utility
        for s in self.students:
            s.compute_utility()

        # 8. Matching
        self.matching()


# -----------------------------
# 🚀 RUN
# -----------------------------
if __name__ == "__main__":

    model = SchoolModel()

    for i in range(10):
        model.step()