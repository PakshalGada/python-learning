print("Enter number of student:")
number = input()


Studentslist = []
average = {}
grade_average = []


def calculate_percentage():
    percentage = float((Physics+Chemistry+Maths)/3.00)
    average[name] = percentage
    grade_average.append(percentage)


for i in range(int(number)):
    name = input("Enter Student Name: ")
    gradingsystem = []
    gradingsystem.append(name)
    for j in range(1):
        markslist = []
        Physics = int(input("Enter Physics Marks:"))
        markslist.append(Physics)
        Chemistry = int(input("Enter Chemistry Marks:"))
        markslist.append(Chemistry)
        Maths = int(input("Enter Maths Marks:"))
        markslist.append(Maths)
        gradingsystem.append(markslist)
        Studentslist.append(tuple(gradingsystem))
        calculate_percentage()


def give_grade():
    marks = float(average[x[0]])
    if 0 <= marks <= 100:
        if marks >= 90:
            grade = 'A'
        elif marks >= 80:
            grade = 'B'
        elif marks >= 70:
            grade = 'C'
        elif marks >= 60:
            grade = 'D'
        else:
            grade = 'F'
        print("Grade:", grade)


print(Studentslist)
print("\n========================================")
print("========================================")
print("\t\n STUDENT GRADING SYSTEM")
print("\n========================================")
print("========================================")

for x in Studentslist:
    print("\nStudent Name: ", x[0])
    print("Student Grade Average: ", average[x[0]])
    give_grade()
    print("\n---------------------------------------")


def topper_list():
    class_rank = 0
    ranking = {k: v for k, v in sorted(
        average.items(), key=lambda item: item[1], reverse=True)}
    for z in ranking:
        class_rank += 1
        print(f"\t{class_rank} :  {z}  |  {ranking[z]}")


highest_grade = max(grade_average)
lowest_grade = min(grade_average)

print("\n========================================")
print("\t     Topper List")
print("========================================")
topper_list()
print("\n---------------------------------------")
print("Highest Grade : ", highest_grade)
print("Lowest Grade : ", lowest_grade)
print("========================================")
print(Studentslist)
