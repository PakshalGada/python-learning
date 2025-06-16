print("Enter your monthly salary:")
salary = input()
print("Enter your monthly rent:")
rent = input()
print("Enter your monthly food expenses:")
food = input()
print("Enter your entertainment expenses:")
entertainment = input()
budget = {
    "salary": float(salary),
    "rent": float(rent),
    "food": float(food),
    "entertainment": float(entertainment)
}

expense = budget["rent"]+budget["food"]+budget["entertainment"]
savings = budget["salary"]-expense
rentper = budget["rent"]/budget["salary"]*100
foodper = budget["food"]/budget["salary"]*100
entertainmentper = budget["entertainment"]/budget["salary"]*100

print("\nPersonal Budget Tracker")
print("======================================================================")
print("Income:", budget["salary"])
print("Total Expense :", expense)
print("----------------------------------------------------------------")
print("Money left :", savings)
print("----------------------------------------------------------------")
print(f"Rent is your {rentper:.2f}% of your salary.")
print(f"Food is your {foodper:.2f}% of your salary.")
print(f"Entertainment is your {entertainmentper:.2f}% of your salary.")
print("======================================================================")
