import json
from datetime import datetime, date, timedelta
import calendar
from itertools import groupby


class Exercise:
    def __init__(self, exerciseDate, duration, exercise_type):
        self.exerciseDate = exerciseDate  # dd-mm-yyyy format
        self.duration = duration  # in minutes
        self.exercise_type = exercise_type
        self.weight = 70

    def calc_Calories(self, mets):
        self.calories = mets*self.weight*(self.duration/60)
        return self.calories


class Running(Exercise):
    def __init__(self, exerciseDate, duration, distance, route=""):
        super().__init__(exerciseDate, duration, "Running")
        self.distance = distance  # in km
        self.route = route
        self.pace = self.calculate_pace()
        self.calories = self.runningCalories()

    def calculate_pace(self):
        return self.duration/self.distance

    def runningCalories(self):
        if self.pace < 4:
            mets = 12
        elif self.pace < 5:
            mets = 10
        elif self.pace < 6:
            mets = 8
        else:
            mets = 6

        return self.calc_Calories(mets)

    def runningInfo(self):
        return f"""Date: {self.exerciseDate} | Duration: {self.duration} mins | Distance: {self.distance} km
Estimated Calories Burned: {self.calories} calories"""


class Swimming(Exercise):
    def __init__(self, exerciseDate, duration, laps, stroke_type, pool_length):
        super().__init__(exerciseDate, duration, "Swimming")
        self.laps = laps
        self.stroke_type = stroke_type
        self.pool_length = pool_length  # in meters
        self.calories = self.swimmingCalories()
        self.strokes = self.strokeType()

    def swimmingCalories(self):
        if self.stroke_type == 1:
            mets = 8
        elif self.stroke_type == 2:
            mets = 7
        elif self.stroke_type == 3:
            mets = 10
        elif self.stroke_type == 4:
            mets = 12
        else:
            mets = 8

        return self.calc_Calories(mets)

    def strokeType(self):
        if self.stroke_type == 1:
            strokes = "Freestyle"
        elif self.stroke_type == 2:
            strokes = "Backstroke"
        elif self.stroke_type == 3:
            strokes = "Breaststroke"
        elif self.stroke_type == 4:
            strokes = "Butterfly"
        else:
            strokes = "Mixed Strokes"

        return strokes

    def swimmingInfo(self):
        return f"""Date: {self.exerciseDate} | Duration: {self.duration} mins | Strokes: {self.strokes} | Pool Length: {self.pool_length} m
Estimated Calories Burned: {self.calories} calories"""


class Cycling(Exercise):
    def __init__(self, exerciseDate, duration, distance, speed, terrain):
        super().__init__(exerciseDate, duration, "Cycling")
        self.distance = distance  # in km
        self.terrain_type = terrain

        self.terraintype = self.terrainType()

        self.terrainMultiplier = self.terrain_multiplier()
        self.speed = speed
        self.speedmets = self.speedMet()
        self.calories = self.cyclingCalories()

    def speedMet(self):
        if self.speed >= 25:
            speedMets = 12
        elif self.speed >= 20:
            speedMets = 10
        elif self.speed >= 15:
            speedMets = 8
        else:
            speedMets = 6
        return speedMets

    def terrain_multiplier(self):
        if self.terrain_type == 1:
            terrainMultiplier = 1
        elif self.terrain_type == 2:
            terrainMultiplier = 1.3

        elif self.terrain_type == 3:
            terrainMultiplier = 1.5
        else:
            terrainMultiplier = 1.2

        return terrainMultiplier

    def terrainType(self):
        if self.terrain_type == 1:
            terrain = "Road"
        elif self.terrain_type == 2:
            terrain = "Hills"
        elif self.terrain_type == 3:
            terrain = "Mountains"
        else:
            terrain = "Mixed"

        return terrain

    def cyclingCalories(self):
        mets = (self.speedmets*self.terrainMultiplier)
        return self.calc_Calories(mets)

    def cyclingInfo(self):
        return f"""Date: {self.exerciseDate} | Duration: {self.duration} mins | Speed: {self.speed} km/h | Distance: {self.distance} m | Terrain: {self.terraintype}
Estimated Calories Burned: {self.calories} calories"""


def save_to_json(workouts, goals={}, filename="fitness-data.json"):
    workouts.sort(key=lambda x: datetime.strptime(x['date'], '%d-%m-%Y'))
    data = {
        "workouts": workouts,
        "goals": goals
    }
    with open(filename, "w") as final:
        json.dump(data, final, indent=4)


def load_from_json(filename="fitness-data.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data.get("workouts", []), data.get("goals", {})
    except FileNotFoundError:
        return [], {}


def todaysWorkout():
    workouts, goals = load_from_json()
    print("\n=== View Today's Workout ===")
    today = date.today()
    ftoday = today.strftime("%d-%m-%Y")
    weekday = today.strftime("%A")

    todays_workout = list(filter(lambda x: x['date'] == ftoday, workouts))

    totalWorkouts = len(todays_workout)
    totalDuration = sum([y['duration'] for y in todays_workout])
    totalCalories = sum([z['calories'] for z in todays_workout])

    for x in todays_workout:
        if x['id'] == 1:
            print(f"\n🏃 {weekday}, {ftoday} - {x['type']} ")
            print(
                f"Distance: {x['distance']} km | Duration: {x['duration']} min | Calories: {x['calories']}")
            print(f"Pace: {x['pace']} min/km | Route: {x['route']}")
        elif x['id'] == 2:
            print(f"\n🏊 {weekday}, {ftoday} - {x['type']} ")
            print(
                f"Duration: {x['duration']} min | Calories: {x['calories']}")
            print(f"Stroke: {x['stroke_type']} | Pool: {x['pool_length']}")
        else:
            print(f"\n🚴 {weekday}, {ftoday} - {x['type']} ")
            print(
                f"Distance: {x['distance']}m | Duration: {x['duration']} min | Calories: {x['calories']}")
            print(f"Speed: {x['speed']} | Terrain: {x['terrain']}")

    print(f"\nTotal: {totalWorkouts} Workouts | {totalDuration} minutes ({totalDuration/60} hours) | {totalCalories} calories ")


def weeklyWorkout():
    workouts, goals = load_from_json()
    weeklyData = []
    today = date.today()
    ftoday = today.strftime("%d-%m-%Y")

    for i in range(7):
        weekDate = today - timedelta(days=i)
        fweekDate = weekDate.strftime("%d-%m-%Y")
        weekWorkout = filter(lambda x: x['date'] == fweekDate, workouts)
        weeklyData.extend(list(weekWorkout))

    date1 = today-timedelta(days=6)
    fdate1 = date1.strftime("%d-%m-%Y")

    totalWeeklyWorkout = len(weeklyData)
    totalWeeklyDuration = sum([y['duration'] for y in weeklyData])
    totalWeeklyCalories = sum([y['calories'] for y in weeklyData])

    print(f"\n=== Weekly Summary ({fdate1} to {ftoday}) === ")
    print("\n📊 Overview:")
    print("\nTotal Workouts:", totalWeeklyWorkout)
    print(
        f"Total Duration: {totalWeeklyDuration} minutes ({totalWeeklyDuration/60} hours)")
    print("Total Calories burned:", totalWeeklyCalories)
    print(
        f"Average per workout: {totalWeeklyDuration/totalWeeklyWorkout} mins , {totalWeeklyCalories/totalWeeklyWorkout} calories")

    def breakdownExercise(exercise):
        weekExercise = list(
            filter(lambda x: x['id'] == exercise, weeklyData))
        exerciseNumbers = len(weekExercise)
        exerciseDuration = sum([y['duration'] for y in weekExercise])
        exerciseCalories = sum([y['calories'] for y in weekExercise])
        if exercise == 1:
            exerciseType = "Running"
        elif exercise == 2:
            exerciseType = "Swimming"
        else:
            exerciseType = "Cycling"

        print(
            f"{exerciseType} : {exerciseNumbers} workouts ({exerciseDuration} mins, {exerciseCalories} calories)")

    print("\n📈 Breakdown by Exercise: \n ")
    breakdownExercise(1)
    breakdownExercise(2)
    breakdownExercise(3)

    weekCalories = {}
    workout_days = []

    for item in weeklyData:
        x = item['date']
        calories = item['calories']
        if x in weekCalories:
            weekCalories[x] += calories
        else:
            weekCalories[x] = calories
            workout_days.append(x)

    maxDay = max(weekCalories)
    maxCalorieWeek = weekCalories[maxDay]
    fmaxDay = datetime.strptime(maxDay, "%d-%m-%Y").date()
    maxweekDay = fmaxDay.strftime('%A')

    print(f"\n🔥 Best Day: {maxweekDay}, {maxDay} ({maxCalorieWeek} calories)")
    workout_days.sort()
    weekDays = []
    for x in workout_days:
        workoutWeekDays = datetime.strptime(x, "%d-%m-%Y").date()
        fworkoutWeekDays = workoutWeekDays.strftime('%a')
        weekDays.append(fworkoutWeekDays)

    weekDayscomma = ''
    for x in weekDays:
        weekDayscomma += x + ', '

    print(f"📅 Workout Days: {weekDayscomma[:-2]}")

    def get_longest_streak(x):
        streakdays = []
        for y in x:
            day = int(y.split('-')[0])
            streakdays.append(day)

        max_streak = 1
        current_streak = 1
        for i in range(1, len(streakdays)):
            if streakdays[i] == streakdays[i-1]+1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)

            else:
                current_streak = 1

        return max_streak

    print("⚡ Current Streak:", get_longest_streak(workout_days))


def monthlyReport():
    workouts, goals = load_from_json()
    currentMonth = datetime.now().strftime('%B')
    currentYear = datetime.now().strftime('%Y')

    month = datetime.now().month
    year = datetime.now().year
    number_of_days = calendar.monthrange(year, month)[1]
    first_date = date(year, month, 1)

    allMonthDates = []
    monthlyWorkoutData = []
    monthData = []
    for i in range(number_of_days):
        z = (first_date + timedelta(days=i)).strftime('%d-%m-%Y')
        monthlyData = list(filter(lambda x: x['date'] == z, workouts))
        allMonthDates.append(z)
        if monthlyData:
            monthlyWorkoutData.append(monthlyData)
            monthData.extend(monthlyData)

    totalMonthlyWorkout = 0
    totalMonthlyDuration = 0
    totalMonthlyCalories = 0
    individualMonthlyData = []
    for x in monthlyWorkoutData:
        for y in x:
            totalMonthlyWorkout += 1
            totalMonthlyDuration += y['duration']
            totalMonthlyCalories += y['calories']
            individualMonthlyData.append(y)

    print(f"\n=== Monthly Report ({currentMonth} {currentYear}) ===")
    print("\n📊 Statistics:")
    print("Total Workouts: ", totalMonthlyWorkout)
    print(f"Total Duration: {totalMonthlyDuration/60} hours")
    print(f"Total Calories: {totalMonthlyCalories}")
    print(
        f"Average per workout: {totalMonthlyDuration/totalMonthlyWorkout} mins , {totalMonthlyCalories/totalMonthlyWorkout} calories")

    print(f"\n📈 Weekly Breakdown:")

    def weeklyBreakdown():

        def get_week_number(date_str):
            monthDate = datetime.strptime(date_str, '%d-%m-%Y')
            day = monthDate.day
            if 1 <= day <= 7:
                return 1
            elif 8 <= day <= 14:
                return 2
            elif 15 <= day <= 21:
                return 3
            elif 22 <= day <= 28:
                return 4
            elif 29 <= day <= 31:
                return 5
            return None

        weekData = {
            1: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
            2: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
            3: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
            4: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
            5: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0}
        }

        for x in individualMonthlyData:
            week = get_week_number(x['date'])
            if week:
                weekData[week]['totalWeekWorkouts'] += 1
                weekData[week]['totalWeekCalories'] += x['calories']

        test_date = date.today()
        res = calendar.monthrange(test_date.year, test_date.month)[1]

        print(
            f"Week 1 (01-07 {currentMonth}): {weekData[1]['totalWeekWorkouts']} workouts, {weekData[1]['totalWeekCalories']} calories")
        print(
            f"Week 2 (08-14 {currentMonth}): {weekData[2]['totalWeekWorkouts']} workouts, {weekData[2]['totalWeekCalories']} calories")
        print(
            f"Week 3 (15-21 {currentMonth}): {weekData[3]['totalWeekWorkouts']} workouts, {weekData[3]['totalWeekCalories']} calories")
        print(
            f"Week 4 (22-28 {currentMonth}): {weekData[4]['totalWeekWorkouts']} workouts, {weekData[4]['totalWeekCalories']} calories")
        print(
            f"Week 5 (29-{res} {currentMonth}): {weekData[5]['totalWeekWorkouts']} workouts, {weekData[5]['totalWeekCalories']} calories")

        print("\n🏆 Personal Records: ")

        maxRunningdist = [x for x in monthData if x['type'] == 'Running']
        longestRun = max(
            maxRunningdist, key=lambda x: x['distance'], default=None)

        maxSwimmingLaps = [x for x in monthData if x['type'] == 'Swimming']
        maxLaps = max(
            maxSwimmingLaps, key=lambda x: x['laps'], default=None)

        longestWorkoutDuration = max(
            monthData, key=lambda x: x['duration'], default=None)

        print(
            f"Longest run: {longestRun['distance']} km ({longestRun['date']})")

        print(
            f"Most laps: {maxLaps['laps']}  ({maxLaps['date']})")

        print(
            f"Longest Workout: {longestWorkoutDuration['duration']} min ({longestWorkoutDuration['date']})")

        dateCounts = [{keys: len(list(value))} for keys, value in groupby(
            monthData, lambda index: index['date'])]

        merged = {k: v for d in dateCounts for k, v in d.items()}
        max_value = max(merged.values())
        most_active_days = [k for k, v in merged.items() if v == max_value]

        activeDay = most_active_days[-1]

        fmostActiveDay = datetime.strptime(activeDay, "%d-%m-%Y").date()
        activeweekDay = fmostActiveDay.strftime('%A')

        print(
            f"\n📅 Most Active Day: {activeDay},{activeweekDay} ({max_value} workouts)")

        max_workouts = max(week['totalWeekWorkouts']
                           for week in weekData.values())

        most_active_weeks = [week_num for week_num, week in weekData.items(
        ) if week['totalWeekWorkouts'] == max_workouts]

        print(
            f"🔥 Best Week: Week {most_active_weeks[0]} ({max_workouts} workouts)")

    weeklyBreakdown()


def setGoals():

    workouts, goals = load_from_json("fitness-data.json")

    weeklyCalories = goals['weekly_calories']
    weeklyWorkout = goals['weekly_workouts']
    monthlyDistance = goals['monthly_distance']

    print("\n=== Goals Setting ===")
    print("\nCurrent Goals: ")
    print("Weekly Calorie Goal:", weeklyCalories, "calories")
    print("Weekly Workout Goal:", weeklyWorkout, "workouts")
    print("Monthly Distance Goal:", monthlyDistance, "km")

    print("\n1. Update Weekly Calorie Goal")
    print("2. Update Weekly Workout Goal")
    print("3. Update Monthly Distance Goal")
    print("4. Back to Main Menu")

    updateGoals = int(input("\nChoose Option: "))
    match updateGoals:
        case 1:
            print("\nCurrent weekly calorie goal: ", weeklyCalories)
            newCalories = int(input("Enter new weekly calorie goal:"))
            goals['weekly_calories'] = newCalories
            save_to_json(workouts, goals)
            print(
                f"✅ Weekly calorie goal updated to {newCalories} calories!")
        case 2:
            print("\nCurrent weekly workout goal: ", weeklyWorkout)
            newWorkout = int(input("Enter new weekly workout goal:"))
            goals['weekly_workouts'] = newWorkout
            save_to_json(workouts, goals)
            print(
                f"✅ Weekly workout goal updated to {newWorkout} workouts!")
        case 3:
            print("\nCurrent monthly distance goal: ", monthlyDistance)
            newDistance = int(input("Enter new monthly distance goal:"))
            goals['monthly_distance'] = newDistance
            save_to_json(workouts, goals)
            print(
                f"✅ Monthly distance goal updated to {newDistance} km !")


def viewProgress():
    workouts, goals = load_from_json()

    month = datetime.now().month
    year = datetime.now().year
    number_of_days = calendar.monthrange(year, month)[1]
    first_date = date(year, month, 1)

    allMonthDates = []
    monthlyWorkoutData = []
    monthData = []
    for i in range(number_of_days):
        z = (first_date + timedelta(days=i)).strftime('%d-%m-%Y')
        monthlyData = list(filter(lambda x: x['date'] == z, workouts))
        allMonthDates.append(z)
        if monthlyData:
            monthlyWorkoutData.append(monthlyData)
            monthData.extend(monthlyData)

    def get_week_number(date_str):
        monthDate = datetime.strptime(date_str, '%d-%m-%Y')
        day = monthDate.day
        if 1 <= day <= 7:
            return 1
        elif 8 <= day <= 14:
            return 2
        elif 15 <= day <= 21:
            return 3
        elif 22 <= day <= 28:
            return 4
        elif 29 <= day <= 31:
            return 5
        return None

    weekData = {
        1: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
        2: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
        3: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
        4: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0},
        5: {'totalWeekWorkouts': 0, 'totalWeekCalories': 0}
    }

    for x in monthData:
        week = get_week_number(x['date'])
        if week:
            weekData[week]['totalWeekWorkouts'] += 1
            weekData[week]['totalWeekCalories'] += x['calories']

    print("\n=== Progress Tracking ===")

    today = date.today()
    ftoday = today.strftime("%d-%m-%Y")
    weekday = today.weekday()

    start_date = today - timedelta(days=weekday + 1)
    end_date = start_date + timedelta(days=6)
    fstartdate = start_date.strftime("%d-%m-%Y")
    fenddate = end_date.strftime("%d-%m-%Y")

    print(f"\n🎯 Current Week Goals ({fstartdate} to {fenddate})")

    thisWeek = get_week_number(ftoday)
    percentage1 = weekData[thisWeek]['totalWeekCalories'] / \
        goals['weekly_calories']*100
    percentage2 = weekData[thisWeek]['totalWeekWorkouts'] / \
        goals['weekly_workouts']*100

    def icon(x):
        if x > 100:
            a = "✅"
        else:
            a = "🔄"
        return a
    bar_length = 20
    filled1 = int(bar_length * min(percentage1/100, 1.0))
    bar1 = '█' * filled1 + '░' * (bar_length - filled1)
    filled2 = int(bar_length * min(percentage2/100, 1.0))
    bar2 = '█' * filled2 + '░' * (bar_length - filled2)
    print(
        f"\nWeekly Calories: {weekData[thisWeek]['totalWeekCalories']}/{goals['weekly_calories']} ({percentage1:.2f} %) {icon(percentage1)}")
    print(bar1, f"{percentage1:.2f}%")
    print(
        f"\nWeekly Workout: {weekData[thisWeek]['totalWeekWorkouts']}/{goals['weekly_workouts']} ({percentage2:.2f} %) {icon(percentage2)}")
    print(bar2, f"{percentage2:.2f}%")

    print("\n🏆 Achievements This Week:")

    if weekData[thisWeek]['totalWeekWorkouts'] >= goals['weekly_workouts']:
        print(
            f"✅ 'Consistency King' - {weekData[thisWeek]['totalWeekWorkouts']} workouts completed")
    else:
        remainingWorkouts = goals['weekly_workouts'] - \
            weekData[thisWeek]['totalWeekWorkouts']
        print(f"🔄 'Consistency King' - {remainingWorkouts} workouts remaing")

    if weekData[thisWeek]['totalWeekCalories'] >= goals['weekly_calories']:
        print(
            f"✅ 'Calorie Crusher' - {weekData[thisWeek]['totalWeekCalories']} calories burned")
    else:
        remainingCalories = goals['weekly_calories'] - \
            weekData[thisWeek]['totalWeekCalories']
        print(f"🔄 'Calorie Crusher' - need {remainingCalories} more calories")

    print("\n📈 Trends:")

    lastWeek1 = weekData[thisWeek-1]['totalWeekWorkouts'] / \
        goals['weekly_workouts']*100
    diffWorkouts = percentage1-lastWeek1

    if diffWorkouts > 0:
        print(f"- Workout frequency: Up {diffWorkouts:.2f}% from last week")
    else:
        print(f"- Workout frequency: Down {diffWorkouts:.2f}% from last week")

    percentage3 = weekData[thisWeek]['totalWeekCalories'] / \
        weekData[thisWeek]['totalWeekWorkouts']

    lastweek3 = weekData[thisWeek-1]['totalWeekCalories'] / \
        weekData[thisWeek-1]['totalWeekWorkouts']
    diffCalories = percentage3-lastweek3

    if diffCalories > 0:
        print(
            f"- Average calories per workout: {percentage3:.2f} (↑{diffCalories:.2f})")
    else:
        print(
            f"- Average calories per workout: {percentage3:.2f} (↓{diffCalories:.2f})")

    currentMonthData = []
    today = date.today()
    ftoday = today.strftime("%d-%m-%Y")

    for i in range(number_of_days):
        weekDate = today - timedelta(days=i)
        fweekDate = weekDate.strftime("%d-%m-%Y")
        weekWorkout = filter(lambda x: x['date'] == fweekDate, workouts)
        currentMonthData.extend(list(weekWorkout))

    weekCalories = {}
    workout_days = []

    for item in currentMonthData:
        x = item['date']
        calories = item['calories']
        if x in weekCalories:
            weekCalories[x] += calories
        else:
            weekCalories[x] = calories
            workout_days.append(x)

    maxDay = max(weekCalories)
    maxCalorieWeek = weekCalories[maxDay]
    fmaxDay = datetime.strptime(maxDay, "%d-%m-%Y").date()
    maxweekDay = fmaxDay.strftime('%A')

    workout_days.sort()
    weekDays = []
    for x in workout_days:
        workoutWeekDays = datetime.strptime(x, "%d-%m-%Y").date()
        fworkoutWeekDays = workoutWeekDays.strftime('%a')
        weekDays.append(fworkoutWeekDays)

    weekDayscomma = ''
    for x in weekDays:
        weekDayscomma += x + ', '

    def get_longest_streak(x):
        streakdays = []
        for y in x:
            day = int(y.split('-')[0])
            streakdays.append(day)

        max_streak = 1
        current_streak = 1
        for i in range(1, len(streakdays)):
            if streakdays[i] == streakdays[i-1]+1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)

            else:
                current_streak = 1

        return max_streak

    print("- Longest Streak this month: ",
          get_longest_streak(workout_days), "days")


def exit():
    print("""\n=== FITNESS TRACKER ===

Data saved to fitness_data.json
Thanks for using Personal Fitness Tracker!
Keep up the great work! 💪

Program terminated.""")


def startProgram():
    workouts, goals = load_from_json()

    print("\n=== FITNESS TRACKER MENU ===")
    print("\n1. Add New Workout")
    print("2. View Today's Workout")
    print("3. Weekly Summary")
    print("4. Monthly Report")
    print("5. Set Goals")
    print("6. View Progress")
    print("7. Exit")
    option = int(input("\nChoose An Option:"))

    match option:
        case 1:
            print("\n=== Add New Workout ===")
            print("Exercise Types: ")
            print("1. Running")
            print("2. Swimming")
            print("3. Cycling")
            exerciseOption = int(input("\nEnter An Option: "))

            match exerciseOption:
                case 1:
                    print("\n=== Running Workout ===")
                    print("Date (in dd-mm-yyyy format) :")
                    exerciseDate = str(input())
                    print("Duration (in minutes) : ")
                    duration = int(input())
                    print("Distance (in km) : ")
                    distance = int(input())
                    print("Route : ")
                    route = str(input())
                    print("\n✅ Running workout saved successfully!")
                    running = Running(
                        exerciseDate, duration, distance, route)
                    workouts.append({
                                    "id": 1,
                                    "type": "Running",
                                    "date": exerciseDate,
                                    "duration": duration,
                                    "calories": running.calories,
                                    "distance": distance,
                                    "pace": running.pace,
                                    "route": route
                                    })
                    save_to_json(workouts)
                    print(running.runningInfo())
                case 2:
                    print("\n=== Swimming Workout ===")
                    print("Date (in dd-mm-yyyy format) :")
                    exerciseDate = str(input())
                    print("Duration (in minutes) : ")
                    duration = int(input())
                    print("Laps : ")
                    laps = int(input())
                    print("Pool Length: ")
                    pool_length = int(input())
                    print("Stroke Type:")
                    print("   1. Freestyle")
                    print("   2. Backstroke")
                    print("   3. Breaststroke")
                    print("   4. Butterfly")
                    print("   5. Mixed strokes")
                    stroke_type = int(input("\nEnter Stroke Type : "))
                    print("\n✅ Swimming workout saved successfully!")
                    swimming = Swimming(
                        exerciseDate, duration, laps, stroke_type, pool_length)
                    workouts.append({
                                    "id": 2,
                                    "type": "Swimming",
                                    "date": exerciseDate,
                                    "duration": duration,
                                    "calories": swimming.calories,
                                    "laps": laps,
                                    "stroke_type": swimming.strokes,
                                    "pool_length": pool_length
                                    })
                    save_to_json(workouts)
                    print(swimming.swimmingInfo())

                case 3:
                    print("\n=== Cycling Workout ====")
                    print("Date (in dd-mm-yyyy format) :")
                    exerciseDate = str(input())
                    print("Duration (in minutes) : ")
                    duration = int(input())
                    print("Distance (in km) : ")
                    distance = int(input())
                    print("Speed: ")
                    speed = int(input())
                    print("Terrain Type: ")
                    print("   1. Road")
                    print("   2. Hills")
                    print("   3. Mountain")
                    print("   4. Mixed")
                    terrain_type = int(input("\nEnter An Option: "))
                    print("\n✅ Swimming workout saved successfully!")
                    cycling = Cycling(
                        exerciseDate, duration, distance, speed, terrain_type)
                    workouts.append({
                                    "id": 3,
                                    "type": "Cycling",
                                    "date": exerciseDate,
                                    "duration": duration,
                                    "calories": cycling.calories,
                                    "distance": distance,
                                    "speed": speed,
                                    "terrain": cycling.terraintype
                                    })
                    save_to_json(workouts)
                    print(cycling.cyclingInfo())

        case 2:
            todaysWorkout()
        case 3:
            weeklyWorkout()

        case 4:
            monthlyReport()

        case 5:
            setGoals()

        case 6:
            viewProgress()

        case 7:
            exit()


startProgram()
