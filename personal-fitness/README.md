

## Topics Covered

- Classes/Objects
- Inheritance
- Modules
- Dates (datetime module)
- JSON

## Task Implementation

### Step 1: Create Base Exercise Class

- Design a base `Exercise` class with common properties:
    - Duration (minutes)
    - Calories burned
    - Date of workout (dd-mm-yyyy format)
    - Exercise type
    - Notes (optional)
- Add methods for calculating calories per minute and displaying workout info

Set
### Step 2: Implement Specific Exercise Classes

Create specialized classes using inheritance:

- **Running**: Add distance, pace, route properties
- **Swimming**: Add laps, stroke type, pool length properties
- **Cycling**: Add distance, speed, terrain type properties
- Each class should override calorie calculation based on exercise type

### Step 3: JSON Data Management

- Create functions to save workout data to JSON file
- Load existing workout history from JSON file when program starts
- Handle file creation if JSON doesn't exist
- Implement backup and data integrity checks

### Step 4: Date and Time Management

- Use the datetime module with dd-mm-yyyy format
- Validate user date inputs and convert formats
- Calculate time between workouts and workout streaks
- Sort and filter workouts by date ranges

### Step 5: Interactive Menu System

- Create user-friendly command-line interface
- Implement input validation for all user entries
- Add error handling for invalid selections
- Provide clear navigation between different features

### Step 6: Statistics and Analytics

- Calculate weekly and monthly fitness statistics
- Track total calories burned, workout frequency
- Analyze performance trends over time
- Generate progress reports from JSON data

### Step 7: Goal Setting and Progress Tracking

- Set fitness goals (weekly calories, workout frequency, distance targets)
- Track progress toward goals with percentage completion
- Send motivational messages and achievements
- Save goals in JSON file for persistence

## Menu Options and Expected Outputs

### Main Menu Interface

```
=== FITNESS TRACKER MENU ===
1. Add New Workout
2. View Recent Workouts  
3. Weekly Summary
4. Monthly Report
5. Set Goals
6. View Progress
7. Exit

Choose an option:
```

### Option 1: Add New Workout

```
=== Add New Workout ===
Exercise Types:
1. Running
2. Swimming  
3. Cycling
Choose exercise type: 1

=== Running Workout ===
Date (dd-mm-yyyy, or press Enter for today): 09-06-2025
Distance (km): 5.2
Duration (minutes): 28
Route (optional): Park Loop

✅ Running workout saved successfully!
Date: 09-06-2025 | Duration: 28 mins | Distance: 5.2 km
Estimated calories burned: 312
```

### Option 2: View Recent Workouts

```
=== RECENT WORKOUTS (Last 7 days) ===

🏃 Monday, 09-06-2025 - Running
   Distance: 5.2 km | Duration: 28 mins | Calories: 312
   Pace: 5:23 min/km | Route: Park Loop

🏊 Sunday, 08-06-2025 - Swimming  
   Laps: 40 | Duration: 45 mins | Calories: 398
   Stroke: Freestyle | Pool: 25m

🚴 Friday, 06-06-2025 - Cycling
   Distance: 15.8 km | Duration: 52 mins | Calories: 445
   Speed: 18.2 km/h | Terrain: Mixed

Total: 3 workouts | 125 minutes | 1,155 calories
```

### Option 3: Weekly Summary

```
=== WEEKLY SUMMARY (03-06-2025 to 09-06-2025) ===

📊 Overview:
Total Workouts: 5
Total Duration: 183 minutes (3.05 hours)
Total Calories: 1,847
Average per workout: 37 mins, 369 calories

📈 Breakdown by Exercise:
Running: 2 workouts (50 mins, 590 calories)
Swimming: 2 workouts (83 mins, 812 calories)  
Cycling: 1 workout (52 mins, 445 calories)

🔥 Best Day: Friday, 06-06-2025 (445 calories)
📅 Workout Days: Mon, Wed, Fri, Sun, Mon
⚡ Current Streak: 2 days
```

### Option 4: Monthly Report

```
=== MONTHLY REPORT (June 2025) ===

📊 Statistics:
Total Workouts: 12
Total Duration: 8.2 hours
Total Calories: 4,234
Average per workout: 41 mins, 353 calories

📈 Weekly Breakdown:
Week 1 (01-07 June): 3 workouts, 1,245 calories
Week 2 (08-14 June): 5 workouts, 1,847 calories
Week 3 (15-21 June): 4 workouts, 1,142 calories

🏆 Personal Records:
Longest run: 8.5 km (15-06-2025)
Most laps: 50 laps (12-06-2025)
Longest workout: 65 minutes (18-06-2025)

📅 Most Active Day: Monday (4 workouts)
🔥 Best Week: Week 2 (5 workouts)
```

### Option 5: Set Goals

```
=== GOAL SETTING ===

Current Goals:
Weekly Calorie Goal: 2,000 calories
Weekly Workout Goal: 4 workouts
Monthly Distance Goal: 100 km

1. Update Weekly Calorie Goal
2. Update Weekly Workout Goal  
3. Update Monthly Distance Goal
4. Back to Main Menu

Choose option: 1

Current weekly calorie goal: 2,000
Enter new weekly calorie goal: 2,500

✅ Weekly calorie goal updated to 2,500 calories!
```

### Option 6: View Progress

```
=== PROGRESS TRACKING ===

🎯 Current Week Goals (03-06-2025 to 09-06-2025):
┌─────────────────────────────────────────────┐
│ Weekly Calories: 1,847/2,500 (74%) 🔄       │
│ ████████████████░░░░░░               74%    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Weekly Workouts: 5/4 (125%) ✅              │
│ ████████████████████████████████   125%     │
└─────────────────────────────────────────────┘

🏆 Achievements This Week:
✅ "Consistency King" - 5 workouts completed
✅ "Early Bird" - 3 morning workouts
🔄 "Calorie Crusher" - Need 153 more calories

📈 Trends:
- Workout frequency: Up 25% from last week
- Average calories per workout: 369 (↑12%)
- Longest streak this month: 5 days
```

### Option 7: Exit

```
=== FITNESS TRACKER ===

Data saved to fitness_data.json
Thanks for using Personal Fitness Tracker!
Keep up the great work! 💪

Program terminated.
```

```
=== PERSONAL FITNESS TRACKER ===

📊 Weekly Summary (June 3-9, 2025):
Total Workouts: 5
Total Duration: 320 minutes (5.3 hours)
Total Calories: 1,847 calories
Average Workout: 64 minutes, 369 calories

🏃 Workout History:
1. Monday, June 9 - Running
   Distance: 5.2 km | Duration: 28 mins | Calories: 312
   Pace: 5:23 min/km | Route: Park Loop

2. Sunday, June 8 - Swimming  
   Laps: 40 | Duration: 45 mins | Calories: 398
   Stroke: Freestyle | Pool: 25m

3. Friday, June 6 - Cycling
   Distance: 15.8 km | Duration: 52 mins | Calories: 445
   Speed: 18.2 km/h | Terrain: Mixed

4. Wednesday, June 4 - Running
   Distance: 3.1 km | Duration: 22 mins | Calories: 278
   Pace: 7:06 min/km | Route: Neighborhood

5. Monday, June 3 - Swimming
   Laps: 30 | Duration: 38 mins | Calories: 414
   Stroke: Mixed | Pool: 25m

🎯 Goal Progress:
Weekly Calorie Goal: 1,847/2,000 calories (92% complete) ✅
Weekly Workout Goal: 5/4 workouts (125% complete) ✅
Monthly Distance Goal: 47.2/100 km (47% complete) 🔄

📈 Performance Trends:
- Average calories/workout increased by 12% this week
- Workout frequency: 5 days (up from 3 last week)
- Current streak: 3 consecutive workout days
- Longest streak this month: 5 days

🏆 Achievements Unlocked:
- "Consistency King" - 5 workouts in one week
- "Calorie Crusher" - Burned 1,800+ calories in a week
- "Early Bird" - 3 morning workouts completed

⚡ Quick Stats:
Favorite Exercise: Swimming (45% of workouts)
Most Productive Day: Friday (avg 445 calories)
Best Month: May 2025 (18 workouts, 6,247 calories)
Total Workouts All Time: 127
Total Calories Burned: 34,892
```

## Sample Class Structure

python

```python
# Base Exercise Class
class Exercise:
    def __init__(self, date, duration, exercise_type, notes=""):
        self.date = date  # dd-mm-yyyy format
        self.duration = duration  # in minutes
        self.exercise_type = exercise_type
        self.notes = notes
        self.calories = 0

# Inheritance Examples
class Running(Exercise):
    def __init__(self, date, duration, distance, route=""):
        super().__init__(date, duration, "Running")
        self.distance = distance  # in km
        self.route = route
        self.pace = self.calculate_pace()

class Swimming(Exercise):
    def __init__(self, date, duration, laps, stroke_type, pool_length=25):
        super().__init__(date, duration, "Swimming")
        self.laps = laps
        self.stroke_type = stroke_type
        self.pool_length = pool_length  # in meters

class Cycling(Exercise):
    def __init__(self, date, duration, distance, terrain="Road"):
        super().__init__(date, duration, "Cycling")
        self.distance = distance  # in km
        self.terrain = terrain
        self.speed = self.calculate_speed()

# JSON Functions
def save_to_json(workouts, goals, filename="fitness_data.json"):
    """Save workout data and goals to JSON file"""
    
def load_from_json(filename="fitness_data.json"):
    """Load workout data from JSON file"""
```

## Expected Outcomes

By completing this task, you will learn to:

- Design and implement object-oriented programs with classes and inheritance
- Work with Python's datetime module for time-based calculations
- Store and retrieve data using JSON format
- Create modular code with proper class hierarchies
- Build practical applications with real-world functionality
- Implement data analysis and reporting features

## Base Formula
**METs × Body Weight (kg) × Duration (hours) = Calories Burned**

METs (Metabolic Equivalent of Task) = How many times more energy an activity uses compared to sitting quietly

## Running Formula

**Calories = METs × Weight × (Duration ÷ 60)**

**METs based on pace:**
- Very Fast (under 4 min/km): 12 METs
- Fast (4-5 min/km): 10 METs  
- Moderate (5-6 min/km): 8 METs
- Slow (over 6 min/km): 6 METs

**Pace calculation:** Duration (minutes) ÷ Distance (km) = Minutes per km

## Swimming Formula

**Calories = METs × Weight × (Duration ÷ 60)**

**METs based on stroke type:**
- Freestyle: 8 METs
- Backstroke: 7 METs
- Breaststroke: 10 METs
- Butterfly: 12 METs
- Mixed strokes: 8 METs

## Cycling Formula

**Calories = (METs × Terrain Multiplier) × Weight × (Duration ÷ 60)**

**METs based on speed:**
- Very Fast (25+ km/h): 12 METs
- Fast (20-24 km/h): 10 METs
- Moderate (15-19 km/h): 8 METs
- Leisurely (under 15 km/h): 6 METs

**Speed calculation:** (Distance × 60) ÷ Duration = km/h

**Terrain multipliers:**
- Road: 1.0 (no change)
- Hills: 1.3 (30% more effort)
- Mountain: 1.5 (50% more effort)
- Mixed: 1.2 (20% more effort)

## Example Calculations

**Running Example:**
- 30 minutes, 5km, 70kg person
- Pace: 30 ÷ 5 = 6 min/km (Moderate = 8 METs)
- Calories: 8 × 70 × (30 ÷ 60) = 8 × 70 × 0.5 = 280 calories

**Swimming Example:**
- 45 minutes, Freestyle, 70kg person
- METs: 8 (Freestyle)
- Calories: 8 × 70 × (45 ÷ 60) = 8 × 70 × 0.75 = 420 calories

**Cycling Example:**
- 60 minutes, 20km, Hills, 70kg person
- Speed: (20 × 60) ÷ 60 = 20 km/h (Fast = 10 METs)
- Terrain: Hills (1.3 multiplier)
- Calories: (10 × 1.3) × 70 × (60 ÷ 60) = 13 × 70 × 1 = 910 calories


{
  "goals": {
    "weekly_calories": 2000,
    "weekly_workouts": 4,
    "monthly_distance": 100
  },
  "workouts": [
    {
      "id": 1,
      "type": "Running",
      "date": "09-06-2025",
      "duration": 28,
      "calories": 312,
      "distance": 5.2,
      "pace": "5:23",
      "route": "Park Loop"
    },
    {
      "id": 2,
      "type": "Swimming",
      "date": "08-06-2025",
      "duration": 45,
      "calories": 398,
      "laps": 40,
      "stroke_type": "Freestyle",
      "pool_length": 25
    },
    {
      "id": 3,
      "type": "Cycling",
      "date": "06-06-2025",
      "duration": 52,
      "calories": 445,
      "distance": 15.8,
      "speed": 18.2,
      "terrain": "Mixed"
    }
  ]
}