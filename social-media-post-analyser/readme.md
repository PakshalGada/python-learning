# Task 3: Social Media Post Analyzer

## Topics Covered
- Strings
- Regular Expressions (RegEx)
- Lists
- Dictionaries
- Lambda Functions

## Task Implementation

### Step 1: Text Processing
- Process sample social media posts or text files
- Clean the text (remove punctuation, convert to lowercase)
- Split posts into individual words

### Step 2: Word Analysis
- Count how often each word appears using dictionaries
- Find the most common words
- Calculate basic statistics (total words, unique words, average post length)

### Step 3: Extract Special Elements
Use patterns to find:
- **Hashtags**: words starting with # (using regular expressions)
- **Mentions**: words starting with @ (using regular expressions)
- **Links**: words containing "http" or "www"

### Step 4: Data Processing
- Use lambda functions to sort and filter data
- Identify common patterns in text using string methods
- Generate simple statistics about the analyzed content

### Step 5: Generate Analysis Report
Create a comprehensive report showing all findings

## Expected Output

=== SOCIAL MEDIA ANALYSIS REPORT ===

📊 Basic Statistics:
Total Posts: 50
Total Words: 486
Unique Words: 312
Average Post Length: 9.7 words

🔥 Top 10 Most Common Words:
1. "new" - 8 times (1.6%)
2. "love" - 7 times (1.4%)
3. "perfect" - 6 times (1.2%)
4. "college" - 6 times (1.2%)
5. "excited" - 5 times (1.0%)
6. "food" - 4 times (0.8%)
7. "weekend" - 4 times (0.8%)
8. "coffee" - 4 times (0.8%)
9. "happy" - 3 times (0.6%)
10. "friends" - 3 times (0.6%)

#️⃣ Popular Hashtags:
1. #college - 8 times
2. #coffee - 4 times
3. #food - 4 times
4. #weekend - 4 times
5. #excited - 3 times

👥 Top Mentions:
1. @mom_official - 2 times
2. @sarah_jones - 1 time
3. @mike_chen - 1 time
4. @roommate_amy - 1 time

## Sample Data

python
posts = [
    "Just had the best #coffee ever! Starting my day right ☕ #morning #blessed",
    "Can't believe it's already Friday! #weekend #excited #finally",
    "Study session with @sarah_jones and @mike_chen #college #finals #stressed",
    "New movie was incredible! #marvel #cinema #amazing definitely recommend",
    "Pizza night with the squad! #foodie #friends #pizza #perfect",
    "Running late for class again... #college #life #struggle #help",
    "Beautiful sunset today! #nature #photography #peaceful #grateful",
    "Just finished my workout! Feeling strong 💪 #fitness #gym #motivation",
    "Rainy day = perfect for reading #books #cozy #rain #peaceful",
    "Can't decide what to wear today #fashion #style #choices #help",
    "Amazing concert last night! @band_name killed it #music #concert #live",
    "Homemade cookies turned out perfect! #baking #cookies #success #yummy",
    "Traffic is crazy today #commute #traffic #late #frustrated",
    "Love this new song! #music #newrelease #obsessed #repeat",
    "Study break with @roommate_amy #college #break #tired #coffee",
    "Best birthday surprise ever! Thank you @mom_official #birthday #family #love",
    "Trying to adult but failing #adult #life #struggle #help",
    "Perfect weather for a walk! #nature #walk #sunshine #happy",
    "New haircut! What do you think? #hair #change #nervous #opinion",
    "Midnight snack attack #food #midnight #hungry #guilty",
    "Group project meeting at 3pm @team_members #college #project #meeting",
    "Just bought new shoes! #shopping #shoes #retail #therapy",
    "Watching the game with @dad_jokes #sports #family #bonding #fun",
    "Procrastination level: expert #college #procrastination #guilty #help",
    "Farmers market haul! #healthy #food #fresh #weekend",
    "Can't sleep... too much #coffee today #insomnia #mistake #regret",
    "New episode of my favorite show! #tv #binge #excited #weekend",
    "Laundry day again... #adult #chores #boring #necessary",
    "Amazing Thai food tonight! #food #thai #delicious #satisfied",
    "Job interview tomorrow! #nervous #excited #career #future",
    "Beach day with @college_friends #beach #sun #vacation #perfect",
    "Trying to eat healthy but #pizza keeps calling #food #struggle #temptation",
    "New book arrived! #books #reading #excited #weekend",
    "Car broke down again... #car #trouble #frustrated #expensive",
    "Spontaneous road trip! #adventure #friends #spontaneous #excited",
    "Midterm grades are in... #college #grades #nervous #results",
    "Cooking dinner for the first time! #cooking #adult #experiment #wish",
    "Love this #weather! Perfect for outdoor activities #nature #outdoor #active",
    "Movie marathon night! #movies #marathon #popcorn #cozy",
    "New coffee shop discovery! #coffee #discovery #local #support",
    "Graduation is getting closer! #college #graduation #excited #nervous",
    "Roommate drama again... #roommate #drama #college #life",
    "Best #friends forever! Love you guys @bestie1 @bestie2 #friendship",
    "Trying to save money but #shopping keeps happening #money #struggle #budget",
    "Perfect study playlist! #music #study #focus #productivity",
    "Date night with @boyfriend_official #date #love #romantic #happy",
    "Family dinner tonight! #family #dinner #love #grateful",
    "New semester, new me! #college #semester #goals #motivation",
    "Late night gaming session! #gaming #night #fun #addicted",
    "Sunday brunch with @mom_official #brunch #family #sunday #tradition"
]

## Expected Outcomes

By completing this task, you will learn to:
- Process and analyze text data effectively
- Use regular expressions for pattern matching
- Work with dictionaries for data counting and storage
- Apply lambda functions for data manipulation
- Generate meaningful statistics from unstructured text data
- Extract insights from social media content patterns
