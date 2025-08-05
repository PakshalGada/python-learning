import re
file = open('posts.txt', 'r')
read = file.readlines()
allPosts = []
allWords = []
uniqueWords = []
cleanWords = []
hashtags = []
mentions = []

for line in read:
    if line[-1] == "\n":
        allPosts.append(line[:-1])
    else:
        allPosts.append(line)


for i in allPosts:
    allWords.extend(i.split(" "))


hashtag_pattern = r'^#[A-Za-z0-9_]+$'
mention_pattern = r'^@[A-Za-z0-9_]+$'
for word in allWords:
    if re.match(hashtag_pattern, word):
        hashtags.append(word)
    elif re.match(mention_pattern, word):
        mentions.append(word)
    else:
        cleaned = re.sub(r'[^\w\s]', '', word).lower().strip()
        if cleaned:
            cleanWords.append(cleaned)


totalPosts = len(allPosts)
totalWords = len(allWords)

uniqueWords = list(set(allWords))
uniqueWord = len(uniqueWords)
averageLength = totalWords/totalPosts


def top10(items):
    rank = 0
    frequency = {}
    for item in items:
        if (item in frequency):
            frequency[item] += 1
        else:
            frequency[item] = 1

    ranking = {k: v for k, v in sorted(
        frequency.items(), key=lambda item: item[1], reverse=True)}
    first10 = dict([z for i, z in enumerate(ranking.items()) if i < 10])

    for z in first10:
        rank += 1
        percentage = ranking[z]/totalWords*100
        print(f"{rank}. {z}: {ranking[z]} ({percentage:.2f}%)")


print("\n=== SOCIAL MEDIA ANALYSIS REPORT ===")
print("\n📊 Basic Statistics: ")
print("Total Posts:", totalPosts)
print("Total Words:", totalWords)
print("Unique Words:", uniqueWord)
print("Average Post Length:", averageLength)

print("\n🔥 Top 10 Most Common Words:")
top10(cleanWords)

print("\n#️⃣ Popular Hashtags:")
top10(hashtags)

print("\n👥 Top Mentions:")
top10(mentions)
