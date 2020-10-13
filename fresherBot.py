import csv
import requests
import aiml
import nltk
from tkinter import filedialog
from tkinter import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import IMGPredict
import transformerTest
import LunarLander



kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="fresherBot.aiml")

v = """
apples => {}
apricots => {}
avocados => }
bananas => {}
blackberries => {}
blueberries => {}
cherries => {}
coconuts => {}
figs => {}
grapefruits => {}
grapes => {}
lemons => {}
limes => {}
mangos => {}
oranges => {}
peaches => {}
pears => {}
pineapples => {}
plums => {}
pomegranites => {}
raspberries => {}
strawberries => {}
tomatoes => {}
watermelons => {}
orchard1 => f1
orchard2 => f2
orchard3 => f3
orchard4 => f4
be_in => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0

with open("qaPairs.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    questions = []
    answers = []
    for row in readCSV:
        question = (row[0])
        answer = (row[1])

        questions.append(question)
        answers.append(answer)


def qaResponse(user_input):
    vectorize = TfidfVectorizer()
    transform = vectorize.fit_transform(questions)

    similarity = cosine_similarity(transform, vectorize.transform([user_input])).flatten()
    bestMatch = similarity.argmax()
    print(answers[bestMatch])


def printWeather():
    owm = requests.get(
        "http://api.openweathermap.org/data/2.5/weather?q=Nottingham&units=metric&APPID"
        "=fb086ab2cf1e6b41015f07f9c5ccf687")
    weather_obj = owm.json()

    weather_conditions = weather_obj['weather']
    weather_description = str(weather_conditions[0])
    weather_description = weather_description.split("'main':", 1)[1]
    weather_description = weather_description.split(',', 1)[0].upper()

    weather_main = weather_obj['main']

    print('The weather in Nottingham is showing:', weather_description)
    print('With the following values:', weather_main)

    if weather_description == 'Rain' or weather_description == 'Drizzle':
        print('Bring your coat fresher!')


def identifyImage():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select image",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    answer = IMGPredict.predict(root.filename)
    root.destroy()
    print("That is most likely a:", answer, "\nIt is rich in vitamins and minerals!")
    print("\nAsk me something else...")


print("\n\nWelcome to the NTU freshers chatbot. I'm here to answer any queries you may have about starting off at NTU.")
print('\nYou can ask me to identify any images of fruit as part of our healthy eating program at NTU. Just type '
      '"image" into the chat.')
print("Also, feel free to ask me about today's weather!")

print("\nIf you would like to watch me master a game, type 'game'")
print('\nType "exit" at anytime to close')

while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    response = kern.respond(userInput.lower())

    if userInput == 'exit':
        break
    elif userInput == 'image':
        identifyImage()
    elif userInput == 'game':
        num_games = int(input("How many games should i play? "))
        while True:
            deleteWeights = input("Would you like to delete previous weights? (y/n) ")
            if deleteWeights == "y" or deleteWeights == "Y":
                LunarLander.play(num_games,True)
                break
            elif deleteWeights == "n" or deleteWeights == "N":
                LunarLander.play(num_games,False)
                break
            else:
                print("Invalid Input")

    else:
        if response[0] == '#':
            params = response[1:].split('$')
            cmd = int(params[0])
            if cmd == 1:
                qaResponse(userInput)
            elif cmd == 2:
                printWeather()
            elif cmd == 4:  # I will plant x in y
                o = 'o' + str(objectCounter)
                objectCounter += 1
                folval['o' + o] = o  # insert constant
                if len(folval[params[1]]) == 1:  # clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                folval[params[1]].add((o,))  # insert type of tree information
                if len(folval["be_in"]) == 1:  # clean up if necessary
                    if ('',) in folval["be_in"]:
                        folval["be_in"].clear()
                folval["be_in"].add((o, folval[params[2]]))  # insert location

            elif cmd == 5:  # Are there any x in y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2]
                result = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if result[2] == True:
                    print("Yes.")
                else:
                    print("No.")
            elif cmd == 6:  # Are all x in y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'all ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    print("Yes. ")
                else:
                    print("No. ")
            elif cmd == 7:  # Which plants are in ...
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                sat = m.satisfiers(e, "x", g)
                if len(sat) == 0:
                    print("None.")
                else:
                    # find satisfying objects in the valuation dictionary,
                    # and print their type names
                    sol = folval.values()
                    for so in sat:
                        for k, v in folval.items():
                            if len(v) > 0:
                                vl = list(v)
                                if len(vl[0]) == 1:
                                    for i in vl:
                                        if i[0] == so:
                                            print(k)
                                            break
            else:
                print(transformerTest.predict(userInput))
        else:
            print(response)
exit()
