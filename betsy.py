import requests

from bs4 import BeautifulSoup
import numpy as np


def scrape_player_grades_for_team(teams, saison, day):
    """
        scrapes kicker.de for individual player grades of the corresponding match day 
        saves data in csv file and returns grades as numpy array

        input:
        team:   list of teamnames
        saison: string for saison (eg. )  
        day:    int of game-day

        output:
        returns an (len(teams), 14) numpy array
    """
    grades = -np.ones([len(teams), 14])

    i = 0
    for team in teams:
        source = requests.get("https://www.kicker.de/bundesliga/" + team + "/topspieler-spieltag/" + saison + "/" + "{}".format(day)).text
        soup = BeautifulSoup(source, 'html.parser')
        j = 0
        for grade in soup.find_all("td", {"class": "kick__table--ranking__master kick__respt-m-w-65 kick__table--ranking__mark"}):
            grades[i,j] = float(grade.get_text().replace(",", "."))
            j = j + 1
            if j == 14:
                break
        i = i + 1

    np.savetxt(saison + "_{}_grades.csv".format(day), grades)
    return grades

# TODO: refine model (e.g. use individual grades for individual players)
def predict_team_grades(grades, past_days, result_type="avg"):
    """
        predicts the team grades based on input array and number of represented days

        input:

        output:
        returns numpy array of shape (no. teams, no. grades) (no. grades = 1 for average)

    """

    if result_type != "avg":
        raise Exception("Result type unknown.")

    grades_pred = np.zeros([len(grades), ])



    return grades_pred

if __name__ == "__main__":

    SAISON = "2021-22"
    next_day = 15

    #read teamnames from teams.csv
    teams = []
    with open("teams.csv", "r") as file:
        for team in file.readlines():
            teams.append(team.strip("\n"))

    # read existing grade files
    team_grades = np.loadtxt(SAISON + "_1_grades.csv")
    for day in range(1, next_day):
        day_grades = np.loadtxt(SAISON + "_{}_grades.csv".format(day))
        team_grades = np.dstack((team_grades, day_grades))

    print(team_grades.shape)

    #for day in range(1,15):
    #    scrape_player_grades_for_team(teams, SAISON, day)
