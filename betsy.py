import requests

from bs4 import BeautifulSoup
import numpy as np


def get_playergrades_for_team(team, saison, day):
    """
        scrapes kicker.de for individual player grades of the corresponding match day 

        input
        team:   string of teamname
        saison: string for saison (eg. )  
        day:    int of game-day

        output
        returns an (11,) numpy array
    """

    source = requests.get("https://www.kicker.de/bundesliga/" + team + "/topspieler-spieltag/" + saison + "/" + "{}".format(day)).text
    soup = BeautifulSoup(source, 'html.parser')
    grades = -np.ones([14,])
    i = 0

    for grade in soup.find_all("td", {"class": "kick__table--ranking__master kick__respt-m-w-65 kick__table--ranking__mark"}):
        grades[i] = float(grade.get_text().replace(",", "."))
        i = i + 1
        if i == 14:
            break

    return grades

def save_grades_to_csv(grades):
    """"""

if __name__ == "__main__":
    #team = "borussia-dortmund"
    teams = [
        "fc-augsburg",
        "1-fc-union-berlin",
        "hertha-bsc",
        "arminia-bielefeld",
        "vfl-bochum",
        "borussia-dortmund",
        "eintracht-frankfurt",
        "sc-freiburg",
        "spvgg-greuther-fuerth",
        "tsg-hoffenheim",
        "1-fc-koeln",
        "rb-leipzig",
        "bayer-04-leverkusen",
        "1-fsc-mainz-05",
        "bor-moenchengladbach",
        "fc-bayern-muenchen",
        "vfb-stuttgart",
        "vfl-wolfsburg"
    ]

    saison = "2021-22"
    day = 14
    
    for team in teams:
        for grade in get_playergrades_for_team(team, saison, day):
            print(f"player: {grade}")