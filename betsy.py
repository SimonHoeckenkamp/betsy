import requests

from bs4 import BeautifulSoup
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

    # check for input parameters
    if result_type != "avg":
        raise Exception("Result type unknown.")

    if past_days > grades.shape[2]:
        raise Exception("Not enough data points available.")

    avg_grades = np.average(grades, axis=1, weights=(grades > 0))

    grades_pred = np.zeros([len(grades), ])



    return grades_pred

# TODO: make model for all players (not only first 11, simplification)
def prepare_data(grades, no_players, no_days):
    """
        gets player grades of multiple games (the more, the better) and calculates numpy array for X values and y values
        
        input: 
            grades: numpy array (team, players, matchday)
            no_players: int of represented best players in the model
            no_days: int of represented days in the model
        output: tuple of numpy arrays (X[no. days, no. players]: playergrades, y: mean team_grade)
    """
    
    X_teams = grades.shape[0]
    X_days = grades.shape[2] - no_days
    
    # no. of data points: X_teams*X_days
    # no. of features: no_days*no_players
    X = np.empty([no_days*no_players, ])
    y = np.empty([1,])
    i = 0
    for team in range(X_teams):
        for day in range(X_days):
            new_point = grades[team, 0:no_players, day:day+no_days].reshape((no_players*no_days))
            new_target = np.mean(grades[team,0:11,day+no_days]).reshape((1))
            
            if i == 0:
                X = new_point
                y = new_target
                i = i + 1
                continue

            X = np.vstack([X, new_point])
            y = np.vstack([y, new_target])
            i = i + 1

    return (X, y)

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
    for day in range(2, next_day):
        day_grades = np.loadtxt(SAISON + "_{}_grades.csv".format(day))
        team_grades = np.dstack((team_grades, day_grades))

    X, y = prepare_data(team_grades, 11, 3)

    # run the principal component analysis on data set
    # TODO: plot the elbow graph for optimal numbers of PC's
    X_PCA = PCA(n_components=15).fit_transform(X)

    # run the train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.33, random_state=42)


    #predict_team_grades(team_grades, 3)

    print(team_grades)

    #for day in range(1,15):
    #    scrape_player_grades_for_team(teams, SAISON, day)
