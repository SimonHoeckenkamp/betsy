import requests

from bs4 import BeautifulSoup
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from tabulate import tabulate

# TODO: substitute -1 values as nan values
def scrape_player_grades_for_teams(teams, saison, day):
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

def scrape_goals_for_teams(teams, saison, day):
    """
        scrapes kicker.de of corresponding day for goals of corresponding matchday

        input: 
            teams as list of teamnames
            saison as string for saison
            day as int for matchday

        output:
            goals as numpy array (teamsize, 4) (goals_1st_half, goals_2nd_half, counter_goals_1st_half, counter_goals_2nd_half)
    """

    goals = np.empty([len(teams), 4])

    source = requests.get("https://www.kicker.de/bundesliga/spieltag/" + saison + "/" + "{}".format(day)).text
    soup = BeautifulSoup(source, 'html.parser')

    #matches = soup.find_all("td", {"class": "kick__table--ranking__master kick__respt-m-w-65 kick__table--ranking__mark"})

    # get the matches
    for match in soup.find_all("div", {"class": "kick__v100-gameList__gameRow"}):

        match_text = match.prettify()

        # extract the information of each match (for two teams)
        # which team played against whom? save data in list
        match_teams = []
        match_goals = []
        for i in range(len(teams)):
            team_position = match_text.find(teams[i]) 
            if (team_position > 0):
                match_teams.append(i)
                match_teams.append(team_position)

        # extract the goals and add to list
        for goal in match.find_all("div", {"class": "kick__v100-scoreBoard__scoreHolder__score"}):
            match_goals.append(int(goal.text))

        # translate lists into numpy array
        # which team was first? insert the data to numpy array
        if (match_teams[1] < match_teams[3]):
            # home-playing team
            goals[match_teams[0], 0] = match_goals[2]
            goals[match_teams[0], 1] = match_goals[0]
            goals[match_teams[0], 2] = match_goals[3]
            goals[match_teams[0], 3] = match_goals[1]
            # outside-playing team
            goals[match_teams[2], 0] = match_goals[3]
            goals[match_teams[2], 1] = match_goals[1]
            goals[match_teams[2], 2] = match_goals[2]
            goals[match_teams[2], 3] = match_goals[0]
        else:
            # home-playing team
            goals[match_teams[2], 0] = match_goals[2]
            goals[match_teams[2], 1] = match_goals[0]
            goals[match_teams[2], 2] = match_goals[3]
            goals[match_teams[2], 3] = match_goals[1]
            # outside-playing team
            goals[match_teams[0], 0] = match_goals[3]
            goals[match_teams[0], 1] = match_goals[1]
            goals[match_teams[0], 2] = match_goals[2]
            goals[match_teams[0], 3] = match_goals[0]           

    np.savetxt(saison + "_{}_goals.csv".format(day), goals)

    return goals

# TODO: check for correct input (teams)
def scrape_matches(teams, saison, day):
    """
        scrapes kicker for matches for a specific match day

        input: 
            teams as list of teamnames (no control instance for right team constellation)
            saison as string for saison
            day as int for matchday

        output:
            matches as list of lists (team1, team2) as team indices
    """

    matches = []

    source = requests.get("https://www.kicker.de/bundesliga/spieltag/" + saison + "/" + "{}".format(day)).text
    soup = BeautifulSoup(source, 'html.parser')

    j = 0

    # get the matches
    for match in soup.find_all("div", {"class": "kick__v100-gameList__gameRow"}):

        match_text = match.prettify()

        # extract the information of each match (for two teams)
        # which team played against whom? save data in list
        match_teams = []
        for i in range(len(teams)):
            team_position = match_text.find(teams[i]) 
            if (team_position > 0):
                match_teams.append(i)
                match_teams.append(team_position)

        # translate lists into result list
        # which team was first? home-playing team
        if (match_teams[1] < match_teams[3]):
            matches.append([match_teams[0], match_teams[2]])
        else:
            matches.append([match_teams[2], match_teams[0]])

        j = j + 1

    return matches

# TODO: make model for all players (not only first 11, simplification)
# TODO: separate days (grades and goals)
def prepare_data(grades, goals, no_players, no_days):
    """
        gets player grades of multiple games (the more, the better) and calculates numpy array for X values and y values
        
        input: 
            grades: numpy array (team, players, matchday)
            no_players: int of represented best players in the model
            no_days: int of represented days in the model
        output: list of numpy arrays [X[no. days, no. players]: playergrades, y: mean team_grade, X_last: last data points for next matchday]
    """
    
    X_teams = grades.shape[0]
    X_days = grades.shape[2] - no_days
    X_goals = goals.shape[1]
    
    # no. of data points: X_teams*X_days
    # no. of features: no_days*no_players
    X = np.empty([no_days*no_players + no_days*X_goals, ])
    y = np.empty([1,])
    X_last = np.empty([X_teams, no_days*no_players + no_days*X_goals])

    i = 0
    for team in range(X_teams):
        for day in range(X_days):
            # data from player grades
            new_point = grades[team, 0:no_players, day:day+no_days].reshape((no_players*no_days))
            # data from goals
            new_point_goals = goals[team, :, day:day+no_days].reshape((X_goals*no_days))
            # combine the data sources
            new_point = np.hstack((new_point, new_point_goals))

            # classifier: +1:win, 0:even, -1:lost
            classifier = 0
            diff = goals[team, :, day+no_days][1] - goals[team, :, day+no_days][3]
            if (diff > 0):
                classifier = 1
            elif (diff < 0):
                classifier = -1
            new_target = np.array([classifier, ])
            
            # fill the first row
            if i == 0:
                X = new_point
                y = new_target
            else:
                X = np.vstack([X, new_point])
                y = np.vstack([y, new_target])

            i = i + 1

        if team == 0:
            X_last = np.hstack((
                grades[team, 0:no_players, X_days:X_days+no_days].reshape((no_players*no_days)),
                goals[team, :, X_days:X_days+no_days].reshape((X_goals*no_days))
                ))
        else:
            X_last = np.vstack([
                X_last, 
                np.hstack((
                    grades[team, 0:no_players, X_days:X_days+no_days].reshape((no_players*no_days)),
                    goals[team, :, X_days:X_days+no_days].reshape((X_goals*no_days))
                    ))
                ])

    return [X, y, X_last]

def loss(y_pred, y_test):
    """ 
        calcs the loss between predicted and test values
    """

    acc = 0
    i = 0
    for i in range(len(y_pred)):
        diff = y_pred[i] - y_test[i]
        if diff != 0:
            acc = acc + 1
    acc = acc / len(y_pred)
    return acc

def predict_match_day(clf, X, teams, matches):
    """
        recommends the bets for matchday 
    """

    # predict the outcomes
    y_pred = clf.predict(X)

    pred_matches = []
    for match in matches:
        pred_matches.append([teams[match[0]], y_pred[match[0]], teams[match[1]], y_pred[match[1]]])

    return pred_matches

if __name__ == "__main__":

    SAISON = "2021-22"
    next_day = 16

    #read teamnames from teams.csv
    teams = []
    with open("teams.csv", "r") as file:
        for team in file.readlines():
            teams.append(team.strip("\n"))

    scrape_player_grades_for_teams(teams, SAISON, next_day-1)
    print("grades scraped...")
    scrape_goals_for_teams(teams, SAISON, next_day-1)
    print("goals scraped...")

    # read existing grade files
    team_grades = np.loadtxt(SAISON + "_1_grades.csv")
    for day in range(2, next_day):
       day_grades = np.loadtxt(SAISON + "_{}_grades.csv".format(day))
       team_grades = np.dstack((team_grades, day_grades))

    # read existing goal files
    team_goals = np.loadtxt(SAISON + "_1_goals.csv")
    for day in range(2, next_day):
       day_goals = np.loadtxt(SAISON + "_{}_goals.csv".format(day))
       team_goals = np.dstack((team_goals, day_goals))


    players = [6]
    days = [4]

    res_loss = np.empty((len(players), len(days)))

    # use for loops for testing multiple settings    
    for i in range(len(players)):
        for j in range(len(days)):
            data = prepare_data(team_grades, team_goals, players[i], days[j])

            X = data[0]
            y = data[1]
            X_last = data[2]

            # dimensionality reduction
            #pca = PCA(n_components=20).fit(X)
            #X_PCA = pca.transform(X)

            # train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            # define the model
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            res_loss[i,j] = loss(y_pred, y_test)

            print("-----------------------------------------------------------------------------------------")
            print("players: {}, days: {}".format(players[i], days[j]))
            print("loss: {}".format(res_loss[i,j]))

            #train the model for bet recommendation
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y.ravel())
            # recommend next bets
            matches = scrape_matches(teams, SAISON, next_day)
            bets = predict_match_day(clf, X_last, teams, matches)

            #print the output
            results =[]
            for match in bets:
                if match[1] == -match[3]:
                    if match[1] > match[3]: 
                        results.append([match[0], match[2], match[0]])                        
                    elif match[1] < match[3]: 
                        results.append([match[0], match[2], match[2]])                        
                    else:
                        results.append([match[0], match[2], "even"]) 

            print(tabulate(results, headers=['team 1', 'team 2', 'winner']))
            print("-----------------------------------------------------------------------------------------")


    #np.savetxt("loss.csv", res_loss)