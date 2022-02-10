from pandas.core.frame import DataFrame
import requests
import os
import csv

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

# TODO: substitute -1 values as nan values
def scrape_and_save_player_grades_for_teams(league, teams, saison, day):
    """
        scrapes kicker.de for individual player grades of the corresponding match day 
        saves data in csv file within corresponding directory

        input:
        league: string of league
        team:   list of teamnames
        saison: string for saison (eg. )  
        day:    int of game-day

        output: bool: True if data scraped from web
    """

    # test if directories/ files exist
    if not os.path.isdir(league):
        os.mkdir(league)
    if not os.path.isdir(league + "/" + saison):
        os.mkdir(league + "/" + saison)

    # if file exists do not run the webscraper
    if os.path.isfile(league + "/" + saison + "/{}_grades.csv".format(day)):
        return False

    grades = -np.ones([len(teams), 14])

    i = 0
    for team in teams:
        source = requests.get("https://www.kicker.de/" + league + "/" + team + "/topspieler-spieltag/" + saison + "/" + "{}".format(day)).text
        soup = BeautifulSoup(source, 'html.parser')
        j = 0
        for grade in soup.find_all("td", {"class": "kick__table--ranking__master kick__respt-m-w-65 kick__table--ranking__mark"}):
            grades[i,j] = float(grade.get_text().replace(",", "."))
            j = j + 1
            if j == 14:
                break
        i = i + 1

    np.savetxt(league + "/" + saison + "/{}_grades.csv".format(day), grades)
    return True

def scrape_and_save_goals_and_home_match_for_teams(league, teams, saison, day):
    """
        scrapes kicker.de of corresponding day for goals of corresponding matchday
        and which team had a home match
        saves the data in csv file within corresponding directory

        input: 
            league as string for league
            teams as list of teamnames
            saison as string for saison
            day as int for matchday

        output: bool: True if data scraped from web
    """

    # test if directories/ files exist
    if not os.path.isdir(league):
        os.mkdir(league)
    if not os.path.isdir(league + "/" + saison):
        os.mkdir(league + "/" + saison)

    # if file exists do not run the webscraper
    if os.path.isfile(league + "/" + saison + "/{}_goals.csv".format(day)) \
        and os.path.isfile(league + "/" + saison + "/{}_homematch.csv".format(day)):
        return False

    goals = np.empty([len(teams), 4])
    home_match = np.empty([len(teams), 1])

    source = requests.get("https://www.kicker.de/" + league + "/spieltag/" + saison + "/" + "{}".format(day)).text
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
            home_match[match_teams[0]] = 1
            # outside-playing team
            goals[match_teams[2], 0] = match_goals[3]
            goals[match_teams[2], 1] = match_goals[1]
            goals[match_teams[2], 2] = match_goals[2]
            goals[match_teams[2], 3] = match_goals[0]
            home_match[match_teams[2]] = 0
        else:
            # home-playing team
            goals[match_teams[2], 0] = match_goals[2]
            goals[match_teams[2], 1] = match_goals[0]
            goals[match_teams[2], 2] = match_goals[3]
            goals[match_teams[2], 3] = match_goals[1]
            home_match[match_teams[2]] = 1
            # outside-playing team
            goals[match_teams[0], 0] = match_goals[3]
            goals[match_teams[0], 1] = match_goals[1]
            goals[match_teams[0], 2] = match_goals[2]
            goals[match_teams[0], 3] = match_goals[0]           
            home_match[match_teams[0]] = 0

    np.savetxt(league + "/" + saison + "/{}_goals.csv".format(day), goals)
    np.savetxt(league + "/" + saison + "/{}_homematch.csv".format(day), home_match)

    return True

# TODO: check for correct input (teams)
def scrape_matches(league, teams, saison, day):
    """
        scrapes kicker for matches for a specific match day (competing teams and home-match-teams)

        input: 
            league as string for corresponding league
            teams as list of teamnames (no control instance for right team constellation)
            saison as string for saison
            day as int for matchday

        output:
            matches: as list of lists (team1, team2) as team indices
            home_matches: numpy array (len(teams), ) shows if a team plays at home or outside
    """

    matches = []

    source = requests.get("https://www.kicker.de/" + league + "/spieltag/" + saison + "/" + "{}".format(day)).text
    soup = BeautifulSoup(source, 'html.parser')

    j = 0

    home_match = np.empty((len(teams), ))

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
            home_match[match_teams[0]] = 1
            home_match[match_teams[2]] = 0
        else:
            matches.append([match_teams[2], match_teams[0]])
            home_match[match_teams[0]] = 0
            home_match[match_teams[2]] = 1

        j = j + 1

    return (matches, home_match)

# TODO: make model for all players (not only first 11, simplification)
# TODO: separate days (grades and goals)
def prepare_data(grades, goals, home_match, no_players, no_days):
    """
        gets player grades of multiple games (the more, the better) and calculates numpy array for X values and y values
        
        input: 
            grades: numpy array (team, players, matchday)
            goals: numpy array (first half goals, second half goals, first half goals (2nd team), second half goals (2nd team))
            home_match: numpy array (1,0) for home or outside match
            no_players: int of represented best players in the model
            no_days: int of represented days in the model
        output: list of numpy arrays [X[no. days, no. players]: playergrades, y: mean team_grade, X_last: last data point for next matchday]
    """
    
    X_teams = grades.shape[0]
    X_days = grades.shape[2] - no_days
    X_goals = goals.shape[1]
    
    # no. of data points: X_teams*X_days
    # no. of features: no_days*no_players
    X = np.empty([no_days * (no_players + X_goals + 1), ])
    y = np.empty([1,])
    X_last = np.empty([X_teams, no_days * (no_players + X_goals + 1)])

    i = 0
    for team in range(X_teams):
        for day in range(X_days):
            # data from player grades
            new_point = grades[team, 0:no_players, day:day+no_days].reshape((no_players*no_days))
            # data from goals
            new_point_goals = goals[team, :, day:day+no_days].reshape((X_goals*no_days))
            #data from home matches
            # TODO: following line needs a fixture: we have a 2D array (teams x days)!
            # addition: only add the actual day to the data point!!!
            new_point_home_match = home_match[day+no_days, team]
            # combine the data sources
            new_point = np.hstack((new_point, new_point_goals, new_point_home_match))

            # classifier: +1:win, 0:even, -1:lost
            classifier = 0
            diff = goals[team, :, day+no_days][1] - goals[team, :, day+no_days][3]
            if (diff > 0):
                classifier = 1
            elif (diff < 0):
                classifier = -1
            new_target = np.array([classifier, ])
            
            if i == 0:
                # fill the first row
                X = new_point
                y = new_target
            else:
                # add the next data point
                X = np.vstack([X, new_point])
                y = np.vstack([y, new_target])

            i = i + 1

        if team == 0:
            # data point for the first team
            X_last = np.hstack((
                grades[team, 0:no_players, X_days:X_days+no_days].reshape((no_players*no_days)),
                goals[team, :, X_days:X_days+no_days].reshape((X_goals*no_days))
                ))
        else:
            # add data point for the next team
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
        returns pandas dataframe: 
        team1, result1, proba1, team2, result2, proba2
    """

    # predict the outcomes
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    pred_matches = []
    for match in matches:
        pred_matches.append([teams[match[0]], y_pred[match[0]], np.max(y_pred_proba[match[0]]), teams[match[1]], y_pred[match[1]], np.max(y_pred_proba[match[1]])])
    
    # fill the dataframe with array data and return the output
    df_pred_matches = pd.DataFrame(pred_matches, columns=['team 1', 'pred. 1', 'prob. 1', 'team 2', 'pred. 2', 'prob. 2'])
    return df_pred_matches

# TODO: max_percentage not used - fixture nesessary!
def set_bets(capital, percentage_cash, max_percentage, pred_matches):
    """
    calculates the individual stocks based on input parameters

    input:
    capital: double, currents assets in betting platform
    percentage_cash: double, percentage which is not used bets
    max_percentage: double, maximum percentage per title (of capital) 
    pred_matches: pandas dataframe with predicted matches (from multiple models)

    output:
    bets: pandas dataframe with matches with recommended tips
    """

    #split up the matches; some might be called more than once
    pred_matches["weight"] = pred_matches["prob. 1"] + pred_matches["prob. 2"]
    first = pred_matches.drop_duplicates(subset=["team 1", "pred. 1"])
    duplicates = pred_matches[pred_matches.duplicated(subset=["team 1", "pred. 1"])]

    bets = pd.concat([first, duplicates]).groupby(['team 1', 'pred. 1', 'team 2', 'pred. 2']).sum().reset_index()
    
    bets["weight"] = bets["weight"] / bets["weight"].sum()
    max_weight = (1 - percentage_cash) * max_percentage

    working_capital = capital * (1 - percentage_cash)

    # round to next .5 value (looks more human-like)
    bets["bet"] = round(2 * bets["weight"] * working_capital) / 2

    return bets

if __name__ == "__main__":
    # maximum no. of match days (eg. bundesliga == 34, Premier League == 38)
    MAX_DAYS = 34
    LEAGUE = "bundesliga" # 2-bundesliga or bundesliga

    # following saisons are considered (strings need to correspond to domains)
    # keep the right order or things get messy
    saisons = ["2019-20","2020-21","2021-22"]
    #saisons = ["2021-22",]

    # next match day (predictions are made for this day but no data is scraped)
    #next_day = 3
    next_day = 22


    # constants for calculating how much money should be betted
    capital = 22.16
    percentage_cash = 0.0
    max_percentage = 0.2

    # write input data to csv file:
    # match day, capital, betted percentage of cash, max percentage
    with open('history.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["{}".format(next_day), "{}".format(capital), "{}".format(percentage_cash), "{}".format(max_percentage)])    

    #read teamnames from teams.csv
    teams = []
    for saison in saisons:
        new_team = []
        with open(LEAGUE + "/" + saison + "_teams.csv", "r") as file:
            for line in file.readlines():
                new_team.append(line.strip("\n"))
        teams.append(new_team)

    count_grade_scrapes = 0
    count_goal_scrapes = 0

    for saison, team in zip(saisons, teams):
        for day in range(1, MAX_DAYS + 1):
            print("SCRAPING --- saison: " + saison + " - day: {}".format(day), end='\r')
            if scrape_and_save_player_grades_for_teams(LEAGUE, team, saison, day):
                count_grade_scrapes += 1
            if scrape_and_save_goals_and_home_match_for_teams(LEAGUE, team, saison, day):
                count_goal_scrapes += 1
            # stop scraping when last match was hit
            if (saison == saisons[-1] and day == next_day - 1):
                print("SCRAPING --- saison: " + saison + " - day: {} \n \
                    --- no. of web accesses: grades: {}, goals/home matches: {}".format(day, count_grade_scrapes, count_goal_scrapes))
                break

    #print("SCRAPER FINISHED SCRAPING!!!")    

    # read existing grade and goal files, start with the first one
    team_grades = np.loadtxt(LEAGUE + "/" + saisons[0] + "/1_grades.csv")
    team_goals = np.loadtxt(LEAGUE + "/" + saisons[0] + "/1_goals.csv")
    team_home_match = np.loadtxt(LEAGUE + "/" + saisons[0] + "/1_homematch.csv")
    for saison in saisons:
        for day in range(2, MAX_DAYS + 1):
            # stop when the the next match day should be predicted
            if (saison == saisons[-1] and day == next_day ):
                break

            day_grades = np.loadtxt(LEAGUE + "/" + saison + "/{}_grades.csv".format(day))
            day_goals = np.loadtxt(LEAGUE + "/" + saison + "/{}_goals.csv".format(day))
            day_home_match = np.loadtxt(LEAGUE + "/" + saison + "/{}_homematch.csv".format(day))
            team_grades = np.dstack((team_grades, day_grades))
            team_goals = np.dstack((team_goals, day_goals))
            # TODO: following line needs a fixture, we need to stack coloumn-wise per day, works temporarily
            team_home_match = np.vstack((team_home_match, day_home_match))
    # recommend next bets
    matches = scrape_matches(LEAGUE, teams[-1], saisons[-1], next_day)

    players = [8]
    #days = [1]
    days = [4]
    
    score = np.empty((len(players), len(days)))

    # use for loops for testing multiple settings    
    # loop for machine learning with Logistic regression
    for i in range(len(players)):
        for j in range(len(days)):
            # collect and prepare the data points
            data = prepare_data(team_grades, team_goals, team_home_match, players[i], days[j])
            X = data[0]
            y = data[1]
            X_last = data[2]
            #last_home_matches = np.loadtxt(LEAGUE + "/" + saisons[-1] + "/{}_homematch.csv".format(next_day))
            # append home-matches to the data array
            X_last = np.column_stack((X_last, matches[1]))

            # dimensionality reduction
            #pca = PCA(n_components=20).fit(X)
            #X_PCA = pca.transform(X)

           # Testing the model on data set
            # run testing algorithm 3 times for averaging the loss
            no_avg = 3
            for no in range(no_avg):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
                clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train.ravel())
                y_pred = clf.predict(X_test)
                score[i,j] = score[i,j] + 1 / no_avg * clf.score(X_test, y_test)

            print("-----------------------------------------------------------------------------------------")
            print("Logistic Regression: --- Matchday: {} ---".format(next_day))
            print("players: {}, days: {}".format(players[i], days[j]))
            print("score: {}".format(score[i,j]))

            #train the model for bet recommendation
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y.ravel())
            
            # run the prediction function and print the result ('raw' data)
            bets_LR = predict_match_day(clf, X_last, teams[-1], matches[0])
            bets_LR = bets_LR[bets_LR["pred. 1"] == -bets_LR["pred. 2"]]
            print(bets_LR)
            print("-----------------------------------------------------------------------------------------")

    # TODO: Refine the neural network with pytorch, temporaly implemented with quick-and-dirty sklearn
    # use for loops for testing multiple settings (for testing purposes)  
    # loop for machine learning with neural network
    for i in range(len(players)):
        for j in range(len(days)):
            data = prepare_data(team_grades, team_goals, team_home_match, players[i], days[j])
            X = data[0]
            y = data[1]
            X_last = data[2]
            #last_home_matches = np.loadtxt(LEAGUE + "/" + saisons[-1] + "/{}_homematch.csv".format(next_day))
            X_last = np.column_stack((X_last, matches[1]))

            score[i,j] = 0

            # Testing the model on data set
            # run testing algorithm 3 times for averaging the loss
            no_avg = 3
            for no in range(no_avg):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
                clf = MLPClassifier(
                    hidden_layer_sizes=(5,100), # old: (20,20)
                    activation='tanh', 
                    alpha= 0.01, 
                    random_state=0,
                    max_iter=10000
                    ).fit(X_train, y_train.ravel())
                y_pred = clf.predict(X_test)
                score[i,j] = score[i,j] + 1 / no_avg * clf.score(X_test, y_test)

            print("-----------------------------------------------------------------------------------------")
            print("Neural Network: --- Matchday: {} ---".format(next_day))
            print("players: {}, days: {}".format(players[i], days[j]))
            print("score: {}".format(score[i,j]))

            #train the model for bet recommendation
            clf = MLPClassifier(
                hidden_layer_sizes=(20,20), 
                activation='tanh', 
                alpha= 0.01, 
                random_state=0, 
                max_iter=10000
                ).fit(X, y.ravel())

            # run the prediction function and print the result ('raw' data)
            bets_NN = predict_match_day(clf, X_last, teams[-1], matches[0])
            bets_NN = bets_NN[bets_NN["pred. 1"] == -bets_NN["pred. 2"]]
            print(bets_NN)
            print("-----------------------------------------------------------------------------------------")

            bets_total = bets_LR.append(bets_NN)

            #print the recommended bets
            print("Tip recommendations")
            print("capital: {} euros, percentage of cash: {}".format(capital, percentage_cash))

            bets = set_bets(capital, percentage_cash, max_percentage, bets_total)
            print(bets.sort_values(by=["bet"]))
            print("--- betted capital: {} euros".format(bets["bet"].sum()))
            print("-----------------------------------------------------------------------------------------")


    #np.savetxt("loss.csv", res_loss)