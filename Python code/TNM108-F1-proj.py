#Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier

#Datasets
circuits = pd.read_csv('./F1Data/circuits.csv', index_col='circuitId')
constructor_results = pd.read_csv('./F1Data/constructor_results.csv', index_col='constructorResultsId')
constructor_standings = pd.read_csv('./F1Data/constructor_standings.csv', index_col='constructorStandingsId')
constructors = pd.read_csv('./F1Data/constructors.csv', index_col='constructorId')
driver_standings = pd.read_csv('./F1Data/driver_standings.csv', index_col='driverStandingsId')
drivers = pd.read_csv('./F1Data/drivers.csv', index_col='driverId')
lap_times = pd.read_csv('./F1Data/lap_times.csv', index_col='raceId')
pit_stops = pd.read_csv('./F1Data/pit_stops.csv', index_col='raceId')
qualifying = pd.read_csv('./F1Data/qualifying.csv', index_col='qualifyId')
races = pd.read_csv('./F1Data/races.csv', index_col='raceId')
results = pd.read_csv('./F1Data/results.csv', index_col='resultId')
seasons = pd.read_csv('./F1Data/seasons.csv', index_col='year')
sprint_results = pd.read_csv('./F1Data/sprint_results.csv', index_col='resultId')
status = pd.read_csv('./F1Data/status.csv', index_col='statusId')

# Profiles
profile1 = {
  'FavDriver': 'Max Verstappen',
  'FavConst' : 'Red Bull',
  'FavCircuit' : 'Red Bull Ring',
  'FavRace'  : 'Belgian Grand Prix 2022'
}
profile2 = {
  'FavDriver': 'Michael Schumacher',
  'FavConst' : 'Ferrari',
  'FavCircuit' : 'Hockenheimring',
  'FavRace'  : 'Belgian Grand Prix 1998'
}
profile3 = {
  'FavDriver': 'Fernando Alonso',
  'FavConst' : 'Renault',
  'FavCircuit' : 'Circuit de Barcelona-Catalunya',
  'FavRace'  : 'Malaysian Grand Prix 2004'
}
profile4 = {
  'FavDriver': 'Lewis Hamilton',
  'FavConst' : 'Mercedes',
  'FavCircuit' : 'Silverstone Circuit',
  'FavRace'  : 'Turkish Grand Prix 2020'
}
profile5 ={
  'FavDriver': 'Al Pease',
  'FavConst' : 'Sauber',
  'FavCircuit' : 'Autódromo José Carlos Pace',
  'FavRace'  : 'Monaco Grand Prix 1996'
}

profiles = [profile1,profile2,profile3, profile4,profile5] # Collect profiles to array

# Functions
def createProfiles(profiles): # Takes the array profiles, containing all profiles
    profileDf = pd.DataFrame()
    indexCounter = 1                # Used in the subsequent for loop
    # Convert profile parameters to relevant indices
    for i in profiles:
    # Driver
        FavoriteDriver = i['FavDriver'].split(' ',1)
        driverMask = (drivers['forename'].str.contains(FavoriteDriver[0])) & (drivers['surname'].str.contains(FavoriteDriver[1]))
        foundDriver = drivers[driverMask]
        profileDf.loc[indexCounter, 'driverId'] = int(foundDriver.index[0])
        profileDf.loc[indexCounter, 'driverName'] = foundDriver['forename'].iloc[0] + " " + foundDriver['surname'].iloc[0]
        profileDf.loc[indexCounter, 'driverDob'] = foundDriver['dob'].iloc[0]

        # Constructor
        FavoriteConst = i['FavConst']
        constMask = (constructors['name'].str.contains(FavoriteConst))
        foundConstructor = constructors[constMask]
        profileDf.loc[indexCounter, 'constructorId'] = int(foundConstructor.index[0])
        profileDf.loc[indexCounter, 'constructorName'] =  foundConstructor['name'].iloc[0]
        profileDf.loc[indexCounter, 'constructorCountry'] = foundConstructor['nationality'].iloc[0]

        # Circuit
        FavoriteCircuit = i['FavCircuit']
        circuitMask = (circuits['name'].str.contains(FavoriteCircuit))
        foundCircuit = circuits[circuitMask]
        profileDf.loc[indexCounter, 'circuitId'] = int(foundCircuit.index[0])
        profileDf.loc[indexCounter, 'circuitName'] = foundCircuit['name'].iloc[0]
        profileDf.loc[indexCounter, 'circuitCountry'] = foundCircuit['country'].iloc[0]

        # Race
        FavoriteRaceText = i['FavRace'].rstrip('0123456789')
        FavortieRaceText = FavoriteRaceText.rstrip('Grand Prix')
        FavoriteRaceYear = i['FavRace'][len(FavoriteRaceText):]
        raceMask = (races['name'].str.contains(FavortieRaceText) & (races['year'] == int(FavoriteRaceYear)))
        foundRace = races[raceMask]
        profileDf.loc[indexCounter, 'raceId'] = int(foundRace.index[0])
        profileDf.loc[indexCounter, 'raceYear'] = foundRace['year'].iloc[0]
        profileDf.loc[indexCounter, 'raceName'] = foundRace['name'].iloc[0]

        # Advance counter
        indexCounter = indexCounter + 1
    return profileDf
def filterRaces():
    # Filter time and unimportant dates
    FilteredRaces = races.drop(columns=['time','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date', 'fp3_time', 'quali_date', 'quali_time','sprint_date','sprint_time'])
    
    # Add qualifying and race order for all races. Includes drivers and constructors
    driversQuali = qualifying.groupby('raceId',group_keys=True)['driverId'].apply(list)
    constructorsQuali = qualifying.groupby('raceId',group_keys=True)['constructorId'].apply(list)
    driversPosition = results.groupby('raceId', group_keys=True)['driverId'].apply(list)
    constructorsPosition = results.groupby('raceId',group_keys=True)['constructorId'].apply(list)

    # Add new columns for qualifying and race order to FilteredRaces
    FilteredRaces['driverQuali'] = driversQuali
    FilteredRaces['constructorsQuali'] = constructorsQuali
    FilteredRaces['driversPosition'] = driversPosition
    FilteredRaces['constructorsPosition'] = constructorsPosition

    return FilteredRaces
def calcCosineSimularity(raceDf):
    driverCosim=[]
    constrCosim=[]

    for i in races.index:
        if (not raceDf['driverQuali'].isnull().loc[i]) & (not raceDf['driversPosition'].isnull().loc[i]):
            if len(raceDf['driverQuali'].loc[i]) == len(raceDf['driversPosition'].loc[i]):
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i]).reshape(1,-1), np.array(raceDf['driversPosition'].loc[i]).reshape(1,-1)))
            elif len(raceDf['driverQuali'].loc[i]) > len(raceDf['driversPosition'].loc[i]):
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i])[: len(raceDf['driversPosition'].loc[i])].reshape(1,-1), np.array(raceDf['driversPosition'].loc[i]).reshape(1,-1)))
            else:
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i]).reshape(1,-1), np.array(raceDf['driversPosition'].loc[i])[: len(raceDf['driverQuali'].loc[i])].reshape(1,-1)))
        else:
            driverCosim.append(np.nan)

        if (not raceDf['constructorsQuali'].isnull().loc[i]) & (not raceDf['constructorsPosition'].isnull().loc[i]):
            if len(raceDf['constructorsQuali'].loc[i]) == len(raceDf['constructorsPosition'].loc[i]):
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i]).reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i]).reshape(1,-1)))
            elif len(raceDf['constructorsQuali'].loc[i]) > len(raceDf['constructorsPosition'].loc[i]):
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i])[: len(raceDf['constructorsPosition'].loc[i])].reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i]).reshape(1,-1)))
            else:
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i]).reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i])[: len(raceDf['constructorsQuali'].loc[i])].reshape(1,-1)))
        else:
            constrCosim.append(np.nan)

    raceDf['driverCosim'] = driverCosim
    raceDf['constrCosim'] = constrCosim


    return raceDf
def calcDriverClimb(raceDf):
    podiumClimb = []

    for i in raceDf.index:
        if (not raceDf['driverQuali'].isnull().loc[i]) & (not raceDf['driversPosition'].isnull().loc[i]):
            podiumClimbRace = []

            for driverIndex, driver in enumerate(raceDf['driversPosition'].loc[i]):
                try:
                    endPosition = driverIndex
                    startPosition = raceDf['driverQuali'].loc[i].index(driver)
                    positionDiff = startPosition - endPosition
                    podiumClimbRace.append([driver, positionDiff])
                except:
                    podiumClimbRace.append(np.nan)
                    continue
            podiumClimb.append(podiumClimbRace)
        else:
            podiumClimb.append(np.nan)
    
    raceDf['podiumClimb'] = podiumClimb
    return raceDf
def calcEuclidianDist(profDf,raceDf):
    profile_cosim = profDf[['raceDriverCosim', 'raceConstrCosim']]
    filtered_cosim = raceDf[['driverCosim', 'constrCosim']]
    euclidianRaces = raceDf

    total_distance_vec = []
    for race in filtered_cosim.iterrows():
        total_distance = 0
        for profile in profile_cosim.iterrows():
            if not type(race[1].iloc[0]) == float: # checks if nan
                x = race[1].iloc[0][0][0] - profile[1].iloc[0] #Driver cosim
                y = race[1].iloc[1][0][0] - profile[1].iloc[1] #constr cosim
                total_distance += math.sqrt(pow(x,2) + pow(y,2))
        if(total_distance == 0):
            total_distance = np.nan
        total_distance_vec.append(total_distance/len(profDf))
    euclidianRaces['TotalDistance'] = total_distance_vec
    return euclidianRaces
def calcEuclidianDistWeighted(profDf,raceDf):
    profile_cosim = profDf[['raceDriverCosim', 'raceConstrCosim', 'driverId']]
    filtered_cosim = raceDf[['driverCosim', 'constrCosim', 'podiumClimb', 'driversPosition']]
    euclidianRacesW = raceDf

    total_distance_vec = []
    for race in filtered_cosim.iterrows():
        total_distance = 0
        for profile in profile_cosim.iterrows():
            if not type(race[1].iloc[0]) == float: # checks if nan
                x = race[1].iloc[0][0][0] - profile[1].iloc[0] #Driver cosim
                y = race[1].iloc[1][0][0] - profile[1].iloc[1] #constr cosim

                winnerWeight = race[1].iloc[2]

                desiredDriver = profile[1].iloc[2] # wanted driver
                driversPosition = race[1].iloc[3]
                winningDrivers = driversPosition
                z = 100
                for index, driver in enumerate(winningDrivers):
                    if driver == desiredDriver:
                         if not type(winnerWeight[index]) == float:
                            driverWeight = winnerWeight[index]
                            z -= (driverWeight[1]+1) * 10/ (index+1)

                total_distance += math.sqrt(pow(x,2) + pow(y,2) + pow(z,2))
        if(total_distance == 0):
            total_distance = np.nan
        total_distance_vec.append(total_distance)
    euclidianRacesW['TotalDistance'] = total_distance_vec
    return euclidianRacesW
def calcKNN(profDf, raceDf):
    profile_cosim = profDf[['raceDriverCosim', 'raceConstrCosim', 'driverId']]
    filtered_cosim = raceDf[['driverCosim', 'constrCosim', 'podiumClimb', 'driversPosition']]
    knnRaces = raceDf
    race_x = []
    race_y = []
    race_class = []
    racesWithoutFloat = []

    for race in filtered_cosim.iterrows():
        z = 0
        if not type(race[1].iloc[0]) == float:
            racesWithoutFloat.append(race)
            for profile in profile_cosim.iterrows():
                if not type(race[1].iloc[0]) == float: # checks if nan
                    winnerWeight = race[1].iloc[2]
                    desiredDriver = profile[1].iloc[2] # wanted driver
                    driversPosition = race[1].iloc[3]
                    winningDrivers = driversPosition

                    for index, driver in enumerate(winningDrivers):
                        if driver == desiredDriver:
                            if not type(winnerWeight[index]) == float:
                                driverWeight = winnerWeight[index]
                                z += abs(driverWeight[1]) * 10/ (index+1)
        if z == 0:
            race_x.append(0)
            race_y.append(0)
            race_class.append(0)
            continue
        race_x.append(race[1].iloc[0][0][0])
        race_y.append(race[1].iloc[1][0][0])
        if z >= 30:
            race_class.append(3)
            continue
        if z >= 10:
            race_class.append(2)
            continue
        if z < 10:
            race_class.append(1)
    
    knnRaces['RaceClass'] = race_class

    data = list(zip(race_x, race_y))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, race_class)

    average_x = 0
    average_y = 0
    for profile in profile_cosim.iterrows():
        new_x = profile[1].iloc[0]
        new_y = profile[1].iloc[1]
        average_x += new_x
        average_y += new_y
    average_x /= (profile_cosim.size / 3)
    average_y /= (profile_cosim.size / 3)
    new_point = [(average_x, average_y)]
    
    prediction = knn.predict(new_point)
    total_distance_vec = []
    for index, race in enumerate(filtered_cosim.iterrows()):
        if race_class[index] == prediction:
            total_distance = 0

            for profile in profile_cosim.iterrows():
                if not type(race[1].iloc[0]) == float: # checks if nan
                    x = race[1].iloc[0][0][0] - profile[1].iloc[0] #Driver cosim
                    y = race[1].iloc[1][0][0] - profile[1].iloc[1] #constr cosim

                    winnerWeight = race[1].iloc[2]
                    desiredDriver = profile[1].iloc[2] # wanted driver

                    z = 100
                    for index, driver in enumerate(winningDrivers):
                        if driver == desiredDriver:
                            if not type(winnerWeight[index]) == float:
                                driverWeight = winnerWeight[index]
                                if index == 0:
                                    z -= (driverWeight[1]+1) * 10
                                else:
                                    z -= (driverWeight[1]+1) * 10 / (index)
                    total_distance += math.sqrt(pow(x,2) + pow(y,2) + pow(z,2))
            total_distance_vec.append(total_distance)
        else:
            total_distance_vec.append(100000)            
    knnRaces['TotalDistance'] = total_distance_vec
    return knnRaces
def getDriverCosim(raceId,raceDf):
  driverCosine = raceDf['driverCosim'].loc[raceId]
  return driverCosine
def getConstructorCosim(raceId,raceDf):
  constructorCosine = raceDf['constrCosim'].loc[raceId]
  return constructorCosine
def getPodiumClimb(raceId,raceDf):
  podiumClimb = raceDf['podiumClimb'].loc[raceId]
  return podiumClimb
def expandProfiles(profDf,raceDf):
    profDf["racePodiumClimb"] = pd.Series(dtype='object',index=profDf.index) # Avoids error when assigning a list to a singular cell
    for i in profDf.index:
        profDf.loc[i,'raceDriverCosim'] = getDriverCosim(profDf.loc[i,'raceId'],raceDf)
        profDf.loc[i,'raceConstrCosim'] = getConstructorCosim(profDf.loc[i,'raceId'],raceDf)
        profDf.at[i,'racePodiumClimb'] = getPodiumClimb(profDf.loc[i,'raceId'],raceDf)
    return profDf
def getRaceByIndex(raceId,raceDf):
  raceInfo = raceDf.loc[raceId]
  raceInfo['circuitName'] = circuits.loc[raceInfo.circuitId,'name']
  return raceInfo
def getDriverByIndex(driverId):
  driverInfo = drivers.loc[driverId]
  return driverInfo
def getConstructorByIndex(constructorId):
  constructorInfo = constructors.loc[constructorId]
  return constructorInfo
def drawScatterPlot(filtRaces):
    plt.scatter(filtRaces['constrCosim'], filtRaces['driverCosim'],c=filtRaces.index, cmap='twilight_shifted')
    plt.show()
    return


#------------------------------------Main-------------------------------------------------
# Following function calls needs to be executed in order
profDf = createProfiles(profiles)                     # Generate the profiles dataframe
filtRaces = filterRaces()                                   # Filter races.cvs and add relevant information
filtRaces = calcCosineSimularity(filtRaces)      # Calculate the cosine similarity for all races between qualifying and race result
filtRaces = calcDriverClimb(filtRaces)              # Calculate the climb for all drivers in all races
profDf = expandProfiles(profDf, filtRaces)      # Add cosine similarity and driver climb from the profiles favorite races

#--------------------------------Diffrent predictions----------------------------------------
# Diffrent predictions are made with diffrent algorithms.
# Then the diffrent predictions are sorted and shortened to n entries, then printed to the console
n = 5

racesEuclidianDist = calcEuclidianDist(profDf,filtRaces)    # Calculate the mean euclidan distance to all races for the profiles, this is without weights
racesEuclidianDist = racesEuclidianDist.sort_values(by='TotalDistance').head(n)

print('Recomended races with unweighted euclidian distance: \n')
print(racesEuclidianDist['name'] + " " + racesEuclidianDist['date'])
print('With the distances: \n')
print(racesEuclidianDist['TotalDistance'])

print("\n")

racesEuclidianDistW = calcEuclidianDistWeighted(profDf,filtRaces)   # Calculate the euclidan distance but with weights depending on favorite drivers
racesEuclidianDistW = racesEuclidianDistW.sort_values(by='TotalDistance').head(n)

print('Recomended races with weighted euclidian distance: \n')
print(racesEuclidianDistW['name'] + " " + racesEuclidianDistW['date'])
print('With the distances: \n')
print(racesEuclidianDistW['TotalDistance'])

print("\n")

racesKNN = calcKNN(profDf,filtRaces)   # Calculate the KNN neighbours
racesKNN = racesKNN.sort_values(by='TotalDistance').head(n)

print('Recomended races with KNN: \n')
print(racesKNN['name'] + " " + racesKNN['date'])
print('With the distances: \n')
print(racesKNN['TotalDistance'])
